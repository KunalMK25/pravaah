"""
Drainage feature generation for the Flood Risk Zonation System.

Two functions are provided:

generate_drainage_proxy(grid, water_bodies, cell_size_m)
    PRIMARY.  Derives a drainage infrastructure availability proxy from
    mapped OSM linestrings (drain, canal, stream, river ways) that are
    already present in the ``water_bodies`` GeoDataFrame fetched by the
    pipeline.  No additional network calls are required.

    The resulting ``DrainageDataset`` has ``source="osm_proxy"``.

    IMPORTANT: this is a *proxy*, not a hydraulic capacity measurement.
    Municipal hydraulic capacity measurements are unavailable; the system
    uses mapped drainage infrastructure as a spatial drainage proxy.

    Proxy formula (per cell):
        density_score_drain   = normalised total length of drain/ditch
                                ways within search_radius_m
        density_score_canal   = normalised total length of canal ways
                                within search_radius_m
        proximity_score_river = 1 - min(dist_to_stream_or_river, MAX_DIST)
                                    / MAX_DIST
        drainage_proxy = clip(
            0.50 * density_score_drain
          + 0.30 * density_score_canal
          + 0.20 * proximity_score_river,
            0, 1
        )

generate_synthetic_drainage(grid, seed)
    FALLBACK.  Retained for backward compatibility and offline/test use.
    Returns DrainageDataset with source="synthetic_fallback".
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

from flood_risk_zonation.models import DrainageDataset

logger = logging.getLogger(__name__)

# Waterway tags treated as primary drainage infrastructure
_DRAIN_TYPES = {"drain", "ditch", "culvert"}
# Secondary channels (may serve drainage or irrigation)
_CANAL_TYPES = {"canal"}
# Natural discharge pathways (lower weight, proximity signal only)
_RIVER_TYPES = {"river", "stream", "tidal_channel"}

# Weight vector for the three proxy components (must sum to 1.0)
_W_DRAIN = 0.50
_W_CANAL = 0.30
_W_RIVER = 0.20

# Beyond this distance (m) a stream/river contributes zero proximity score
_MAX_RIVER_DIST_M = 5_000.0

# Search radius as a multiple of cell_size_m for density computation
_RADIUS_MULT = 2.0


def _synthetic_drainage_fallback(
    grid: gpd.GeoDataFrame,
    seed: int = 42,
) -> np.ndarray:
    """
    Assign synthetic drainage scores [0, 1] to grid cells.

    Scores are inversely correlated with population density (if available)
    to simulate urban impervious surface effects.

    Returns
    -------
    np.ndarray
        1-D float32 array of scores in [0, 1].
    """
    n = len(grid)
    rng = np.random.default_rng(seed)

    if "population_density" in grid.columns:
        pop = grid["population_density"].fillna(0).values.astype(np.float64)
        pop_max = pop.max()
        if pop_max > 0:
            pop_norm = pop / pop_max
        else:
            pop_norm = np.zeros(n)
        noise = rng.uniform(0, 0.1, n)
        scores = np.clip(1.0 - pop_norm * 0.8 + noise, 0.0, 1.0).astype(np.float32)
    else:
        scores = rng.uniform(0.2, 1.0, n).astype(np.float32)

    return scores


def _extract_drainage_lines(
    water_bodies: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame | None:
    """
    Filter *water_bodies* to linestring-type drainage features and tag each
    row with ``_dtype``: ``"drain"``, ``"canal"``, or ``"river"``.

    Returns None if no qualifying linestrings are found.
    """
    if water_bodies is None or len(water_bodies) == 0:
        return None

    lines = []
    for _, row in water_bodies.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type not in {"LineString", "MultiLineString"}:
            continue  # only linestrings carry drainage density information
        wtype = str(row.get("water_type", "")).lower().strip()
        if wtype in _DRAIN_TYPES:
            dtype = "drain"
        elif wtype in _CANAL_TYPES:
            dtype = "canal"
        elif wtype in _RIVER_TYPES:
            dtype = "river"
        elif "drain" in wtype or "ditch" in wtype:
            dtype = "drain"
        elif "canal" in wtype:
            dtype = "canal"
        elif "river" in wtype or "stream" in wtype:
            dtype = "river"
        else:
            continue  # not a drainage feature
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        lines.append({"geometry": geom, "_dtype": dtype})

    if not lines:
        return None

    crs = water_bodies.crs if water_bodies.crs is not None else "EPSG:4326"
    return gpd.GeoDataFrame(lines, crs=crs)


def generate_drainage_proxy(
    grid: gpd.GeoDataFrame,
    water_bodies: gpd.GeoDataFrame | None,
    cell_size_m: float = 500.0,
    seed: int = 42,
) -> DrainageDataset:
    """
    Derive a drainage infrastructure availability proxy from OSM linestrings.

    Uses ``drain``, ``canal``, ``stream``, and ``river`` way geometries
    already present in *water_bodies* (fetched by the pipeline — no extra
    network calls).  Falls back to synthetic scores when no linestrings are
    available.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid GeoDataFrame with ``cell_id``, ``centroid_lat``, ``centroid_lon``.
    water_bodies : gpd.GeoDataFrame | None
        OSM water feature GeoDataFrame (polygons and linestrings).
        If None or empty, falls back to synthetic drainage.
    cell_size_m : float
        Grid cell size in metres.  Search radius = 2 x cell_size_m.
    seed : int
        Fallback random seed.

    Returns
    -------
    DrainageDataset
        ``source="osm_proxy"`` when linestrings are found;
        ``source="synthetic_fallback"`` when no linestrings are available.

    Notes
    -----
    This feature is a **proxy**, not a hydraulic capacity measurement.
    Presence and density of mapped drainage infrastructure is used as a
    spatial surrogate for drainage adequacy.
    """
    cell_ids = list(grid["cell_id"].astype(str))
    n = len(grid)

    # ── Step 1: Extract drainage linestrings from the existing water_bodies GDF
    drain_lines = _extract_drainage_lines(water_bodies)

    if drain_lines is None or len(drain_lines) == 0:
        logger.info(
            "Drainage proxy: no OSM drainage linestrings found — "
            "using synthetic fallback."
        )
        scores = _synthetic_drainage_fallback(grid, seed=seed)
        return DrainageDataset(
            capacity_scores=scores,
            cell_ids=cell_ids,
            source="synthetic_fallback",
        )

    logger.info(
        "Drainage proxy: %d OSM linestrings (%d drain, %d canal, %d river).",
        len(drain_lines),
        int((drain_lines["_dtype"] == "drain").sum()),
        int((drain_lines["_dtype"] == "canal").sum()),
        int((drain_lines["_dtype"] == "river").sum()),
    )

    # ── Step 2: Reproject to metric CRS once (EPSG:3857, Web Mercator)
    try:
        drain_m = drain_lines.to_crs("EPSG:3857")
    except Exception as exc:
        logger.warning(
            "Drainage proxy: CRS reprojection failed (%s) — synthetic fallback.", exc
        )
        scores = _synthetic_drainage_fallback(grid, seed=seed)
        return DrainageDataset(
            capacity_scores=scores, cell_ids=cell_ids, source="synthetic_fallback"
        )

    try:
        from shapely.geometry import Point
        centroids_m = gpd.GeoSeries(
            [
                Point(lon, lat)
                for lon, lat in zip(
                    grid["centroid_lon"].values, grid["centroid_lat"].values
                )
            ],
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
    except Exception as exc:
        logger.warning(
            "Drainage proxy: centroid projection failed (%s) — synthetic fallback.", exc
        )
        scores = _synthetic_drainage_fallback(grid, seed=seed)
        return DrainageDataset(
            capacity_scores=scores, cell_ids=cell_ids, source="synthetic_fallback"
        )

    search_radius_m = _RADIUS_MULT * cell_size_m

    # ── Step 3: Split linestrings by category
    mask_drain = drain_m["_dtype"] == "drain"
    mask_canal = drain_m["_dtype"] == "canal"
    mask_river = drain_m["_dtype"] == "river"

    drain_geoms = drain_m.loc[mask_drain, "geometry"].reset_index(drop=True) if mask_drain.any() else None
    canal_geoms = drain_m.loc[mask_canal, "geometry"].reset_index(drop=True) if mask_canal.any() else None
    river_geoms = drain_m.loc[mask_river, "geometry"].reset_index(drop=True) if mask_river.any() else None

    # ── Step 4: Build STRtree spatial indices (O(log M) per query)
    drain_idx = drain_geoms.sindex if drain_geoms is not None and len(drain_geoms) > 0 else None
    canal_idx = canal_geoms.sindex if canal_geoms is not None and len(canal_geoms) > 0 else None
    river_idx = river_geoms.sindex if river_geoms is not None and len(river_geoms) > 0 else None

    # ── Step 5: Per-cell raw scores
    raw_drain = np.zeros(n, dtype=np.float64)
    raw_canal = np.zeros(n, dtype=np.float64)
    raw_river = np.zeros(n, dtype=np.float64)

    for i, centroid in enumerate(centroids_m):
        buf = centroid.buffer(search_radius_m)

        # Drain density: total clipped length of drain linestrings in buffer
        if drain_idx is not None:
            cands = list(drain_idx.intersection(buf.bounds))
            if cands:
                total_len = 0.0
                for j in cands:
                    try:
                        seg = drain_geoms.iloc[j].intersection(buf)
                        if not seg.is_empty:
                            total_len += seg.length
                    except Exception:
                        pass
                raw_drain[i] = total_len

        # Canal density
        if canal_idx is not None:
            cands = list(canal_idx.intersection(buf.bounds))
            if cands:
                total_len = 0.0
                for j in cands:
                    try:
                        seg = canal_geoms.iloc[j].intersection(buf)
                        if not seg.is_empty:
                            total_len += seg.length
                    except Exception:
                        pass
                raw_canal[i] = total_len

        # River/stream proximity: inverse distance score
        if river_idx is not None:
            search_buf = centroid.buffer(_MAX_RIVER_DIST_M)
            cands = list(river_idx.intersection(search_buf.bounds))
            if cands:
                min_dist = _MAX_RIVER_DIST_M
                for j in cands:
                    try:
                        d = centroid.distance(river_geoms.iloc[j])
                        if d < min_dist:
                            min_dist = d
                    except Exception:
                        pass
                # Proximity score: 1.0 at d=0, 0.0 at d>=_MAX_RIVER_DIST_M
                raw_river[i] = max(0.0, 1.0 - min_dist / _MAX_RIVER_DIST_M)

    # ── Step 6: Robust [5th, 95th]-percentile normalisation of length scores
    def _normalise_lengths(arr: np.ndarray) -> np.ndarray:
        """Scale positive values to [0, 1] using robust percentiles."""
        finite = arr[arr > 0]
        if len(finite) == 0:
            return np.zeros_like(arr)
        p5 = float(np.percentile(finite, 5))
        p95 = float(np.percentile(finite, 95))
        if p95 <= p5:
            # All non-zero values are identical: treat as binary present/absent
            return (arr > 0).astype(np.float64)
        return np.clip((arr - p5) / (p95 - p5), 0.0, 1.0)

    norm_drain = _normalise_lengths(raw_drain)
    norm_canal = _normalise_lengths(raw_canal)
    norm_river = raw_river  # already in [0, 1] by construction

    # ── Step 7: Weighted combination → final proxy score
    combined = (
        _W_DRAIN * norm_drain
        + _W_CANAL * norm_canal
        + _W_RIVER * norm_river
    )
    scores = np.clip(combined, 0.0, 1.0).astype(np.float32)

    n_nonzero = int((scores > 0).sum())
    logger.info(
        "Drainage proxy: %d/%d cells non-zero (mean=%.3f, std=%.3f).",
        n_nonzero, n, float(scores.mean()), float(scores.std()),
    )

    return DrainageDataset(
        capacity_scores=scores,
        cell_ids=cell_ids,
        source="osm_proxy",
    )


def generate_synthetic_drainage(
    grid: gpd.GeoDataFrame,
    seed: int = 42,
) -> DrainageDataset:
    """
    Assign synthetic drainage scores [0, 1] to grid cells.

    .. deprecated::
        Prefer :func:`generate_drainage_proxy` which uses real OSM data.
        This function is retained for backward compatibility, offline mode,
        and test fixtures that do not supply water body data.

    Scores are inversely correlated with population density (if available)
    to simulate urban impervious surface effects.  Higher population
    density → lower drainage capacity heuristic.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid GeoDataFrame.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DrainageDataset
        Per-cell synthetic drainage scores; ``source="synthetic_fallback"``.
    """
    cell_ids = list(grid["cell_id"].astype(str))
    scores = _synthetic_drainage_fallback(grid, seed=seed)
    return DrainageDataset(
        capacity_scores=scores,
        cell_ids=cell_ids,
        source="synthetic_fallback",
    )

