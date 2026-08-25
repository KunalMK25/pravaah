"""
PRAVAAH-AI — Spatial Red / Yellow / Green Zone Classifier.

METHODOLOGY (declared, transparent):
────────────────────────────────────────────────────────────────────────────
Three-tier operational spatial zone system derived from, but distinct from,
the underlying ML hazard model.  The two representations are kept separate:

  underlying_hazard_class  — the ML model output (High / Medium / Low / Water)
  spatial_zone             — the operational attention class (RED / YELLOW / GREEN / WATER)

ZONE DEFINITIONS:
  RED    — primary hazard zone: cell whose underlying hazard class is "High".
           Based directly on the ML model's risk_class. Not proximity-derived.

  YELLOW — secondary attention zone: cell that does not itself meet the
           primary hazard threshold (not RED, not Water) but lies within
           the 8-neighbour footprint of at least one RED cell, OR whose
           underlying class is "Medium" (moderate model-confirmed hazard).
           Purpose: precautionary monitoring, secondary exposure boundary.
           A YELLOW cell is NOT a confirmed flood cell. The underlying
           hazard_class is preserved unchanged.

  GREEN  — lower-risk area / potential safe zone: cell that is not RED,
           not YELLOW, not Water.  May be considered for relocation
           candidate evaluation, but does NOT automatically mean
           "safe" or "official evacuation shelter".  The underlying
           risk_score and risk_class are preserved.

  WATER  — permanent water body (inherits "Water" risk_class directly).
           Cannot be a relocation candidate.

ADJACENCY LOGIC:
  8-neighbour adjacency is used because grid cells form a raster-like
  regular grid in WGS84 degree space.  A RED cell influences all 8
  surrounding neighbours (N, NE, E, SE, S, SW, W, NW).

  Implementation: for each RED cell we identify its positional index
  in a 2-D coordinate grid and mark the 8 surrounding index positions.
  We use numpy integer grid indices rather than spatial intersection
  for performance on grids up to 100 000 cells.

  The Medium-class YELLOW override ensures that model-confirmed medium-
  risk cells are never downgraded to GREEN even if they happen not to
  be adjacent to a RED cell in the sample grid.

IMPORTANT:
  classify_spatial_zones() adds a "spatial_zone" column to the scored_grid
  GeoDataFrame.  It does NOT modify "risk_class", "risk_score", or any
  other existing column.  The original hazard information is fully preserved.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Zone constants ─────────────────────────────────────────────────────────────
ZONE_RED    = "RED"
ZONE_YELLOW = "YELLOW"
ZONE_GREEN  = "GREEN"
ZONE_WATER  = "WATER"

# ── Colours used in map rendering ─────────────────────────────────────────────
ZONE_COLOR_MAP = {
    ZONE_RED:    "#c0392b",   # strong red
    ZONE_YELLOW: "#f39c12",   # amber / orange-yellow
    ZONE_GREEN:  "#27ae60",   # medium green
    ZONE_WATER:  "#2980b9",   # blue
}

# ── Map from existing risk_class to default zone (before adjacency) ───────────
_INITIAL_ZONE: dict[str, str] = {
    "High":   ZONE_RED,
    "Medium": ZONE_YELLOW,   # medium-risk cells are YELLOW by model class
    "Low":    ZONE_GREEN,
    "Water":  ZONE_WATER,
}


def _build_coordinate_index(
    lats: np.ndarray,
    lons: np.ndarray,
    n_lat_bins: int,
    n_lon_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map continuous lat/lon centroids onto a 2-D integer grid index.

    Returns
    -------
    row_idx : np.ndarray of int  (0 … n_lat_bins-1)
    col_idx : np.ndarray of int  (0 … n_lon_bins-1)
    """
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if lat_range == 0:
        row_idx = np.zeros(len(lats), dtype=int)
    else:
        row_idx = np.clip(
            ((lats - lat_min) / lat_range * (n_lat_bins - 1)).astype(int),
            0, n_lat_bins - 1,
        )

    if lon_range == 0:
        col_idx = np.zeros(len(lons), dtype=int)
    else:
        col_idx = np.clip(
            ((lons - lon_min) / lon_range * (n_lon_bins - 1)).astype(int),
            0, n_lon_bins - 1,
        )

    return row_idx, col_idx


def classify_spatial_zones(
    scored_grid: gpd.GeoDataFrame,
    adjacency: str = "8-neighbour",
    medium_is_yellow: bool = True,
) -> gpd.GeoDataFrame:
    """
    Derive the RED / YELLOW / GREEN / WATER spatial attention zone for every
    grid cell and add a ``spatial_zone`` column to the GeoDataFrame.

    The function NEVER modifies ``risk_class``, ``risk_score``, or any
    other existing column.  All original hazard information is preserved.

    Parameters
    ----------
    scored_grid : gpd.GeoDataFrame
        Phase 1 pipeline output.  Must have a ``risk_class`` column.
        Should also have ``centroid_lat`` and ``centroid_lon`` columns;
        if absent, they are derived from ``geometry.centroid``.
    adjacency : str
        ``"8-neighbour"`` (default) or ``"4-neighbour"``.
        8-neighbour marks all 8 surrounding cells of each RED cell as
        candidate YELLOW cells; 4-neighbour marks only N/S/E/W.
    medium_is_yellow : bool
        If True (default), cells whose underlying ``risk_class`` is
        "Medium" are always assigned ZONE_YELLOW regardless of adjacency.
        This ensures model-confirmed medium-risk cells are not downgraded
        to GREEN simply because no RED cell happens to be adjacent.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of ``scored_grid`` with a new ``spatial_zone`` column.
        The original grid is not mutated.

    Notes
    -----
    Spatial attention classes:
      RED    → underlying risk_class == "High"
      YELLOW → (a) adjacent to a RED cell (8-neighbour) or
               (b) underlying risk_class == "Medium" [when medium_is_yellow=True]
      GREEN  → not RED, not YELLOW, not Water
      WATER  → underlying risk_class == "Water"
    """
    result = scored_grid.copy()

    if "risk_class" not in result.columns:
        logger.warning("scored_grid has no 'risk_class' column — all zones set to GREEN.")
        result["spatial_zone"] = ZONE_GREEN
        return result

    # ── Step 1: Initial zone assignment from risk_class ────────────────────────
    zones = result["risk_class"].map(
        lambda rc: _INITIAL_ZONE.get(str(rc), ZONE_GREEN)
    ).values.copy()
    # zones is a numpy object array of zone strings

    # ── Step 2: Build 2-D grid index for adjacency computation ────────────────
    if "centroid_lat" in result.columns and "centroid_lon" in result.columns:
        lats = result["centroid_lat"].values.astype(float)
        lons = result["centroid_lon"].values.astype(float)
    else:
        centroids = result.geometry.centroid
        lats = centroids.y.values.astype(float)
        lons = centroids.x.values.astype(float)

    n_cells = len(result)
    if n_cells == 0:
        result["spatial_zone"] = zones
        return result

    # Estimate grid dimensions from unique lat/lon values
    # (accounts for non-square bboxes and different cell sizes)
    unique_lats = np.unique(np.round(lats, 6))
    unique_lons = np.unique(np.round(lons, 6))
    n_lat_bins = max(1, len(unique_lats))
    n_lon_bins = max(1, len(unique_lons))

    row_idx, col_idx = _build_coordinate_index(lats, lons, n_lat_bins, n_lon_bins)

    # Build a lookup: (row, col) → list of cell array-positions
    cell_pos_map: dict[tuple[int, int], list[int]] = {}
    for pos in range(n_cells):
        key = (int(row_idx[pos]), int(col_idx[pos]))
        cell_pos_map.setdefault(key, []).append(pos)

    # ── Step 3: Mark neighbours of RED cells as YELLOW ────────────────────────
    # Collect all RED-cell grid positions first, then mark neighbours
    if adjacency == "4-neighbour":
        _offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # 8-neighbour (default) — includes diagonals
        _offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),           ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

    red_positions: set[int] = set()
    for pos in range(n_cells):
        if zones[pos] == ZONE_RED:
            red_positions.add(pos)

    yellow_candidates: set[int] = set()
    for pos in red_positions:
        r, c = int(row_idx[pos]), int(col_idx[pos])
        for dr, dc in _offsets:
            nb_key = (r + dr, c + dc)
            if nb_key in cell_pos_map:
                for nb_pos in cell_pos_map[nb_key]:
                    if zones[nb_pos] not in (ZONE_RED, ZONE_WATER):
                        yellow_candidates.add(nb_pos)

    for pos in yellow_candidates:
        zones[pos] = ZONE_YELLOW

    # ── Step 4: Medium override ────────────────────────────────────────────────
    # Any Medium-class cell that was not already RED/WATER/YELLOW is upgraded
    # to YELLOW to preserve the model-derived risk signal.
    if medium_is_yellow:
        risk_classes = result["risk_class"].values
        for pos in range(n_cells):
            if str(risk_classes[pos]) == "Medium" and zones[pos] == ZONE_GREEN:
                zones[pos] = ZONE_YELLOW

    result["spatial_zone"] = zones
    n_red    = int((zones == ZONE_RED).sum())
    n_yellow = int((zones == ZONE_YELLOW).sum())
    n_green  = int((zones == ZONE_GREEN).sum())
    n_water  = int((zones == ZONE_WATER).sum())
    logger.info(
        "Spatial zones: RED=%d  YELLOW=%d  GREEN=%d  WATER=%d  (total=%d, adjacency=%s)",
        n_red, n_yellow, n_green, n_water, n_cells, adjacency,
    )
    return result


def get_zone_for_habitation(
    hab_lat: float,
    hab_lon: float,
    zoned_grid: gpd.GeoDataFrame,
    n_nearest: int = 4,
) -> str:
    """
    Return the spatial zone for a habitation point by nearest-cell lookup.

    Parameters
    ----------
    hab_lat, hab_lon : float
        Habitation coordinates.
    zoned_grid : gpd.GeoDataFrame
        Grid with ``spatial_zone`` column (output of classify_spatial_zones).
    n_nearest : int
        Number of nearest cells to inspect; the most severe zone wins.

    Returns
    -------
    str
        ZONE_RED | ZONE_YELLOW | ZONE_GREEN | ZONE_WATER
    """
    if "spatial_zone" not in zoned_grid.columns:
        return ZONE_GREEN

    from math import cos, radians

    if "centroid_lat" in zoned_grid.columns:
        glat = zoned_grid["centroid_lat"].values.astype(float)
        glon = zoned_grid["centroid_lon"].values.astype(float)
    else:
        centroids = zoned_grid.geometry.centroid
        glat = centroids.y.values.astype(float)
        glon = centroids.x.values.astype(float)

    dlat = glat - hab_lat
    dlon = (glon - hab_lon) * cos(radians(hab_lat))
    dist_sq = dlat ** 2 + dlon ** 2
    n = min(n_nearest, len(dist_sq))
    idx = np.argpartition(dist_sq, n - 1)[:n]

    priority = {ZONE_RED: 0, ZONE_YELLOW: 1, ZONE_GREEN: 2, ZONE_WATER: 3}
    zones_nearby = [str(zoned_grid.iloc[i]["spatial_zone"]) for i in idx]
    return min(zones_nearby, key=lambda z: priority.get(z, 9))
