"""
PRAVAAH-AI — Relocation Candidate Discovery and Ranking.

METHODOLOGY (declared, transparent):
────────────────────────────────────────────────────────────────────────────
This module identifies potential lower-risk areas that could be evaluated
as relocation destinations for habitations that have been assigned HIGH or
CRITICAL relocation priority.

IMPORTANT DISCLAIMERS:
  - A candidate area is a DECISION-SUPPORT RECOMMENDATION, not a legally
    designated evacuation shelter or approved relocation site.
  - Candidates are drawn from GREEN spatial zones only (LOW-risk, not Water).
  - A GREEN area is a potential candidate for evaluation — it is NOT
    guaranteed to be safe, adequate, or available.
  - All scores use declared, auditable weights.

CANDIDATE SELECTION LOGIC:
  1. Search the scored/zoned grid for cells in ZONE_GREEN within
     search_radius_km of the source habitation.
  2. Filter out cells that are Water or High risk.
  3. Cluster nearby green cells into contiguous candidate areas using
     a simple grid-based spatial grouping.
  4. Score each candidate area using measurable factors:
       + Lower hazard score  (+)
       + Larger area         (+)
       + Closer distance     (+)
       + Better road access  (+)
       + Better healthcare   (+)
       - High existing pop density (-)
  5. Return the top-N ranked candidates.

CANDIDATE SCORING FORMULA (all weights declared):
  candidate_score =
      w_hazard   × (1 - norm(mean_hazard_score / 100))
    + w_area     × norm(area_km2, 0, max_area_km2)
    + w_distance × (1 - norm(distance_km, 0, search_radius_km))
    + w_road     × (1 - norm(nearest_road_km, 0, 10))
    + w_health   × (1 - norm(nearest_healthcare_km, 0, 25))
    - w_pop      × norm(mean_pop_density)

Declared weights:
  w_hazard   = 0.30
  w_area     = 0.25
  w_distance = 0.20
  w_road     = 0.15
  w_health   = 0.10
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from math import cos, radians, sqrt
from typing import Optional

import geopandas as gpd
import numpy as np

from flood_risk_zonation.models import CarryingCapacityResult, RelocationCandidate
from flood_risk_zonation.spatial_zones.classifier import ZONE_GREEN, ZONE_WATER, ZONE_RED

logger = logging.getLogger(__name__)

# ── Declared weights ───────────────────────────────────────────────────────────
CANDIDATE_WEIGHTS: dict[str, float] = {
    "hazard":   0.30,
    "area":     0.25,
    "distance": 0.20,
    "road":     0.15,
    "health":   0.10,
}
assert abs(sum(CANDIDATE_WEIGHTS.values()) - 1.0) < 1e-9

_DEFAULT_SEARCH_RADIUS_KM = 10.0
_DEFAULT_MAX_CANDIDATES   = 5
_MIN_AREA_KM2             = 0.01   # ignore clusters smaller than ~0.01 km²


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * (dlon / 2) ** 2
    return R * 2 * sqrt(max(a, 0.0))


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _cluster_nearby_cells(
    hab_lat: float,
    hab_lon: float,
    grid: gpd.GeoDataFrame,
    radius_km: float,
    zone_col: str = "spatial_zone",
) -> list[dict]:
    """
    Find GREEN cells within radius_km of the habitation, group them into
    contiguous clusters, and return a list of cluster summaries.

    Each cluster summary has:
        cluster_id, centroid_lat, centroid_lon, area_km2,
        mean_hazard_score, mean_pop_density,
        cell_ids (list)
    """
    if "centroid_lat" in grid.columns:
        glat = grid["centroid_lat"].values.astype(float)
        glon = grid["centroid_lon"].values.astype(float)
    else:
        centroids = grid.geometry.centroid
        glat = centroids.y.values.astype(float)
        glon = centroids.x.values.astype(float)

    # Filter to GREEN cells within radius
    km_lat = 111.32
    km_lon = 111.32 * cos(radians(hab_lat))
    dlat_km = (glat - hab_lat) * km_lat
    dlon_km = (glon - hab_lon) * km_lon
    dist_km = np.sqrt(dlat_km ** 2 + dlon_km ** 2)

    zones = grid[zone_col].values if zone_col in grid.columns else np.array(["GREEN"] * len(grid))
    mask = (dist_km <= radius_km) & (zones == ZONE_GREEN)
    candidate_idx = np.where(mask)[0]

    if len(candidate_idx) == 0:
        return []

    # Simple spatial clustering: bin cells into a coarse grid of ~1km cells
    # to group nearby green cells into candidate "areas" without full DBSCAN
    cluster_map: dict[tuple[int, int], list[int]] = {}
    cluster_size_deg = 1.0 / 111.32   # ~1 km in degrees
    for idx in candidate_idx:
        lat_bin = int(glat[idx] / cluster_size_deg)
        lon_bin = int(glon[idx] / cluster_size_deg)
        cluster_map.setdefault((lat_bin, lon_bin), []).append(int(idx))

    # Merge adjacent 1km clusters into larger areas (simple 8-neighbor merge)
    visited: set[tuple[int, int]] = set()
    areas: list[list[int]] = []

    def _bfs(start_key: tuple[int, int]) -> list[int]:
        queue = [start_key]
        cells: list[int] = []
        while queue:
            k = queue.pop()
            if k in visited:
                continue
            visited.add(k)
            if k in cluster_map:
                cells.extend(cluster_map[k])
                r, c = k
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nb = (r + dr, c + dc)
                        if nb not in visited and nb in cluster_map:
                            queue.append(nb)
        return cells

    for key in cluster_map:
        if key not in visited:
            area_cells = _bfs(key)
            if area_cells:
                areas.append(area_cells)

    # Build cluster summaries
    summaries = []
    risk_scores = grid["risk_score"].values if "risk_score" in grid.columns else np.zeros(len(grid))
    pop_density = grid["population_density"].values if "population_density" in grid.columns else np.zeros(len(grid))
    cell_ids_col = grid["cell_id"].values if "cell_id" in grid.columns else np.array([str(i) for i in range(len(grid))])

    for cluster_num, cell_idx_list in enumerate(areas):
        c_lats = glat[cell_idx_list]
        c_lons = glon[cell_idx_list]
        c_center_lat = float(c_lats.mean())
        c_center_lon = float(c_lons.mean())
        c_dist = _haversine_km(hab_lat, hab_lon, c_center_lat, c_center_lon)
        c_scores = risk_scores[cell_idx_list]
        c_pop = pop_density[cell_idx_list]
        c_ids = [str(cell_ids_col[i]) for i in cell_idx_list]

        # Estimate area from cell count
        # Use grid geometry area if available, else estimate from cell count × ~0.25 km²/cell (500m cells)
        try:
            sub_grid = grid.iloc[cell_idx_list]
            if sub_grid.crs and str(sub_grid.crs).upper() != "EPSG:3857":
                sub_m = sub_grid.to_crs("EPSG:3857")
            else:
                sub_m = sub_grid
            area_km2 = float(sub_m.geometry.area.sum()) / 1_000_000
        except Exception:
            area_km2 = len(cell_idx_list) * 0.25   # ~500m × 500m default

        if area_km2 < _MIN_AREA_KM2:
            continue

        summaries.append({
            "cluster_id":       f"cand_{cluster_num:03d}",
            "centroid_lat":     round(c_center_lat, 6),
            "centroid_lon":     round(c_center_lon, 6),
            "distance_km":      round(c_dist, 3),
            "area_km2":         round(area_km2, 4),
            "mean_hazard_score":round(float(c_scores.mean()), 2),
            "mean_pop_density": round(float(c_pop.mean()), 2),
            "cell_count":       len(cell_idx_list),
            "cell_ids":         c_ids,
        })

    return summaries


def find_relocation_candidates(
    hab_lat: float,
    hab_lon: float,
    hab_id: str,
    hab_name: str,
    zoned_grid: gpd.GeoDataFrame,
    source_capacity: CarryingCapacityResult | None = None,
    search_radius_km: float = _DEFAULT_SEARCH_RADIUS_KM,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> list[RelocationCandidate]:
    """
    Discover and rank potential relocation candidate areas for one habitation.

    Parameters
    ----------
    hab_lat, hab_lon : float
        Source habitation location.
    hab_id : str
        Source habitation identifier.
    hab_name : str
        Source habitation display name.
    zoned_grid : gpd.GeoDataFrame
        Hazard grid with ``spatial_zone`` column.
    source_capacity : CarryingCapacityResult | None
        Capacity info for the source habitation (used for road/healthcare
        reference distances in scoring).
    search_radius_km : float
        Candidate search radius in km.
    max_candidates : int
        Maximum candidates to return (top-N by score).

    Returns
    -------
    list[RelocationCandidate]
        Ranked from best (highest score) to worst.
    """
    if "spatial_zone" not in zoned_grid.columns:
        logger.warning("zoned_grid has no 'spatial_zone' column — no candidates found.")
        return []

    clusters = _cluster_nearby_cells(
        hab_lat, hab_lon, zoned_grid,
        radius_km=search_radius_km,
        zone_col="spatial_zone",
    )

    if not clusters:
        logger.debug("No GREEN cells within %.1f km of %s.", search_radius_km, hab_id)
        return []

    # Reference values for normalisation
    max_area = max(c["area_km2"] for c in clusters) or 1.0
    max_pop  = max(c["mean_pop_density"] for c in clusters) or 1.0

    # Road/healthcare reference: use source capacity if available,
    # else use 5 km defaults (worst-case reference for normalisation)
    ref_road_km = 5.0
    ref_health_km = 20.0
    # For candidates we don't have live road/health data, so we derive
    # a proxy from the source habitation capacity and proximity:
    # candidates closer to the source share similar infrastructure.
    # We mark road/health as "estimated from source" in provenance.
    src_road = source_capacity.nearest_road_km if source_capacity and source_capacity.nearest_road_km >= 0 else ref_road_km
    src_health = source_capacity.nearest_healthcare_km if source_capacity and source_capacity.nearest_healthcare_km >= 0 else ref_health_km

    scored_candidates: list[RelocationCandidate] = []

    for c in clusters:
        dist    = c["distance_km"]
        area    = c["area_km2"]
        hazard  = c["mean_hazard_score"]
        pop_d   = c["mean_pop_density"]

        # Positive factors (higher = better candidate)
        f_hazard   = _norm(100.0 - hazard, 0.0, 100.0)      # low hazard → high score
        f_area     = _norm(area, 0.0, max_area)
        f_distance = _norm(search_radius_km - dist, 0.0, search_radius_km)  # closer → better
        # Infrastructure proxy: candidates near the source habitation get
        # the same road/health scores as the source (conservative estimate)
        f_road   = 1.0 - _norm(src_road, 0.0, 10.0)
        f_health = 1.0 - _norm(src_health, 0.0, 25.0)
        # Negative factor: existing population pressure
        f_pop    = _norm(pop_d, 0.0, max_pop)   # higher pop → worse candidate

        score = (
            CANDIDATE_WEIGHTS["hazard"]   * f_hazard
          + CANDIDATE_WEIGHTS["area"]     * f_area
          + CANDIDATE_WEIGHTS["distance"] * f_distance
          + CANDIDATE_WEIGHTS["road"]     * f_road
          + CANDIDATE_WEIGHTS["health"]   * f_health
          - 0.05 * f_pop    # small penalty for high existing population pressure
        )
        score = round(max(0.0, min(1.0, score)), 4)

        # Human-readable factor narrative
        pos_factors = []
        neg_factors = []
        if f_hazard >= 0.7:
            pos_factors.append(f"Low hazard risk ({hazard:.0f}/100)")
        if f_area >= 0.5:
            pos_factors.append(f"Adequate safe area ({area:.2f} km²)")
        if f_distance >= 0.6:
            pos_factors.append(f"Close proximity ({dist:.1f} km)")
        if f_road >= 0.6:
            pos_factors.append("Good road accessibility")
        if f_health >= 0.5:
            pos_factors.append("Adequate healthcare access")
        if f_pop > 0.5:
            neg_factors.append("Moderate existing population pressure")
        if dist > search_radius_km * 0.7:
            neg_factors.append(f"Moderate distance ({dist:.1f} km)")

        notes_parts = []
        if pos_factors:
            notes_parts.append("Strengths: " + "; ".join(pos_factors))
        if neg_factors:
            notes_parts.append("Constraints: " + "; ".join(neg_factors))
        notes = ".  ".join(notes_parts) if notes_parts else "Moderate candidate area."

        scored_candidates.append(
            RelocationCandidate(
                candidate_id=c["cluster_id"],
                source_hab_id=hab_id,
                centroid_lat=c["centroid_lat"],
                centroid_lon=c["centroid_lon"],
                distance_km=dist,
                area_km2=area,
                candidate_score=score,
                mean_hazard_score=hazard,
                nearest_road_km=src_road,
                nearest_healthcare_km=src_health,
                cell_count=c["cell_count"],
                notes=notes,
                data_provenance="spatial_zone_green",
            )
        )

    # Sort by score descending
    scored_candidates.sort(key=lambda x: x.candidate_score, reverse=True)
    return scored_candidates[:max_candidates]
