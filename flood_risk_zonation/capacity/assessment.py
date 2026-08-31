"""
PRAVAAH-AI — Carrying-capacity assessment.

METHODOLOGY (declared, transparent, no black-box AI):

Carrying capacity answers: "Can the area absorb displaced people from this
habitation if they need to evacuate or relocate?"

Components assessed:
  1. nearby_safe_area_km2   — low-risk land within search_radius_km
                               (not Water, not High-risk)
  2. nearest_healthcare_km  — network distance (or straight-line fallback) to
                               nearest OSM hospital or clinic; -1 if none found
  3. nearest_road_km        — network distance (or straight-line fallback) to
                               nearest OSM highway (primary | secondary | trunk |
                               motorway); -1 if none
  4. shelter_capacity       — curated or "unavailable" (not fabricated)

Composite capacity_score = weighted average of normalised components.
Declared weights are visible in CAPACITY_WEIGHTS below.

Status thresholds (documented):
  ADEQUATE : capacity_score >= 0.60
  STRESSED : capacity_score >= 0.35
  CRITICAL : capacity_score <  0.35

Healthcare and road data: fetched via OSM Overpass (same architecture as
the water-body ingest module — cached, retried, gracefully degraded).
Results are cached. Fallback to -1 (unknown) on failure.

ROUTING (NEW):
  Where available, network distances are calculated using shortest-path
  routing on the OSM road network graph. If routing fails, the system
  gracefully falls back to straight-line (haversine) distance. The routing
  method is tracked via provenance in the result notes.
"""
from __future__ import annotations

import json
import logging
import time
from math import cos, radians, sqrt
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.models import CarryingCapacityResult, ExposureResult
from flood_risk_zonation.utils.routing import (
    NetworkDistance,
    build_road_graph,
    shortest_network_distance,
)

logger = logging.getLogger(__name__)

# ── Declared weights ──────────────────────────────────────────────────────────
CAPACITY_WEIGHTS: dict[str, float] = {
    "safe_area":        0.45,   # most important — is there somewhere safe to go?
    "road_access":      0.30,   # can they get there?
    "healthcare":       0.25,   # is there medical support?
}
assert abs(sum(CAPACITY_WEIGHTS.values()) - 1.0) < 1e-9

# ── Status thresholds ─────────────────────────────────────────────────────────
_STATUS_ADEQUATE = 0.60
_STATUS_STRESSED = 0.35

# ── Safe-area search radius ───────────────────────────────────────────────────
_DEFAULT_SAFE_RADIUS_KM = 5.0   # documented search radius
_SAFE_AREA_PER_PERSON_KM2 = 0.001  # 1000 m² per person minimum safe land

# ── OSM infrastructure queries ────────────────────────────────────────────────
_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

_HEALTHCARE_QUERY = (
    "[out:json][timeout:45];\n"
    "(\n"
    "  node[\"amenity\"=\"hospital\"]({s},{w},{n},{e});\n"
    "  node[\"amenity\"=\"clinic\"]({s},{w},{n},{e});\n"
    "  node[\"amenity\"=\"health_centre\"]({s},{w},{n},{e});\n"
    "  node[\"amenity\"=\"doctors\"]({s},{w},{n},{e});\n"
    "  way[\"amenity\"=\"hospital\"]({s},{w},{n},{e});\n"
    ");\n"
    "out center;"
)

_SHELTER_QUERY = (
    "[out:json][timeout:45];\n"
    "(\n"
    "  node[\"amenity\"=\"shelter\"]({s},{w},{n},{e});\n"
    "  node[\"amenity\"=\"community_centre\"]({s},{w},{n},{e});\n"
    "  node[\"amenity\"=\"social_centre\"]({s},{w},{n},{e});\n"
    "  way[\"amenity\"=\"shelter\"]({s},{w},{n},{e});\n"
    "  way[\"amenity\"=\"community_centre\"]({s},{w},{n},{e});\n"
    ");\n"
    "out center;"
)

_ROAD_QUERY = (
    "[out:json][timeout:45];\n"
    "(\n"
    "  way[\"highway\"~\"^(motorway|trunk|primary|secondary|secondary_link|trunk_link|motorway_link)$\"]"
    "({s},{w},{n},{e});\n"
    ");\n"
    "out geom;"
)


class OverpassError(IOError):
    """Raised when Overpass returns non-200."""


@retry(
    retry=retry_if_exception_type((OverpassError, requests.RequestException)),
    stop=stop_after_attempt(2),  # Reduced from 3 to 2
    wait=wait_exponential(multiplier=1, min=1, max=10),  # Reduced wait
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_with_retry(query: str) -> dict:
    """POST query to Overpass mirrors with retry.
    
    PERFORMANCE: Balanced timeout (15s) with 2 retries = max 30s per API call.
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "SIH26191-CapacityAssessment/1.0",
    }
    last_exc: Exception = OverpassError("No mirrors tried")
    for mirror in _MIRRORS:
        try:
            r = requests.post(mirror, data=query.encode("utf-8"), headers=headers, timeout=15)  # Increased from 8s to 15s
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5)
            last_exc = OverpassError(f"{mirror} → HTTP {r.status_code}")
        except requests.RequestException as exc:
            last_exc = exc
    raise last_exc


def _fetch(query: str) -> dict | None:
    try:
        return _fetch_with_retry(query)
    except Exception as exc:
        logger.warning("Overpass fetch failed: %s", exc)
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * (dlon / 2) ** 2
    return R * 2 * sqrt(max(a, 0))


def _bbox_expanded(bbox: BoundingBox, extra_deg: float = 0.1) -> BoundingBox:
    """Return a slightly enlarged bbox for infrastructure queries."""
    return BoundingBox(
        min_lon=bbox.min_lon - extra_deg,
        min_lat=bbox.min_lat - extra_deg,
        max_lon=bbox.max_lon + extra_deg,
        max_lat=bbox.max_lat + extra_deg,
    )


# ── Healthcare facility cache ─────────────────────────────────────────────────

def _load_healthcare(
    bbox: BoundingBox,
    cache_dir: Path,
    allow_network: bool,
) -> list[tuple[float, float]]:
    """Return list of (lat, lon) tuples for healthcare facilities in bbox."""
    bkey = f"{bbox.min_lon:.4f}_{bbox.min_lat:.4f}_{bbox.max_lon:.4f}_{bbox.max_lat:.4f}"
    cache_path = cache_dir / f"hc_{bkey}.json"

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            logger.debug("Healthcare from cache: %d facilities.", len(data))
            return [(d["lat"], d["lon"]) for d in data]
        except Exception:
            pass

    if not allow_network:
        return []

    query = _HEALTHCARE_QUERY.format(
        s=bbox.min_lat, w=bbox.min_lon, n=bbox.max_lat, e=bbox.max_lon
    )
    raw = _fetch(query)
    if raw is None:
        return []

    points: list[dict] = []
    for el in raw.get("elements", []):
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat and lon:
            points.append({"lat": float(lat), "lon": float(lon)})

    try:
        cache_path.write_text(json.dumps(points), encoding="utf-8")
    except Exception:
        pass

    logger.info("Fetched %d healthcare facilities.", len(points))
    return [(p["lat"], p["lon"]) for p in points]


def _load_shelters(
    bbox: BoundingBox,
    cache_dir: Path,
    allow_network: bool,
) -> list[tuple[float, float]]:
    """Return list of (lat, lon) tuples for shelter facilities in bbox."""
    bkey = f"{bbox.min_lon:.4f}_{bbox.min_lat:.4f}_{bbox.max_lon:.4f}_{bbox.max_lat:.4f}"
    cache_path = cache_dir / f"shelters_{bkey}.json"

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            logger.debug("Shelters from cache: %d facilities.", len(data))
            return [(d["lat"], d["lon"]) for d in data]
        except Exception:
            pass

    if not allow_network:
        return []

    query = _SHELTER_QUERY.format(
        s=bbox.min_lat, w=bbox.min_lon, n=bbox.max_lat, e=bbox.max_lon
    )
    raw = _fetch(query)
    if raw is None:
        return []

    points: list[dict] = []
    for el in raw.get("elements", []):
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat and lon:
            points.append({"lat": float(lat), "lon": float(lon)})

    try:
        cache_path.write_text(json.dumps(points), encoding="utf-8")
    except Exception:
        pass

    logger.info("Fetched %d shelter facilities.", len(points))
    return [(p["lat"], p["lon"]) for p in points]


def _load_roads(
    bbox: BoundingBox,
    cache_dir: Path,
    allow_network: bool,
) -> list[tuple[float, float]]:
    """
    Return list of (lat, lon) midpoints of major road segments in bbox.
    
    Also caches full geometry for potential routing use.
    """
    bkey = f"{bbox.min_lon:.4f}_{bbox.min_lat:.4f}_{bbox.max_lon:.4f}_{bbox.max_lat:.4f}"
    cache_path = cache_dir / f"roads_{bkey}.json"

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            logger.debug("Roads from cache: %d points.", len(data))
            return [(d["lat"], d["lon"]) for d in data]
        except Exception:
            pass

    if not allow_network:
        return []

    query = _ROAD_QUERY.format(
        s=bbox.min_lat, w=bbox.min_lon, n=bbox.max_lat, e=bbox.max_lon
    )
    raw = _fetch(query)
    if raw is None:
        return []

    points: list[dict] = []
    for el in raw.get("elements", []):
        geom = el.get("geometry", [])
        if geom:
            # Sample multiple points from the geometry to better represent road network
            # Use ~2 km spacing (approx 0.018 degrees)
            for coord in geom:
                points.append({"lat": float(coord["lat"]), "lon": float(coord["lon"])})

    try:
        cache_path.write_text(json.dumps(points), encoding="utf-8")
    except Exception:
        pass

    logger.info("Fetched %d road geometry points.", len(points))
    return [(p["lat"], p["lon"]) for p in points]


def _nearest_km(
    hab_lat: float,
    hab_lon: float,
    points: list[tuple[float, float]],
    road_graph=None,
) -> tuple[float, str]:
    """
    Return (distance_km, method) tuple.
    
    Distance is in km to the nearest point.
    Method is one of: 'network_routing', 'straight_line_fallback', 'unavailable'.
    
    Returns (-1.0, 'unavailable') if no points available.
    Uses routing if graph is available, otherwise falls back to haversine.
    
    OPTIMIZATION: Uses vectorized haversine calculation when many points exist.
    """
    if not points:
        return -1.0, "unavailable"
    
    # Attempt routing if graph provided
    if road_graph is not None:
        try:
            result = shortest_network_distance(
                hab_lat, hab_lon, points,
                graph=road_graph,
                allow_fallback=True
            )
            return result.distance_km, result.method
        except Exception as e:
            logger.debug("Routing error: %s; using fallback", e)
    
    # Fallback to haversine (vectorized for performance)
    if len(points) > 10:
        # Vectorized calculation for many points
        import numpy as np
        lats = np.array([p[0] for p in points])
        lons = np.array([p[1] for p in points])
        dlat = np.radians(lats - hab_lat)
        dlon = np.radians(lons - hab_lon)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(hab_lat)) * np.cos(np.radians(lats)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371 * c
        min_dist = float(np.min(distances))
    else:
        # Traditional loop for few points
        distances = [_haversine_km(hab_lat, hab_lon, lat, lon) for lat, lon in points]
        min_dist = min(distances)
    
    return round(min_dist, 3), "straight_line_fallback"


def _compute_safe_area(
    hab_lat: float,
    hab_lon: float,
    scored_grid: gpd.GeoDataFrame,
    radius_km: float = _DEFAULT_SAFE_RADIUS_KM,
) -> float:
    """
    Compute nearby safe-area in km² within radius_km of the habitation.

    'Safe' = Low-risk class, not Water.  We use the degree-proxy distance
    for speed (same as exposure analysis — acceptable for 5 km radius).
    """
    if "risk_class" not in scored_grid.columns:
        return 0.0

    # Degree-proxy distance filter
    dlat = scored_grid["centroid_lat"].values - hab_lat if "centroid_lat" in scored_grid.columns else (
        scored_grid.geometry.centroid.y.values - hab_lat
    )
    dlon = (
        scored_grid["centroid_lon"].values - hab_lon
        if "centroid_lon" in scored_grid.columns
        else scored_grid.geometry.centroid.x.values - hab_lon
    )
    # Rough km conversion (equirectangular)
    km_lat = 111.32
    km_lon = 111.32 * cos(radians(hab_lat))
    dist_km = np.sqrt((dlat * km_lat) ** 2 + (dlon * km_lon) ** 2)

    nearby_mask = dist_km <= radius_km
    nearby = scored_grid[nearby_mask]

    safe = nearby[nearby["risk_class"] == "Low"]
    if len(safe) == 0:
        return 0.0

    # Estimate area from cell size (assume square cells)
    # If geometry is available, compute actual area; else estimate from cell count
    try:
        if safe.crs and str(safe.crs).upper() != "EPSG:3857":
            safe_m = safe.to_crs("EPSG:3857")
        else:
            safe_m = safe
        total_area_m2 = safe_m.geometry.area.sum()
        return round(total_area_m2 / 1_000_000, 4)   # → km²
    except Exception:
        # Fallback: count cells × assumed cell area
        n_cells = len(safe)
        # Guess cell size from centroid spacing
        if n_cells >= 2 and "centroid_lat" in scored_grid.columns:
            lat_vals = scored_grid["centroid_lat"].values
            unique_lats = np.unique(lat_vals)
            if len(unique_lats) >= 2:
                step_deg = float(np.median(np.diff(np.sort(unique_lats))))
                cell_km = step_deg * 111.32
                return round(n_cells * cell_km ** 2, 4)
        return round(n_cells * 0.25, 4)   # assume 500m × 500m cells = 0.25 km²


def _capacity_status(score: float) -> str:
    if score >= _STATUS_ADEQUATE:
        return "ADEQUATE"
    if score >= _STATUS_STRESSED:
        return "STRESSED"
    return "CRITICAL"


def assess_capacity(
    exposure: ExposureResult,
    scored_grid: gpd.GeoDataFrame,
    bbox: BoundingBox,
    cache_dir: str | Path = "data/cache/capacity",
    allow_network: bool = True,
    search_radius_km: float = _DEFAULT_SAFE_RADIUS_KM,
    hc_points: Optional[list[tuple[float, float]]] = None,
    road_points: Optional[list[tuple[float, float]]] = None,
    road_graph: Optional[object] = None,
) -> CarryingCapacityResult:
    """
    Assess carrying capacity for a single habitation.

    Parameters
    ----------
    exposure : ExposureResult
        Exposure output for this habitation.
    scored_grid : gpd.GeoDataFrame
        Phase 1 hazard grid (used for safe-area computation).
    bbox : BoundingBox
        Study area bounds — used for infrastructure queries.
    cache_dir : Path
        Directory for OSM infrastructure cache.
    allow_network : bool
        Whether to fetch live OSM data.
    search_radius_km : float
        Radius for nearby safe-area search.
    hc_points : list of (lat, lon) tuples, optional
        Pre-loaded healthcare facility locations. If None, will be loaded.
    road_points : list of (lat, lon) tuples, optional
        Pre-loaded road points. If None, will be loaded.
    road_graph : networkx graph, optional
        Pre-constructed road network graph. If None, will fall back to haversine distances.
        The caller is responsible for deciding whether to build the graph based on performance
        constraints (e.g., do not build for >500-node networks).

    Returns
    -------
    CarryingCapacityResult
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Expand bbox slightly for infrastructure queries so edge habitations
    # find nearby roads / hospitals just outside the study area.
    infra_bbox = _bbox_expanded(bbox, extra_deg=0.08)

    # 1. Safe area
    safe_area_km2 = _compute_safe_area(
        exposure.lat, exposure.lon, scored_grid, radius_km=search_radius_km
    )

    # 2. Healthcare facilities (use provided or load)
    if hc_points is None:
        hc_points = _load_healthcare(infra_bbox, cache_path, allow_network)
    
    # 3. Roads (load if not already provided)
    if road_points is None:
        road_points = _load_roads(infra_bbox, cache_path, allow_network)
    
    # Note: road_graph may be None intentionally to signal fallback to haversine for large networks.
    # Do not rebuild it here; respect the caller's decision. If building is needed, it must be
    # done by the caller with appropriate size checks (see sih_pipeline.py for >500-node guard).

    # 4. Calculate distances with routing
    nearest_hc_km, hc_method = _nearest_km(
        exposure.lat, exposure.lon, hc_points,
        road_graph=road_graph
    )
    nearest_road_km, road_method = _nearest_km(
        exposure.lat, exposure.lon, road_points,
        road_graph=road_graph
    )

    # ── Normalise to [0, 1] per component (higher = better capacity) ──────────
    # Safe area: 0 → 0, 5 km² → 0.5, 10+ km² → 1.0 (log-scaled)
    import math
    c_safe = min(1.0, math.log1p(safe_area_km2) / math.log1p(10.0))

    # Road access: 0 km → 1.0 (excellent), 5+ km → 0.0 (poor)
    if nearest_road_km < 0:
        c_road = 0.3  # unknown — assume moderate (precautionary)
    else:
        c_road = max(0.0, 1.0 - (nearest_road_km / 5.0))

    # Healthcare: 0 km → 1.0, 20+ km → 0.0
    if nearest_hc_km < 0:
        c_hc = 0.3   # unknown — assume moderate
    else:
        c_hc = max(0.0, 1.0 - (nearest_hc_km / 20.0))

    # ── Weighted composite ────────────────────────────────────────────────────
    score = (
        c_safe * CAPACITY_WEIGHTS["safe_area"]
        + c_road * CAPACITY_WEIGHTS["road_access"]
        + c_hc   * CAPACITY_WEIGHTS["healthcare"]
    )
    score = round(max(0.0, min(1.0, score)), 4)
    status = _capacity_status(score)

    # ── Build notes string with provenance ────────────────────────────────────
    notes_parts = [
        f"Safe area within {search_radius_km:.0f}km: {safe_area_km2:.2f} km²",
    ]
    if nearest_road_km >= 0:
        notes_parts.append(f"Nearest major road: {nearest_road_km:.1f} km ({road_method})")
    else:
        notes_parts.append("Nearest major road: not found in area")
    if nearest_hc_km >= 0:
        notes_parts.append(f"Nearest healthcare: {nearest_hc_km:.1f} km ({hc_method})")
    else:
        notes_parts.append("Nearest healthcare: not found in area")

    logger.debug(
        "Capacity: %s → score=%.3f status=%s (road=%s, hc=%s)", 
        exposure.hab_id, score, status, road_method, hc_method
    )

    return CarryingCapacityResult(
        hab_id=exposure.hab_id,
        capacity_score=score,
        capacity_status=status,
        safe_area_km2=safe_area_km2,
        search_radius_km=search_radius_km,
        nearest_healthcare_km=nearest_hc_km,
        nearest_road_km=nearest_road_km,
        shelter_capacity=None,
        shelter_source="unavailable",
        notes="; ".join(notes_parts),
    )
