"""
PRAVAAH-AI -- Habitation / settlement ingestion from OpenStreetMap.

Coverage strategy (v2):
  1. OSM place nodes  -- city/town/village/hamlet/suburb/neighbourhood/
                         locality/isolated_dwelling/farm/allotments
     These are named-settlement markers; one point per settlement.

  2. Residential building ways -- building=house/residential/apartments/
     detached/semidetached_house/terrace/bungalow/dormitory/hut/cabin
     Each qualifying building polygon is converted to its centroid so it
     appears as a habitation point.  Non-residential building types
     (industrial, warehouse, commercial, school, hospital, etc.) are
     excluded by a strict allowlist.

  3. Residential landuse polygons -- landuse=residential
     Large residential zones that may not have individual place nodes
     or building records.  Each polygon centroid becomes one habitation
     point (minimum area 1000 m2 to exclude slivers).

All three are fetched in a single compound Overpass query and cached
together under the same bbox key.

Deduplication:
  A building- or landuse-derived point that falls within _DEDUP_RADIUS_DEG
  (approx 50 m) of an existing place node is discarded.  This prevents
  a building cluster co-located with a place node from creating a duplicate.
  Stable OSM IDs (osm_{id}, bld_{id}, luse_{id}) ensure no two records
  share the same identifier.

Architecture mirrors the water-body ingestion module:
  - Same mirror list and tenacity @retry decorator
  - Same disk-cache by rounded bbox key (GeoJSON)
  - Same fallback / empty GeoDataFrame convention
  - source attribute per record:
      "osm_overpass"  -- from place node
      "osm_building"  -- from residential building footprint
      "osm_landuse"   -- from residential landuse polygon
  - HabitationDataset.source:
      "osm_overpass"           -- live fetch succeeded (any records found)
      "osm_overpass_buildings" -- live fetch, only buildings/landuse (no place nodes)
      "osm_cache"              -- all data from cache
      "fallback"               -- network unavailable or zero features found
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Point
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.models import Habitation, HabitationDataset

logger = logging.getLogger(__name__)

_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# ── Allowlist: ONLY these building values are treated as residential habitation
# This is intentionally conservative -- we prefer false negatives over false
# positives (i.e. it is better to miss an unusual residential type than to
# incorrectly include a warehouse or school as a habitation).
_RESIDENTIAL_BUILDING_TYPES: frozenset[str] = frozenset({
    "house",
    "residential",
    "apartments",
    "detached",
    "semidetached_house",
    "terrace",
    "bungalow",
    "dormitory",
    "hut",
    "cabin",
})

# Minimum polygon area (degrees^2) for residential landuse to be included.
# Approx 1000 m^2 at the equator ~= 8e-8 deg^2; we use 1e-7 to be safe.
_MIN_LANDUSE_AREA_DEG2: float = 1e-7

# Spatial deduplication radius in degrees (~50 m at the equator).
# A building/landuse centroid within this radius of an existing place node
# is considered a duplicate and discarded.
_DEDUP_RADIUS_DEG: float = 0.00045

# ── Compound Overpass query -- all three data layers in one request ────────────
# Uses 'out geom' for ways so that polygon centroids can be derived.
# Nodes use 'out body' (no geometry needed beyond lat/lon).
_HABITATION_QUERY = (
    "[out:json][timeout:60];\n"
    "(\n"
    # --- Layer 1: named settlement place nodes ---
    "  node[\"place\"=\"city\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"town\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"village\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"hamlet\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"suburb\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"neighbourhood\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"locality\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"isolated_dwelling\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"farm\"]({s},{w},{n},{e});\n"
    "  node[\"place\"=\"allotments\"]({s},{w},{n},{e});\n"
    # --- Layer 2: residential building ways (strict allowlist) ---
    "  way[\"building\"~\"^(house|residential|apartments|detached|"
    "semidetached_house|terrace|bungalow|dormitory|hut|cabin)$\"]({s},{w},{n},{e});\n"
    # --- Layer 3: residential landuse polygons ---
    "  way[\"landuse\"=\"residential\"]({s},{w},{n},{e});\n"
    ");\n"
    "out geom;"
)


class OverpassError(IOError):
    """Raised when an Overpass mirror returns a non-200 response."""


@retry(
    retry=retry_if_exception_type((OverpassError, requests.RequestException)),
    stop=stop_after_attempt(2),  # Reduced from 3 to 2
    wait=wait_exponential(multiplier=1, min=1, max=10),  # Reduced wait
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_with_retry(query: str) -> dict:
    """POST query to each Overpass mirror; retry up to 2 times.
    
    PERFORMANCE: Balanced timeout (15s) with 2 retries = max 30s per API call.
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "SIH26191-HabitationIngest/2.0",
    }
    last_exc: Exception = OverpassError("No mirrors tried")
    for mirror in _MIRRORS:
        try:
            r = requests.post(
                mirror,
                data=query.encode("utf-8"),
                headers=headers,
                timeout=15,  # Increased from 8s to 15s for reliability
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5)
            last_exc = OverpassError(f"Mirror {mirror} returned HTTP {r.status_code}")
        except requests.RequestException as exc:
            logger.debug("Mirror %s failed: %s", mirror, exc)
            last_exc = exc
    raise last_exc


def _fetch(query: str) -> dict | None:
    try:
        return _fetch_with_retry(query)
    except Exception as exc:
        logger.warning("All Overpass retries exhausted for habitation query: %s", exc)
        return None


def _parse_population(tags: dict) -> Optional[int]:
    """
    Extract population from OSM tags.

    Returns an int if the tag is a valid positive number, else None.
    We do NOT invent population values.
    """
    raw = tags.get("population", "")
    if not raw:
        return None
    try:
        val = int(str(raw).replace(",", "").strip())
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


def _parse_way_centroid(element: dict) -> tuple[float, float] | None:
    """
    Derive (lat, lon) centroid from an Overpass way element returned
    with ``out geom``.

    Overpass includes a ``geometry`` list of {lat, lon} dicts for each
    node of the way when ``out geom`` is used.  We average them to get a
    representative centroid.

    Returns None if geometry is missing or has fewer than 3 nodes
    (degenerate polygon).
    """
    geom = element.get("geometry", [])
    if len(geom) < 3:
        return None
    lats = [pt["lat"] for pt in geom if "lat" in pt and "lon" in pt]
    lons = [pt["lon"] for pt in geom if "lat" in pt and "lon" in pt]
    if len(lats) < 3:
        return None
    return float(np.mean(lats)), float(np.mean(lons))


def _osm_nodes_to_habitations(osm_data: dict, source: str) -> list[Habitation]:
    """Convert place node elements from Overpass JSON to Habitation objects."""
    results: list[Habitation] = []
    for el in osm_data.get("elements", []):
        if el.get("type") != "node":
            continue
        tags = el.get("tags", {})
        place = tags.get("place", "")
        if not place:
            continue
        osm_id = el.get("id")
        lat = el.get("lat")
        lon = el.get("lon")
        if lat is None or lon is None:
            continue
        name = tags.get("name", tags.get("name:en", "")).strip()
        pop = _parse_population(tags)
        metadata: dict = {}
        for k in ("admin_level", "is_in", "wikidata", "wikipedia", "alt_name"):
            if k in tags:
                metadata[k] = tags[k]
        results.append(
            Habitation(
                hab_id=f"osm_{osm_id}",
                name=name,
                hab_type=place,
                lat=float(lat),
                lon=float(lon),
                source=source,
                population=pop,
                osm_id=osm_id,
                metadata=metadata,
            )
        )
    return results


def _osm_ways_to_habitations(osm_data: dict, source_tag: str) -> list[Habitation]:
    """
    Convert residential building way elements and residential landuse way
    elements from Overpass JSON (fetched with ``out geom``) to Habitation
    objects.

    Only building types in _RESIDENTIAL_BUILDING_TYPES are included.
    Landuse=residential ways above _MIN_LANDUSE_AREA_DEG2 are included.

    Returns
    -------
    list[Habitation]
        Building-derived habitations tagged source="osm_building" and
        landuse-derived habitations tagged source="osm_landuse".
        Population is always None (buildings do not have population tags).
    """
    results: list[Habitation] = []
    for el in osm_data.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        osm_id = el.get("id")

        # --- Check: residential building ---
        building_val = tags.get("building", "").strip().lower()
        if building_val in _RESIDENTIAL_BUILDING_TYPES:
            centroid = _parse_way_centroid(el)
            if centroid is None:
                continue
            lat, lon = centroid
            name = tags.get("name", tags.get("addr:street", "")).strip()
            results.append(
                Habitation(
                    hab_id=f"bld_{osm_id}",
                    name=name,
                    hab_type=f"building_{building_val}",
                    lat=lat,
                    lon=lon,
                    source="osm_building",
                    population=None,  # never fabricate
                    osm_id=osm_id,
                    metadata={"building": building_val},
                )
            )
            continue

        # --- Check: residential landuse polygon ---
        landuse_val = tags.get("landuse", "").strip().lower()
        if landuse_val == "residential":
            geom_pts = el.get("geometry", [])
            if len(geom_pts) < 3:
                continue
            # Estimate polygon area in degrees^2 to filter tiny slivers
            lats = [pt["lat"] for pt in geom_pts if "lat" in pt]
            lons = [pt["lon"] for pt in geom_pts if "lon" in pt]
            if len(lats) < 3:
                continue
            # Simple shoelace formula for signed area
            n_pts = len(lats)
            area = 0.0
            for i in range(n_pts):
                j = (i + 1) % n_pts
                area += lons[i] * lats[j]
                area -= lons[j] * lats[i]
            area = abs(area) / 2.0
            if area < _MIN_LANDUSE_AREA_DEG2:
                continue
            centroid = _parse_way_centroid(el)
            if centroid is None:
                continue
            lat, lon = centroid
            name = tags.get("name", "").strip()
            results.append(
                Habitation(
                    hab_id=f"luse_{osm_id}",
                    name=name,
                    hab_type="residential_landuse",
                    lat=lat,
                    lon=lon,
                    source="osm_landuse",
                    population=None,  # never fabricate
                    osm_id=osm_id,
                    metadata={"landuse": "residential"},
                )
            )

    return results


def _deduplicate_habitations(
    habitations: list[Habitation],
    radius_deg: float = _DEDUP_RADIUS_DEG,
) -> list[Habitation]:
    """
    Remove building/landuse-derived habitation points that are within
    *radius_deg* of an existing place-node habitation.

    Priority order: place nodes > buildings > landuse.

    Algorithm:
      1. Separate into place-node records and derived records.
      2. For each derived record, compute minimum distance to any place-node
         centroid (vectorised).
      3. Discard derived records within radius_deg of any place node.
      4. Return place nodes + surviving derived records.

    The radius is expressed in degrees so it works for any bounding box
    worldwide without city-specific tuning.

    Parameters
    ----------
    habitations : list[Habitation]
    radius_deg : float
        Deduplication search radius in WGS84 degrees (~50 m at equator).

    Returns
    -------
    list[Habitation]
        Deduplicated habitations.
    """
    if not habitations:
        return []

    # Separate by source type
    place_habs = [h for h in habitations if h.source in {"osm_overpass", "osm_cache"}]
    derived_habs = [h for h in habitations if h.source in {"osm_building", "osm_landuse"}]

    if not place_habs:
        # No place nodes to deduplicate against — keep all unique hab_ids
        seen_ids: set[str] = set()
        result = []
        for h in derived_habs:
            if h.hab_id not in seen_ids:
                seen_ids.add(h.hab_id)
                result.append(h)
        return result

    if not derived_habs:
        return place_habs

    # Vectorised distance check: derived point vs all place node points
    place_lats = np.array([h.lat for h in place_habs], dtype=np.float64)
    place_lons = np.array([h.lon for h in place_habs], dtype=np.float64)

    kept_derived: list[Habitation] = []
    seen_ids = {h.hab_id for h in place_habs}

    for h in derived_habs:
        if h.hab_id in seen_ids:
            continue
        dlat = place_lats - h.lat
        dlon = place_lons - h.lon
        min_dist = float(np.sqrt(np.min(dlat ** 2 + dlon ** 2)))
        if min_dist >= radius_deg:
            kept_derived.append(h)
            seen_ids.add(h.hab_id)

    result = place_habs + kept_derived
    n_dropped = len(derived_habs) - len(kept_derived)
    if n_dropped > 0:
        logger.debug(
            "Deduplication: dropped %d derived records within %.5f deg of a place node.",
            n_dropped, radius_deg,
        )
    return result


def _habitations_to_gdf(habitations: list[Habitation]) -> gpd.GeoDataFrame:
    """Convert a list of Habitation objects to a GeoDataFrame for caching."""
    if not habitations:
        return gpd.GeoDataFrame(
            columns=["hab_id", "name", "hab_type", "lat", "lon", "source",
                     "population", "osm_id", "geometry"],
            crs="EPSG:4326",
        )
    rows = []
    for h in habitations:
        rows.append({
            "hab_id": h.hab_id,
            "name": h.name,
            "hab_type": h.hab_type,
            "lat": h.lat,
            "lon": h.lon,
            "source": h.source,
            "population": h.population,
            "osm_id": h.osm_id,
            "geometry": Point(h.lon, h.lat),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _gdf_to_habitations(gdf: gpd.GeoDataFrame, source: str) -> list[Habitation]:
    """Reconstruct Habitation objects from a cached GeoDataFrame."""
    results = []
    for _, row in gdf.iterrows():
        pop_raw = row.get("population")
        pop = int(pop_raw) if pop_raw is not None and str(pop_raw) not in ("", "nan", "None") else None
        # Preserve original per-record source if stored; fall back to dataset source
        rec_source = str(row.get("source", source))
        if rec_source not in {"osm_overpass", "osm_building", "osm_landuse", "fallback"}:
            rec_source = source
        results.append(
            Habitation(
                hab_id=str(row.get("hab_id", f"cached_{_}")),
                name=str(row.get("name", "")),
                hab_type=str(row.get("hab_type", "unknown")),
                lat=float(row.get("lat", 0.0)),
                lon=float(row.get("lon", 0.0)),
                source=rec_source,
                population=pop,
                osm_id=row.get("osm_id"),
                metadata={},
            )
        )
    return results


def _fallback_habitations(bbox: BoundingBox) -> list[Habitation]:
    """
    Return a minimal synthetic set of habitation points when OSM is unavailable.

    These are clearly marked source="fallback" and should never be presented
    as real settlement data.  They allow the pipeline to complete in offline
    or CI environments and demonstrate the downstream analysis stages.
    """
    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    center_lon = (bbox.min_lon + bbox.max_lon) / 2.0
    d = min(
        (bbox.max_lat - bbox.min_lat) * 0.25,
        (bbox.max_lon - bbox.min_lon) * 0.25,
    )
    return [
        Habitation(
            hab_id="fallback_001",
            name="Settlement A (fallback)",
            hab_type="village",
            lat=center_lat + d,
            lon=center_lon - d,
            source="fallback",
            population=None,
        ),
        Habitation(
            hab_id="fallback_002",
            name="Settlement B (fallback)",
            hab_type="hamlet",
            lat=center_lat - d,
            lon=center_lon + d,
            source="fallback",
            population=None,
        ),
        Habitation(
            hab_id="fallback_003",
            name="Settlement C (fallback)",
            hab_type="suburb",
            lat=center_lat,
            lon=center_lon,
            source="fallback",
            population=None,
        ),
    ]


def load_habitations(
    bounding_box: BoundingBox,
    cache_dir: str | Path | None = "data/cache/habitations",
    allow_network: bool = True,
) -> HabitationDataset:
    """
    Load habitation data for a bounding box from OSM.

    Coverage (in a single compound Overpass request):
      - Place nodes:        named settlements (city, town, village, hamlet,
                            suburb, neighbourhood, locality, isolated_dwelling,
                            farm, allotments)
      - Residential ways:   building=house/residential/apartments/detached/
                            semidetached_house/terrace/bungalow/dormitory/
                            hut/cabin (centroids of qualifying footprints)
      - Residential landuse: landuse=residential polygon centroids

    All three layers are combined, deduplicated (building/landuse centroids
    within ~50 m of a place node are discarded), and cached together.

    Resolution order:
    1. Local GeoJSON cache (instant, no network required).
    2. Live Overpass API with 3 tenacity retries.
    3. Fallback: synthetic minimal habitation set (source="fallback");
       pipeline continues but UI should show a warning.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent to query.
    cache_dir : str | Path | None
        Directory for GeoJSON cache.  Pass None to disable caching.
    allow_network : bool
        If False, skip network calls (returns cached or fallback).

    Returns
    -------
    HabitationDataset
        Dataset with habitations list and source provenance.
    """
    min_lon = round(bounding_box.min_lon, 4)
    min_lat = round(bounding_box.min_lat, 4)
    max_lon = round(bounding_box.max_lon, 4)
    max_lat = round(bounding_box.max_lat, 4)
    bbox_key = f"hab_{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}"

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{bbox_key}.geojson"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                gdf = gpd.read_file(str(cache_path))
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                habs = _gdf_to_habitations(gdf, source="osm_cache")
                logger.info("Habitations from cache: %d features.", len(habs))
                return HabitationDataset(habitations=habs, source="osm_cache", bbox_key=bbox_key)
            except Exception as exc:
                logger.warning("Habitation cache read failed: %s", exc)

    if not allow_network:
        logger.warning("Network disabled -- returning fallback habitations.")
        habs = _fallback_habitations(bounding_box)
        return HabitationDataset(habitations=habs, source="fallback", bbox_key=bbox_key)

    query = _HABITATION_QUERY.format(
        s=min_lat, w=min_lon, n=max_lat, e=max_lon
    )
    logger.info("Fetching habitations from Overpass API for %s...", bounding_box)
    osm_data = _fetch(query)

    if osm_data is not None:
        # Parse all three layers
        place_habs = _osm_nodes_to_habitations(osm_data, source="osm_overpass")
        derived_habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        all_habs = _deduplicate_habitations(place_habs + derived_habs)

        n_place = len(place_habs)
        n_bld = sum(1 for h in derived_habs if h.source == "osm_building")
        n_luse = sum(1 for h in derived_habs if h.source == "osm_landuse")
        logger.info(
            "Habitations fetched: %d total (%d place nodes, %d buildings, "
            "%d landuse) after dedup=%d.",
            len(all_habs), n_place, n_bld, n_luse, len(all_habs),
        )

        # Persist to cache
        if cache_path is not None:
            try:
                gdf = _habitations_to_gdf(all_habs)
                if len(gdf) > 0:
                    gdf.to_file(str(cache_path), driver="GeoJSON")
                else:
                    cache_path.write_text(
                        json.dumps({"type": "FeatureCollection", "features": []}),
                        encoding="utf-8",
                    )
            except Exception as exc:
                logger.warning("Failed to cache habitations: %s", exc)

        if len(all_habs) == 0:
            logger.info("No OSM habitation features found -- using fallback habitations.")
            habs = _fallback_habitations(bounding_box)
            return HabitationDataset(habitations=habs, source="fallback", bbox_key=bbox_key)

        # Report dataset-level source
        dataset_source = "osm_overpass" if n_place > 0 else "osm_overpass_buildings"
        return HabitationDataset(habitations=all_habs, source=dataset_source, bbox_key=bbox_key)

    logger.warning("Overpass unavailable after retries -- using fallback habitations.")
    habs = _fallback_habitations(bounding_box)
    return HabitationDataset(habitations=habs, source="fallback", bbox_key=bbox_key)


# Keep the original function name as an alias for external callers that import it
# directly (e.g. test fixtures).  Internal code should use _osm_nodes_to_habitations.
_osm_to_habitations = _osm_nodes_to_habitations
