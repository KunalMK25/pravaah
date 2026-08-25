"""
PRAVAAH — Habitation / settlement ingestion from OpenStreetMap.

Fetches settlement nodes (place=city/town/village/hamlet/suburb/neighbourhood/
locality/isolated_dwelling/farm) from Overpass API for any bounding box worldwide.

Architecture mirrors the water-body ingestion module:
  - Same mirror list and tenacity @retry decorator
  - Same disk-cache by rounded bbox key
  - Same fallback / empty GeoDataFrame convention
  - source attribute: "osm_overpass" | "osm_cache" | "curated" | "fallback"
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
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

# ── Overpass query — settlement place nodes only ──────────────────────────────
# We query *nodes* with a place tag.  Relations and ways are excluded because
# their centroids are unreliable without full geometry resolution and massively
# inflate the response size.  Suburb/neighbourhood are included for dense
# urban areas (Bangalore micro-zones, Chennai wards etc.).
_HABITATION_QUERY = (
    "[out:json][timeout:60];\n"
    "(\n"
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
    ");\n"
    "out body;"
)


class OverpassError(IOError):
    """Raised when an Overpass mirror returns a non-200 response."""


@retry(
    retry=retry_if_exception_type((OverpassError, requests.RequestException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_with_retry(query: str) -> dict:
    """POST query to each Overpass mirror; retry up to 3 times."""
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "SIH26191-HabitationIngest/1.0",
    }
    last_exc: Exception = OverpassError("No mirrors tried")
    for mirror in _MIRRORS:
        try:
            r = requests.post(
                mirror,
                data=query.encode("utf-8"),
                headers=headers,
                timeout=30,
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


def _osm_to_habitations(osm_data: dict, source: str) -> list[Habitation]:
    """Convert raw Overpass JSON to a list of Habitation objects."""
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
        results.append(
            Habitation(
                hab_id=str(row.get("hab_id", f"cached_{_}")),
                name=str(row.get("name", "")),
                hab_type=str(row.get("hab_type", "unknown")),
                lat=float(row.get("lat", 0.0)),
                lon=float(row.get("lon", 0.0)),
                source=source,
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
    Load settlement / habitation nodes for a bounding box from OSM.

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
        logger.warning("Network disabled — returning fallback habitations.")
        habs = _fallback_habitations(bounding_box)
        return HabitationDataset(habitations=habs, source="fallback", bbox_key=bbox_key)

    query = _HABITATION_QUERY.format(
        s=min_lat, w=min_lon, n=max_lat, e=max_lon
    )
    logger.info("Fetching habitations from Overpass API for %s…", bounding_box)
    osm_data = _fetch(query)

    if osm_data is not None:
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        logger.info("Fetched %d habitation nodes from Overpass.", len(habs))
        if cache_path is not None:
            try:
                gdf = _habitations_to_gdf(habs)
                if len(gdf) > 0:
                    gdf.to_file(str(cache_path), driver="GeoJSON")
                else:
                    cache_path.write_text(
                        json.dumps({"type": "FeatureCollection", "features": []}),
                        encoding="utf-8",
                    )
            except Exception as exc:
                logger.warning("Failed to cache habitations: %s", exc)
        # If OSM found nothing meaningful, add synthetic fallback points so
        # the pipeline still demonstrates the downstream SIH stages.
        if len(habs) == 0:
            logger.info("No OSM habitation nodes found — using fallback habitations.")
            habs = _fallback_habitations(bounding_box)
            return HabitationDataset(habitations=habs, source="fallback", bbox_key=bbox_key)
        return HabitationDataset(habitations=habs, source="osm_overpass", bbox_key=bbox_key)

    logger.warning("Overpass unavailable after retries — using fallback habitations.")
    habs = _fallback_habitations(bounding_box)
    return HabitationDataset(habitations=habs, source="fallback", bbox_key=bbox_key)
