"""
OSM water body ingestion for the Flood Risk Zonation System.

Fetches ALL water body types live from Overpass API for any bbox worldwide.
Caches results locally. Retries with exponential back-off via tenacity.
Falls back to a bundled GeoJSON snapshot (Bangalore) when all retries fail.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from flood_risk_zonation.config import BoundingBox

logger = logging.getLogger(__name__)

_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

_QUERY = (
    "[out:json][timeout:60];\n"
    "(\n"
    "  way[\"natural\"=\"water\"]({s},{w},{n},{e});\n"
    "  relation[\"natural\"=\"water\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"river\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"canal\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"stream\"]({s},{w},{n},{e});\n"
    "  way[\"waterway\"=\"drain\"]({s},{w},{n},{e});\n"
    "  way[\"landuse\"=\"reservoir\"]({s},{w},{n},{e});\n"
    "  way[\"landuse\"=\"basin\"]({s},{w},{n},{e});\n"
    "  way[\"natural\"=\"bay\"]({s},{w},{n},{e});\n"
    "  way[\"natural\"=\"coastline\"]({s},{w},{n},{e});\n"
    ");\n"
    "out geom;"
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
    """
    POST *query* to each Overpass mirror in turn.

    Retried up to 3 times (tenacity) with 2 s → 4 s → 8 s back-off.
    Raises OverpassError / requests.RequestException on every attempt so
    tenacity can intercept and retry.
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "FloodRiskZonation/1.0",
    }
    last_exc: Exception = OverpassError("No mirrors tried")
    for mirror in _MIRRORS:
        try:
            r = requests.post(
                mirror,
                data=query.encode("utf-8"),
                headers=headers,
                timeout=25,
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(5)
            last_exc = OverpassError(
                f"Mirror {mirror} returned HTTP {r.status_code}"
            )
        except requests.RequestException as exc:
            logger.debug("Mirror %s failed: %s", mirror, exc)
            last_exc = exc
    raise last_exc


def _fetch(query: str) -> dict | None:
    """
    Call _fetch_with_retry; return None if all retries are exhausted.

    This wrapper keeps the public API of load_water_bodies unchanged.
    """
    try:
        return _fetch_with_retry(query)
    except Exception as exc:
        logger.warning("All Overpass retries exhausted: %s", exc)
        return None


def _osm_to_gdf(osm_data: dict, bbox: BoundingBox) -> gpd.GeoDataFrame:
    bbox_poly = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    features = []
    for el in osm_data.get("elements", []):
        element_type = el.get("type")
        tags = el.get("tags", {})
        wtype = tags.get("natural", tags.get("waterway", tags.get("landuse", "water")))
        try:
            if element_type == "way":
                pts = el.get("geometry", [])
                if len(pts) < 2:
                    continue
                coords = [(p["lon"], p["lat"]) for p in pts]
                is_closed = len(coords) >= 4 and coords[0] == coords[-1]
                geom = Polygon(coords).buffer(0) if is_closed else LineString(coords)
            elif element_type == "relation":
                outer_lines = []
                for member in el.get("members", []):
                    if member.get("role", "outer") not in {"", "outer"}:
                        continue
                    pts = member.get("geometry", [])
                    if len(pts) >= 2:
                        outer_lines.append(
                            LineString([(p["lon"], p["lat"]) for p in pts])
                        )
                polygons = list(polygonize(unary_union(outer_lines))) if outer_lines else []
                if not polygons:
                    continue
                geom = unary_union(polygons).buffer(0)
            else:
                continue
            if not geom.is_valid or geom.is_empty:
                continue
            if not geom.intersects(bbox_poly):
                continue
            features.append(
                {
                    "geometry": geom.intersection(bbox_poly),
                    "water_type": wtype,
                    "name": tags.get("name", ""),
                }
            )
        except Exception:
            continue
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(features, crs="EPSG:4326")


def _fallback_gdf() -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame tagged 'fallback' for provenance tracking.

    Loads the bundled Bangalore water-body snapshot when available
    (data/water_bodies/bangalore_fallback.geojson).  Falls back to an
    empty GeoDataFrame if the file hasn't been committed yet.
    """
    _BUNDLED = Path(__file__).parent.parent.parent / "data" / "water_bodies" / "bangalore_fallback.geojson"
    if _BUNDLED.exists():
        try:
            gdf = gpd.read_file(str(_BUNDLED))
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf.attrs["source"] = "fallback"
            logger.info("Loaded bundled fallback water bodies (%d features).", len(gdf))
            return gdf
        except Exception as exc:
            logger.warning("Could not load bundled fallback GeoJSON: %s", exc)

    # File not present yet — return empty but still tag correctly
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf.attrs["source"] = "fallback"
    return gdf


def load_water_bodies(
    bounding_box: BoundingBox,
    data_dir=None,
    allow_network: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load all water body polygons for the bbox from OSM Overpass API.

    Resolution order:
    1. Local GeoJSON cache (instant, no network)
    2. Live Overpass API with 3 tenacity retries (exponential back-off)
    3. Fallback: empty GeoDataFrame tagged "fallback" — pipeline continues;
       caller is responsible for surfacing a warning to the user.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent. Coordinates are rounded to 4 decimal places for
        consistent cache-key generation.
    data_dir : str | Path | None
        Directory for local GeoJSON cache. Pass None to skip caching.
    allow_network : bool
        If False, skip the live API call (returns empty or cached data only).

    Returns
    -------
    gpd.GeoDataFrame
        Water body polygons with a ``source`` attribute:
        ``"osm_cache"`` | ``"osm_overpass"`` | ``"unavailable"`` | ``"fallback"``
    """
    # Round bbox coords to 4 dp — avoids cache misses from float noise
    min_lon = round(bounding_box.min_lon, 4)
    min_lat = round(bounding_box.min_lat, 4)
    max_lon = round(bounding_box.max_lon, 4)
    max_lat = round(bounding_box.max_lat, 4)

    cache_path = None
    if data_dir is not None:
        cache_dir = Path(data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ck = f"wb_{min_lon:.4f}_{min_lat:.4f}_{max_lon:.4f}_{max_lat:.4f}.geojson"
        cache_path = cache_dir / ck
        if cache_path.exists():
            try:
                gdf = gpd.read_file(str(cache_path))
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                gdf.attrs["source"] = "osm_cache"
                logger.info("Water bodies from cache: %d features.", len(gdf))
                return gdf
            except Exception as e:
                logger.warning("Cache read failed: %s", e)

    if not allow_network:
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        empty.attrs["source"] = "unavailable"
        return empty

    query = _QUERY.format(s=min_lat, w=min_lon, n=max_lat, e=max_lon)
    logger.info("Fetching water bodies from Overpass API for %s...", bounding_box)
    osm_data = _fetch(query)

    if osm_data is not None:
        gdf = _osm_to_gdf(osm_data, bounding_box)
        gdf.attrs["source"] = "osm_overpass"
        logger.info("Fetched %d water features from Overpass.", len(gdf))
        if cache_path is not None:
            try:
                if len(gdf) > 0:
                    gdf.to_file(str(cache_path), driver="GeoJSON")
                else:
                    cache_path.write_text(
                        json.dumps({"type": "FeatureCollection", "features": []}),
                        encoding="utf-8",
                    )
            except Exception:
                pass
        return gdf

    # All retries exhausted → fallback
    logger.warning(
        "Overpass API unavailable after retries. Falling back to empty water bodies."
    )
    return _fallback_gdf()
