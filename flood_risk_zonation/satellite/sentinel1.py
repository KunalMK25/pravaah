"""
PRAVAAH-AI — Sentinel-1 satellite flood observation integration.

Supports ingestion of Sentinel-1-derived flood observations from:
- GeoTIFF raster flood masks
- GeoJSON flood polygons

This module does NOT perform raw SAR processing.
It assumes inputs are pre-processed flood products with known methods.

Processing pipeline:
  1. Load observation from file/provider
  2. Validate geometry and CRS
  3. Align with analysis grid if needed
  4. Compute statistics
  5. Return result with complete provenance

Scientific integrity:
  - Never fabricates satellite observations
  - Explicit UNKNOWN/UNAVAILABLE states when data unavailable
  - Temporal information preserved
  - Processing method tracked
  - Confidence reflects data quality, not statistical confidence
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.satellite.observations import (
    RasterFloodMaskProvider,
    VectorFloodPolygonProvider,
)
from flood_risk_zonation.satellite.provider import UnknownSentinel1Provider
from flood_risk_zonation.satellite.result import Sentinel1ObservationResult

logger = logging.getLogger(__name__)


def load_sentinel1_observation(
    bbox: BoundingBox,
    geotiff_path: Optional[str | Path] = None,
    geojson_path: Optional[str | Path] = None,
    acquisition_date: Optional[str] = None,
    max_days_old: int = 365,
) -> Sentinel1ObservationResult:
    """
    Load Sentinel-1 flood observation for an area.

    Attempts providers in order:
    1. GeoTIFF (if path provided)
    2. GeoJSON (if path provided)
    3. UNKNOWN fallback (explicit unavailable state)

    Parameters
    ----------
    bbox : BoundingBox
        Analysis area
    geotiff_path : str | Path | None
        Path to GeoTIFF flood mask file (optional)
    geojson_path : str | Path | None
        Path to GeoJSON flood polygons file (optional)
    acquisition_date : str | None
        Target acquisition date (ISO-8601, e.g., "2024-08-26")
        Ignored by local file providers.
    max_days_old : int
        Reject observations older than this many days (default: 365)

    Returns
    -------
    Sentinel1ObservationResult
        Always returns a result. Never raises exception.
        If data unavailable, returns UNKNOWN or UNAVAILABLE result.
    """
    bbox_tuple = (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)

    # Try GeoTIFF first
    if geotiff_path is not None:
        logger.info("Attempting to load Sentinel-1 from GeoTIFF: %s", geotiff_path)
        provider = RasterFloodMaskProvider(geotiff_path)
        result = provider.load_observation(
            bbox_tuple,
            acquisition_date=acquisition_date,
            max_days_old=max_days_old,
        )
        if result.observation_status == "OBSERVED":
            logger.info("Successfully loaded Sentinel-1 from GeoTIFF")
            return result
        logger.debug("GeoTIFF provider failed or unavailable; trying next provider")

    # Try GeoJSON next
    if geojson_path is not None:
        logger.info("Attempting to load Sentinel-1 from GeoJSON: %s", geojson_path)
        provider = VectorFloodPolygonProvider(geojson_path)
        result = provider.load_observation(
            bbox_tuple,
            acquisition_date=acquisition_date,
            max_days_old=max_days_old,
        )
        if result.observation_status == "OBSERVED":
            logger.info("Successfully loaded Sentinel-1 from GeoJSON")
            return result
        logger.debug("GeoJSON provider failed or unavailable; trying next provider")

    # Fallback: UNKNOWN
    logger.info("No Sentinel-1 observation available; returning UNKNOWN state")
    provider = UnknownSentinel1Provider()
    return provider.load_observation(
        bbox_tuple,
        acquisition_date=acquisition_date,
        max_days_old=max_days_old,
    )
