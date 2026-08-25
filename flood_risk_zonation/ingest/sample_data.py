"""
Bundled offline sample data for the Flood Risk Zonation System.

Provides pre-defined BoundingBox regions and functions to generate
deterministic synthetic data for each, used when:
  - the "Use offline sample data" sidebar checkbox is checked, or
  - all live API retries have been exhausted (automatic fallback).

No network calls are made from this module.
"""
from __future__ import annotations

from typing import NamedTuple

import geopandas as gpd

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.ingest.elevation import generate_synthetic_elevation
from flood_risk_zonation.ingest.rainfall import generate_synthetic_rainfall
from flood_risk_zonation.models import RainfallDataset, RasterDataset


class DemoRegion(NamedTuple):
    name: str
    bbox: BoundingBox
    # Elevation parameters tuned to each region's real character
    base_elevation_m: float
    relief_m: float
    # Rainfall parameters (mean annual mm, rough max-24h mm)
    mean_rainfall_mm: float
    seed: int


DEMO_REGIONS: dict[str, DemoRegion] = {
    "Bangalore (Gottigere)": DemoRegion(
        name="Bangalore (Gottigere)",
        bbox=BoundingBox(min_lon=77.55, min_lat=12.84, max_lon=77.62, max_lat=12.91),
        base_elevation_m=880.0,
        relief_m=60.0,
        mean_rainfall_mm=970.0,
        seed=42,
    ),
    "Chennai Marina (Coastal)": DemoRegion(
        name="Chennai Marina (Coastal)",
        bbox=BoundingBox(min_lon=80.24, min_lat=12.98, max_lon=80.31, max_lat=13.05),
        base_elevation_m=6.0,
        relief_m=20.0,
        mean_rainfall_mm=1400.0,
        seed=7,
    ),
    "Dal Lake, Srinagar": DemoRegion(
        name="Dal Lake, Srinagar",
        bbox=BoundingBox(min_lon=74.83, min_lat=34.07, max_lon=74.90, max_lat=34.14),
        base_elevation_m=1580.0,
        relief_m=80.0,
        mean_rainfall_mm=650.0,
        seed=13,
    ),
}


def get_demo_elevation(region: DemoRegion, resolution_m: float = 30.0) -> RasterDataset:
    """Return a deterministic synthetic DEM for the given demo region."""
    raster = generate_synthetic_elevation(
        region.bbox,
        resolution_m=resolution_m,
        base_elevation_m=region.base_elevation_m,
        relief_m=region.relief_m,
        seed=region.seed,
    )
    # Tag the source so provenance tracking works correctly
    import dataclasses
    return dataclasses.replace(raster, source="offline_sample")


def get_demo_rainfall(region: DemoRegion, resolution_m: float = 1000.0) -> RainfallDataset:
    """Return deterministic synthetic rainfall for the given demo region."""
    import numpy as np
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    from datetime import date

    raster = generate_synthetic_rainfall(
        region.bbox,
        resolution_m=resolution_m,
        seed=region.seed,
    )
    # Scale mean annual to region-calibrated value
    raw_mean = raster.mean_annual_mm
    if raw_mean.max() > raw_mean.min():
        normalised = (raw_mean - raw_mean.min()) / (raw_mean.max() - raw_mean.min())
    else:
        normalised = np.ones_like(raw_mean) * 0.5
    calibrated_mean = (region.mean_rainfall_mm * 0.8 + normalised * region.mean_rainfall_mm * 0.4).astype(
        np.float32
    )
    calibrated_max24h = (calibrated_mean / 365.0 * 8.0).astype(np.float32)

    import dataclasses
    return dataclasses.replace(
        raster,
        mean_annual_mm=calibrated_mean,
        max_24h_mm=calibrated_max24h,
        source="offline_sample",
    )


def get_demo_water_bodies(region: DemoRegion) -> gpd.GeoDataFrame:
    """
    Return an empty GeoDataFrame tagged as offline_sample.

    For demo regions we rely on elevation-based water masking (elevation ≤ 1m)
    rather than fabricating polygon geometry.
    """
    gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf.attrs["source"] = "offline_sample"
    return gdf
