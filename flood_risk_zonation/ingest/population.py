"""
Population density raster ingestion for the Flood Risk Zonation System.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.exceptions import DataIngestionError
from flood_risk_zonation.models import RasterDataset

logger = logging.getLogger(__name__)


def load_population(bounding_box: BoundingBox, data_dir: Path | str) -> RasterDataset:
    """
    Load a population density raster (e.g. WorldPop GeoTIFF) clipped to the bounding box.

    Falls back to synthetic population data if no files are found.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent.
    data_dir : Path | str
        Directory containing population density GeoTIFF files.

    Returns
    -------
    RasterDataset
        Population density raster (persons per km²).
    """
    data_dir = Path(data_dir)
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

    if not tif_files:
        logger.warning(
            "No population GeoTIFF files found in %s. Using synthetic population data.", data_dir
        )
        return _generate_synthetic_population(bounding_box)

    try:
        import rasterio
        from rasterio.windows import from_bounds as window_from_bounds

        with rasterio.open(tif_files[0]) as src:
            window = window_from_bounds(
                bounding_box.min_lon, bounding_box.min_lat,
                bounding_box.max_lon, bounding_box.max_lat,
                src.transform,
            )
            array = src.read(1, window=window).astype(np.float32)
            array = np.clip(array, 0, None)  # population density >= 0
            transform = src.window_transform(window)
            crs = src.crs

        return RasterDataset(
            array=array,
            transform=transform,
            crs=crs,
            nodata=None,
            source=str(tif_files[0]),
        )
    except Exception as exc:
        logger.warning("Failed to load population data: %s. Using synthetic fallback.", exc)
        return _generate_synthetic_population(bounding_box)


def _generate_synthetic_population(
    bounding_box: BoundingBox,
    resolution_m: float = 1000.0,
    seed: int = 42,
) -> RasterDataset:
    """Generate synthetic population density data."""
    deg_per_m = 1.0 / 111_320.0
    resolution_deg = resolution_m * deg_per_m

    ncols = max(2, int((bounding_box.max_lon - bounding_box.min_lon) / resolution_deg))
    nrows = max(2, int((bounding_box.max_lat - bounding_box.min_lat) / resolution_deg))

    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=500.0, size=(nrows, ncols)).astype(np.float32)
    smoothed = gaussian_filter(raw, sigma=max(1, min(nrows, ncols) // 5))

    transform = from_bounds(
        bounding_box.min_lon, bounding_box.min_lat,
        bounding_box.max_lon, bounding_box.max_lat,
        ncols, nrows,
    )
    crs = CRS.from_epsg(4326)

    return RasterDataset(
        array=smoothed,
        transform=transform,
        crs=crs,
        nodata=None,
        source="synthetic",
    )
