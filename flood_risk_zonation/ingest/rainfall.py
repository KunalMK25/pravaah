"""
Rainfall data ingestion for the Flood Risk Zonation System.

Supports GPM IMERG and IMD gridded data (GeoTIFF/NetCDF), with a
synthetic fallback using Gaussian-smoothed random arrays.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from scipy.ndimage import gaussian_filter

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.exceptions import DataIngestionError
from flood_risk_zonation.models import RainfallDataset
from flood_risk_zonation.utils.validation import impute_missing_values

logger = logging.getLogger(__name__)


def load_rainfall(bounding_box: BoundingBox, data_dir: Path | str) -> RainfallDataset:
    """
    Load GPM IMERG or IMD gridded rainfall data clipped to the bounding box.

    Searches data_dir for .tif/.tiff files. Falls back to synthetic data
    if no files are found, logging a warning.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent.
    data_dir : Path | str
        Directory containing rainfall GeoTIFF files.

    Returns
    -------
    RainfallDataset
        Rainfall statistics for the bounding box.
    """
    data_dir = Path(data_dir)
    tif_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))

    if not tif_files:
        logger.warning(
            "No rainfall GeoTIFF files found in %s. "
            "Falling back to synthetic rainfall data.", data_dir
        )
        return generate_synthetic_rainfall(bounding_box, seed=42)

    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    errors: list[str] = []
    for tif_path in sorted(tif_files):
        try:
            with rasterio.open(tif_path) as src:
                if src.crs is None:
                    continue
                left, bottom, right, top = transform_bounds(
                    "EPSG:4326",
                    src.crs,
                    bounding_box.min_lon,
                    bounding_box.min_lat,
                    bounding_box.max_lon,
                    bounding_box.max_lat,
                )
                bounds = src.bounds
                if right <= bounds.left or left >= bounds.right or top <= bounds.bottom or bottom >= bounds.top:
                    continue
                window = window_from_bounds(
                    max(left, bounds.left),
                    max(bottom, bounds.bottom),
                    min(right, bounds.right),
                    min(top, bounds.top),
                    src.transform,
                ).round_offsets().round_lengths()
                mean_annual = src.read(1, window=window).astype(np.float32)
                if mean_annual.size == 0:
                    continue
                if src.count >= 2:
                    max_24h = src.read(2, window=window).astype(np.float32)
                else:
                    max_24h = (mean_annual / 365.0 * 5.0).astype(np.float32)
                if src.nodata is not None:
                    mean_annual[mean_annual == src.nodata] = np.nan
                    max_24h[max_24h == src.nodata] = np.nan
                mean_annual = impute_missing_values(mean_annual)
                max_24h = impute_missing_values(max_24h)
                return RainfallDataset(
                    mean_annual_mm=mean_annual,
                    max_24h_mm=max_24h,
                    transform=src.window_transform(window),
                    crs=src.crs,
                    temporal_range=(date(2001, 1, 1), date(2023, 12, 31)),
                    source=str(tif_path),
                )
        except Exception as exc:
            errors.append(f"{tif_path.name}: {exc}")

    logger.warning(
        "No rainfall raster covers the requested area%s. Using synthetic fallback.",
        f" ({'; '.join(errors)})" if errors else "",
    )
    return generate_synthetic_rainfall(bounding_box, seed=42)


def generate_synthetic_rainfall(
    bounding_box: BoundingBox,
    resolution_m: float = 1000.0,
    seed: int = 42,
) -> RainfallDataset:
    """
    Generate synthetic rainfall data using Gaussian-smoothed random arrays.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent.
    resolution_m : float
        Pixel resolution in metres.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    RainfallDataset
        Synthetic rainfall dataset.
    """
    deg_per_m = 1.0 / 111_320.0
    resolution_deg = resolution_m * deg_per_m

    width_deg = bounding_box.max_lon - bounding_box.min_lon
    height_deg = bounding_box.max_lat - bounding_box.min_lat

    ncols = max(2, int(width_deg / resolution_deg))
    nrows = max(2, int(height_deg / resolution_deg))

    rng = np.random.default_rng(seed)

    # Mean annual rainfall: 500–2000 mm range
    raw_mean = rng.random((nrows, ncols)).astype(np.float32)
    smoothed_mean = gaussian_filter(raw_mean, sigma=max(1, min(nrows, ncols) // 5))
    smoothed_mean = (smoothed_mean - smoothed_mean.min()) / (smoothed_mean.max() - smoothed_mean.min() + 1e-9)
    mean_annual = (500.0 + smoothed_mean * 1500.0).astype(np.float32)

    # Max 24h rainfall: 20–200 mm range, correlated with mean annual
    raw_max = rng.random((nrows, ncols)).astype(np.float32)
    smoothed_max = gaussian_filter(raw_max, sigma=max(1, min(nrows, ncols) // 5))
    smoothed_max = (smoothed_max - smoothed_max.min()) / (smoothed_max.max() - smoothed_max.min() + 1e-9)
    max_24h = (20.0 + smoothed_max * 180.0).astype(np.float32)

    transform = from_bounds(
        bounding_box.min_lon, bounding_box.min_lat,
        bounding_box.max_lon, bounding_box.max_lat,
        ncols, nrows,
    )
    crs = CRS.from_epsg(4326)

    return RainfallDataset(
        mean_annual_mm=mean_annual,
        max_24h_mm=max_24h,
        transform=transform,
        crs=crs,
        temporal_range=(date(2001, 1, 1), date(2023, 12, 31)),
        source="synthetic",
    )
