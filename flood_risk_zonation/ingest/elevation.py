"""
Elevation data ingestion for the Flood Risk Zonation System.

Provides loaders for real SRTM GeoTIFF files and synthetic DEM generation
for demo/test mode using Perlin noise.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.exceptions import DataIngestionError
from flood_risk_zonation.models import RasterDataset

logger = logging.getLogger(__name__)


def load_elevation(bounding_box: BoundingBox, data_dir: Path | str) -> RasterDataset:
    """
    Load a SRTM elevation GeoTIFF clipped to the bounding box.

    Searches data_dir for any .tif or .tiff file. Raises DataIngestionError
    if no file is found or the bounding box falls outside coverage.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent to clip to.
    data_dir : Path | str
        Directory containing SRTM GeoTIFF files.

    Returns
    -------
    RasterDataset
        Elevation raster clipped to the bounding box.

    Raises
    ------
    DataIngestionError
        If no GeoTIFF file is found or the bbox is outside coverage.
    """
    data_path = Path(data_dir)
    if data_path.is_file():
        tif_files = [data_path]
    else:
        tif_files = sorted(data_path.glob("*.tif")) + sorted(data_path.glob("*.tiff"))

    if not tif_files:
        raise DataIngestionError(
            f"No SRTM GeoTIFF files found in {data_path}. "
            "Use generate_synthetic_elevation() for demo/test mode."
        )

    errors: list[str] = []
    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                if src.crs is None:
                    errors.append(f"{tif_path.name}: missing CRS")
                    continue
                target_bounds = rasterio.warp.transform_bounds(
                    "EPSG:4326",
                    src.crs,
                    bounding_box.min_lon,
                    bounding_box.min_lat,
                    bounding_box.max_lon,
                    bounding_box.max_lat,
                )
                left, bottom, right, top = target_bounds
                bounds = src.bounds
                if right <= bounds.left or left >= bounds.right or top <= bounds.bottom or bottom >= bounds.top:
                    continue

                # Clip partially overlapping requests to the source raster.
                left = max(left, bounds.left)
                bottom = max(bottom, bounds.bottom)
                right = min(right, bounds.right)
                top = min(top, bounds.top)
                from rasterio.windows import from_bounds as window_from_bounds
                window = window_from_bounds(left, bottom, right, top, src.transform)
                window = window.round_offsets().round_lengths()
                array = src.read(1, window=window).astype(np.float32)
                if array.size == 0:
                    continue
                transform = src.window_transform(window)
                nodata = src.nodata
                if nodata is not None:
                    array[array == nodata] = np.nan

                return RasterDataset(
                    array=array,
                    transform=transform,
                    crs=src.crs,
                    nodata=nodata,
                    source=str(tif_path),
                )
        except (rasterio.errors.RasterioIOError, ValueError) as exc:
            errors.append(f"{tif_path.name}: {exc}")

    detail = f" ({'; '.join(errors)})" if errors else ""
    raise DataIngestionError(
        f"No elevation GeoTIFF in {data_path} covers {bounding_box}.{detail}"
    )


def resample_raster(raster_dataset: RasterDataset, target_resolution_m: float) -> RasterDataset:
    """
    Resample a RasterDataset to a target resolution using bilinear interpolation.

    Parameters
    ----------
    raster_dataset : RasterDataset
        Source raster to resample.
    target_resolution_m : float
        Target resolution in metres.

    Returns
    -------
    RasterDataset
        Resampled raster at the target resolution.
    """
    src_array = raster_dataset.array
    src_transform = raster_dataset.transform
    src_height, src_width = src_array.shape

    # Current pixel size in degrees (approximate)
    pixel_size_deg = abs(src_transform.a)
    # Convert target resolution to degrees (approximate at equator)
    target_deg = target_resolution_m / 111_320.0

    scale_factor = pixel_size_deg / target_deg
    new_height = max(1, int(src_height * scale_factor))
    new_width = max(1, int(src_width * scale_factor))

    dst_array = np.empty((new_height, new_width), dtype=np.float32)

    # Compute new transform
    left = src_transform.c
    top = src_transform.f
    right = left + src_width * src_transform.a
    bottom = top + src_height * src_transform.e

    new_transform = from_bounds(left, bottom, right, top, new_width, new_height)

    rasterio.warp.reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=raster_dataset.crs,
        dst_transform=new_transform,
        dst_crs=raster_dataset.crs,
        resampling=Resampling.bilinear,
    )

    return RasterDataset(
        array=dst_array,
        transform=new_transform,
        crs=raster_dataset.crs,
        nodata=raster_dataset.nodata,
        source=raster_dataset.source,
    )


def generate_synthetic_elevation(
    bounding_box: BoundingBox,
    resolution_m: float = 30.0,
    base_elevation_m: float = 50.0,
    relief_m: float = 100.0,
    seed: int = 42,
) -> RasterDataset:
    """
    Generate a synthetic DEM using random noise smoothed with a Gaussian filter
    to simulate realistic terrain with valleys, ridges, and flat plains.

    Falls back gracefully if the 'noise' library is not installed.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent.
    resolution_m : float
        Pixel resolution in metres.
    base_elevation_m : float
        Mean elevation of the synthetic terrain.
    relief_m : float
        Peak-to-peak elevation range.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    RasterDataset
        Synthetic elevation raster.
    """
    from scipy.ndimage import gaussian_filter

    deg_per_m = 1.0 / 111_320.0
    resolution_deg = resolution_m * deg_per_m

    width_deg = bounding_box.max_lon - bounding_box.min_lon
    height_deg = bounding_box.max_lat - bounding_box.min_lat

    ncols = max(2, int(width_deg / resolution_deg))
    nrows = max(2, int(height_deg / resolution_deg))

    rng = np.random.default_rng(seed)
    raw = rng.random((nrows, ncols)).astype(np.float32)
    # Smooth to create realistic terrain
    smoothed = gaussian_filter(raw, sigma=max(1, min(nrows, ncols) // 10))
    # Scale to [base, base + relief]
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-9)
    array = (base_elevation_m + smoothed * relief_m).astype(np.float32)

    transform = from_bounds(
        bounding_box.min_lon, bounding_box.min_lat,
        bounding_box.max_lon, bounding_box.max_lat,
        ncols, nrows,
    )

    from rasterio.crs import CRS
    crs = CRS.from_epsg(4326)

    return RasterDataset(
        array=array,
        transform=transform,
        crs=crs,
        nodata=None,
        source="synthetic",
    )


def fetch_elevation_api(
    bounding_box: BoundingBox,
    resolution_m: float = 500.0,
    timeout: int = 15,
) -> RasterDataset | None:
    """
    Fetch SRTM elevation from the OpenTopoData API for any bbox worldwide.

    Returns a RasterDataset with real elevation values (ocean = 0 or NaN),
    or None if the request fails or times out.

    Ocean pixels in SRTM are returned as 0 by OpenTopoData, making them
    easily distinguishable from land (typically > 0 m).

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent.
    resolution_m : float
        Sampling resolution in metres. Coarser = fewer API points = faster.
    timeout : int
        Request timeout in seconds.
    """
    try:
        import requests
        from math import cos, radians, ceil
        from rasterio.transform import from_bounds as _from_bounds
        from rasterio.crs import CRS

        # Build a regular grid of sample points
        center_lat = (bounding_box.min_lat + bounding_box.max_lat) / 2.0
        deg_per_m_lat = 1.0 / 111_320.0
        deg_per_m_lon = 1.0 / (111_320.0 * cos(radians(center_lat)))

        step_lat = resolution_m * deg_per_m_lat
        step_lon = resolution_m * deg_per_m_lon

        lats = np.arange(bounding_box.min_lat, bounding_box.max_lat, step_lat)
        lons = np.arange(bounding_box.min_lon, bounding_box.max_lon, step_lon)

        if len(lats) == 0 or len(lons) == 0:
            return None

        # Limit to 100 points to stay within API free tier
        max_pts = 100
        step_r = max(1, len(lats) * len(lons) // max_pts)
        points = [
            (lat, lon)
            for i, lat in enumerate(lats)
            for j, lon in enumerate(lons)
            if (i * len(lons) + j) % step_r == 0
        ][:max_pts]

        if not points:
            return None

        # Query OpenTopoData SRTM30m dataset
        locations = "|".join(f"{lat},{lon}" for lat, lon in points)
        url = f"https://api.opentopodata.org/v1/srtm30m?locations={locations}"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            logger.warning("OpenTopoData API returned %d", resp.status_code)
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        # Build a coarse elevation array from returned points
        nrows = len(lats)
        ncols = len(lons)
        array = np.full((nrows, ncols), np.nan, dtype=np.float32)

        for res in results:
            lat = res.get("location", {}).get("lat")
            lon = res.get("location", {}).get("lng")
            elev = res.get("elevation")
            if lat is None or lon is None or elev is None:
                continue
            # Find nearest row/col
            r = int(round((lat - bounding_box.min_lat) / step_lat))
            c = int(round((lon - bounding_box.min_lon) / step_lon))
            if 0 <= r < nrows and 0 <= c < ncols:
                array[r, c] = float(elev) if elev is not None else 0.0

        # Fill NaN with nearest valid value - use high value (not 0!) to avoid
        # treating unsampled cells as ocean
        mask_nan = np.isnan(array)
        if mask_nan.any() and not mask_nan.all():
            col_medians = np.nanmedian(array, axis=0)
            for c in range(ncols):
                nan_rows = np.where(np.isnan(array[:, c]))[0]
                if len(nan_rows) > 0:
                    fill = col_medians[c] if not np.isnan(col_medians[c]) else 100.0
                    array[nan_rows, c] = fill
        # Replace any remaining NaN with 100m (safe default — won't trigger ocean mask)
        array = np.where(np.isnan(array), 100.0, array).astype(np.float32)

        transform = _from_bounds(
            bounding_box.min_lon, bounding_box.min_lat,
            bounding_box.max_lon, bounding_box.max_lat,
            ncols, nrows,
        )

        logger.info(
            "OpenTopoData SRTM fetched: %d points, elevation range %.0f–%.0f m",
            len(results), float(array.min()), float(array.max()),
        )

        return RasterDataset(
            array=array,
            transform=transform,
            crs=CRS.from_epsg(4326),
            nodata=None,
            source="opentopodata_srtm",
        )

    except Exception as exc:
        logger.warning("OpenTopoData API fetch failed: %s", exc)
        return None
