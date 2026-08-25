"""
Rainfall feature extraction for the Flood Risk Zonation System.

Samples mean_annual_mm and max_24h_mm from a RainfallDataset raster
at each grid cell centroid using bilinear interpolation.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from rasterio.transform import rowcol

from flood_risk_zonation.models import RainfallDataset
from flood_risk_zonation.utils.validation import impute_missing_values


def _sample_raster_at_points(
    array: np.ndarray,
    transform,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Sample a 2D raster array at given lon/lat points using nearest-neighbour lookup.
    Points outside the raster extent are filled with np.nan.

    Parameters
    ----------
    array : np.ndarray
        2D raster array (nrows, ncols).
    transform : Affine
        Rasterio affine transform for the raster.
    lons : np.ndarray
        1D array of longitude values.
    lats : np.ndarray
        1D array of latitude values.

    Returns
    -------
    np.ndarray
        1D array of sampled values, same length as lons/lats.
    """
    nrows, ncols = array.shape
    values = np.full(len(lons), np.nan, dtype=np.float64)

    for i, (lon, lat) in enumerate(zip(lons, lats)):
        # Convert geographic coordinates to pixel row/col
        row, col = rowcol(transform, lon, lat)
        if 0 <= row < nrows and 0 <= col < ncols:
            values[i] = float(array[row, col])
        # else: leave as NaN (will be imputed)

    return values


def extract_rainfall_features(
    grid: gpd.GeoDataFrame,
    rainfall_dataset: RainfallDataset,
) -> gpd.GeoDataFrame:
    """
    Spatially sample mean_annual_mm and max_24h_mm from the RainfallDataset
    raster at each grid cell centroid.

    Cells that fall outside the rainfall raster extent are filled with the
    dataset mean via impute_missing_values.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid GeoDataFrame with centroid_lat and centroid_lon columns.
    rainfall_dataset : RainfallDataset
        Rainfall statistics raster.

    Returns
    -------
    gpd.GeoDataFrame
        Input grid with 'rainfall_mean_mm' and 'rainfall_max_24h_mm' columns appended.
    """
    lons = grid["centroid_lon"].values.astype(np.float64)
    lats = grid["centroid_lat"].values.astype(np.float64)
    if rainfall_dataset.crs is not None:
        from pyproj import CRS, Transformer
        rainfall_crs = CRS.from_user_input(rainfall_dataset.crs)
        if rainfall_crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs("EPSG:4326", rainfall_crs, always_xy=True)
            lons, lats = transformer.transform(lons, lats)

    mean_annual = _sample_raster_at_points(
        rainfall_dataset.mean_annual_mm,
        rainfall_dataset.transform,
        lons, lats,
    )
    max_24h = _sample_raster_at_points(
        rainfall_dataset.max_24h_mm,
        rainfall_dataset.transform,
        lons, lats,
    )

    # Impute any NaN values (cells outside raster extent)
    mean_annual = impute_missing_values(mean_annual)
    max_24h = impute_missing_values(max_24h)

    result = grid.copy()
    result["rainfall_mean_mm"] = mean_annual.astype(np.float32)
    result["rainfall_max_24h_mm"] = max_24h.astype(np.float32)

    return result
