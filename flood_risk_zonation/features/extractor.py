"""
Feature assembly and normalization for the Flood Risk Zonation System.

Orchestrates all terrain, hydrological, and rainfall feature computations
and assembles them into a single feature matrix aligned to the grid.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

from flood_risk_zonation.features.hydrological import (
    compute_distance_to_water,
    compute_drainage_density,
)
from flood_risk_zonation.features.rainfall_features import extract_rainfall_features
from flood_risk_zonation.features.terrain import (
    compute_aspect,
    compute_curvature,
    compute_slope,
    compute_twi,
)
from flood_risk_zonation.models import DrainageDataset, RainfallDataset, RasterDataset
from flood_risk_zonation.utils.validation import impute_missing_values

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "twi",
    "rainfall_mean_mm",
    "rainfall_max_24h_mm",
    "dist_water_m",
    "drainage_capacity",
    "population_density",
    "aspect_deg",
    "curvature",
]

# Physical validity ranges for each feature
FEATURE_RANGES = {
    "elevation_m": (-500.0, 9000.0),
    "slope_deg": (0.0, 90.0),
    "twi": (-50.0, 50.0),          # finite range after clamping
    "rainfall_mean_mm": (0.0, 20000.0),
    "rainfall_max_24h_mm": (0.0, 5000.0),
    "dist_water_m": (0.0, 10_000.0),
    "drainage_capacity": (0.0, 1.0),
    "population_density": (0.0, 1e7),
    "aspect_deg": (0.0, 360.0),
    "curvature": (-100.0, 100.0),
}


def _sample_raster_mean(raster: RasterDataset, grid: gpd.GeoDataFrame) -> np.ndarray:
    """Sample mean raster value per grid cell centroid (nearest-neighbour)."""
    from rasterio.transform import rowcol
    nrows, ncols = raster.array.shape
    lons = grid["centroid_lon"].values.astype(np.float64)
    lats = grid["centroid_lat"].values.astype(np.float64)
    if raster.crs is not None:
        from pyproj import CRS, Transformer
        raster_crs = CRS.from_user_input(raster.crs)
        if raster_crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            lons, lats = transformer.transform(lons, lats)
    values = np.full(len(grid), np.nan, dtype=np.float64)
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        row, col = rowcol(raster.transform, lon, lat)
        if 0 <= row < nrows and 0 <= col < ncols:
            values[i] = float(raster.array[row, col])
    return impute_missing_values(values)


def extract_features(
    grid: gpd.GeoDataFrame,
    elevation_raster: RasterDataset,
    rainfall_data: RainfallDataset,
    water_bodies: gpd.GeoDataFrame,
    population_raster: RasterDataset,
    drainage_data: DrainageDataset,
) -> gpd.GeoDataFrame:
    """
    Compute all features for each grid cell and assemble into the grid GeoDataFrame.

    Steps:
    1. Sample elevation per cell centroid
    2. Compute terrain features (slope, TWI, aspect, curvature) from the DEM
    3. Compute hydrological features (distance to water, drainage density)
    4. Extract rainfall features (mean annual, max 24h)
    5. Sample population density per cell centroid
    6. Impute any remaining NaN values
    7. Clamp features to physically valid ranges

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid GeoDataFrame with centroid_lat, centroid_lon columns.
    elevation_raster : RasterDataset
        DEM raster.
    rainfall_data : RainfallDataset
        Rainfall statistics raster.
    water_bodies : gpd.GeoDataFrame
        OSM water body polygons.
    population_raster : RasterDataset
        Population density raster.
    drainage_data : DrainageDataset
        Per-cell drainage capacity scores.

    Returns
    -------
    gpd.GeoDataFrame
        Input grid with all FEATURE_COLUMNS appended.
    """
    result = grid.copy()

    # --- Elevation ---
    elevation = _sample_raster_mean(elevation_raster, grid)
    result["elevation_m"] = elevation.astype(np.float32)

    # --- Terrain features from DEM array ---
    dem = impute_missing_values(elevation_raster.array)
    from pyproj import CRS
    elevation_crs = CRS.from_user_input(elevation_raster.crs)
    if elevation_crs.is_geographic:
        center_lat = float(grid["centroid_lat"].mean())
        x_size_m = abs(elevation_raster.transform.a) * 111_320.0 * np.cos(np.radians(center_lat))
        y_size_m = abs(elevation_raster.transform.e) * 111_320.0
        cell_size_m = max(1e-6, float(np.sqrt(x_size_m * y_size_m)))
    else:
        cell_size_m = max(1e-6, abs(float(elevation_raster.transform.a)))

    slope = compute_slope(dem, cell_size_m)
    twi = compute_twi(dem, cell_size_m)
    aspect = compute_aspect(dem)
    curvature = compute_curvature(dem, cell_size_m)

    # Map DEM pixels back to grid cells via centroid sampling
    def _map_dem_to_grid(dem_feature: np.ndarray) -> np.ndarray:
        """Sample a DEM-derived feature array at each grid cell centroid."""
        nrows, ncols = dem_feature.shape
        lons = grid["centroid_lon"].values.astype(np.float64)
        lats = grid["centroid_lat"].values.astype(np.float64)
        if elevation_crs != CRS.from_epsg(4326):
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", elevation_crs, always_xy=True)
            lons, lats = transformer.transform(lons, lats)
        values = np.full(len(grid), np.nan, dtype=np.float64)
        from rasterio.transform import rowcol
        for i, (lon, lat) in enumerate(zip(lons, lats)):
            row, col = rowcol(elevation_raster.transform, lon, lat)
            if 0 <= row < nrows and 0 <= col < ncols:
                values[i] = float(dem_feature[row, col])
        return impute_missing_values(values)

    result["slope_deg"] = _map_dem_to_grid(slope).astype(np.float32)
    result["twi"] = _map_dem_to_grid(twi).astype(np.float32)
    result["aspect_deg"] = _map_dem_to_grid(aspect).astype(np.float32)
    result["curvature"] = _map_dem_to_grid(curvature).astype(np.float32)

    # --- Hydrological features ---
    dist_water = compute_distance_to_water(result, water_bodies)
    result["dist_water_m"] = dist_water.astype(np.float32)

    drainage = compute_drainage_density(result, drainage_data)
    result["drainage_capacity"] = drainage.astype(np.float32)

    # --- Rainfall features ---
    result = extract_rainfall_features(result, rainfall_data)

    # --- Population density ---
    pop = _sample_raster_mean(population_raster, grid)
    result["population_density"] = pop.astype(np.float32)

    # --- Final imputation pass ---
    for col in FEATURE_COLUMNS:
        if col in result.columns:
            result[col] = impute_missing_values(result[col].values).astype(np.float32)

    # --- Clamp to physical ranges ---
    for col, (lo, hi) in FEATURE_RANGES.items():
        if col in result.columns:
            result[col] = np.clip(result[col].values, lo, hi).astype(np.float32)

    logger.debug("Feature extraction complete. Grid shape: %s", result.shape)
    return result
