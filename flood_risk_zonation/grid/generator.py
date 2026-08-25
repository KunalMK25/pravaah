"""
Grid generation engine for the Flood Risk Zonation System.

Converts a BoundingBox into a regular rectangular grid of Shapely Polygon
cells, each approximately cell_size_meters × cell_size_meters.
"""
from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.exceptions import ConfigurationError


def generate_grid(
    bounding_box: BoundingBox,
    cell_size_meters: float = 500.0,
    crs: str = "EPSG:4326",
    max_cells: int = 100_000,
) -> gpd.GeoDataFrame:
    """
    Partition the bounding box into a regular grid of square cells.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent (min_lon, min_lat, max_lon, max_lat) in WGS84.
    cell_size_meters : float
        Approximate cell edge length in metres. Converted to degrees using
        the latitude-dependent formula:
            cell_deg = cell_size_meters / (111_320 * cos(radians(center_lat)))
    crs : str
        Output coordinate reference system (default WGS84 EPSG:4326).

    Returns
    -------
    gpd.GeoDataFrame
        One row per grid cell with columns:
        cell_id (str), geometry (Polygon), centroid_lat (float), centroid_lon (float).
    """
    center_lat = (bounding_box.min_lat + bounding_box.max_lat) / 2.0
    lat_rad = math.radians(center_lat)

    # Convert cell size from metres to degrees
    cell_deg_lat = cell_size_meters / 111_320.0
    cell_deg_lon = cell_size_meters / (111_320.0 * math.cos(lat_rad))

    n_lon = math.ceil((bounding_box.max_lon - bounding_box.min_lon) / cell_deg_lon)
    n_lat = math.ceil((bounding_box.max_lat - bounding_box.min_lat) / cell_deg_lat)
    estimated_cells = n_lon * n_lat
    if estimated_cells > max_cells:
        raise ConfigurationError(
            f"Requested grid would contain approximately {estimated_cells:,} cells, "
            f"exceeding the configured limit of {max_cells:,}. Reduce the area or "
            "choose a larger cell size."
        )

    # Generate grid coordinates only after the allocation guard.
    lons = np.arange(bounding_box.min_lon, bounding_box.max_lon, cell_deg_lon)
    lats = np.arange(bounding_box.min_lat, bounding_box.max_lat, cell_deg_lat)

    cells = []
    cell_ids = []
    centroid_lats = []
    centroid_lons = []

    for row_idx, lat in enumerate(lats):
        for col_idx, lon in enumerate(lons):
            # Create cell polygon
            min_lon = lon
            max_lon = min(lon + cell_deg_lon, bounding_box.max_lon)
            min_lat = lat
            max_lat = min(lat + cell_deg_lat, bounding_box.max_lat)

            polygon = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat),
                (min_lon, min_lat),
            ])

            cells.append(polygon)
            cell_ids.append(f"{row_idx}_{col_idx}")
            centroid_lats.append((min_lat + max_lat) / 2.0)
            centroid_lons.append((min_lon + max_lon) / 2.0)

    gdf = gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "centroid_lat": centroid_lats,
            "centroid_lon": centroid_lons,
        },
        geometry=cells,
        crs=crs,
    )

    return gdf
