"""
CRS (Coordinate Reference System) utility helpers for the Flood Risk Zonation System.

Functions
---------
degrees_to_metres       — convert a degree distance to metres at a given latitude.
reproject_raster        — reproject a RasterDataset to a target CRS.
reproject_geodataframe  — reproject a GeoDataFrame to a target CRS.
"""

from __future__ import annotations

import copy
from math import cos, radians

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling

from flood_risk_zonation.models import RasterDataset


def degrees_to_metres(degrees: float, latitude: float) -> float:
    """
    Convert a degree distance to metres at a given latitude.

    Uses the equirectangular approximation:
        metres = degrees * 111_320 * cos(radians(latitude))

    Parameters
    ----------
    degrees : float
        Distance in decimal degrees.
    latitude : float
        Reference latitude in decimal degrees (WGS84).

    Returns
    -------
    float
        Approximate distance in metres.
    """
    return degrees * 111_320 * cos(radians(latitude))


def reproject_raster(raster_dataset: RasterDataset, target_crs: str) -> RasterDataset:
    """
    Reproject a RasterDataset to the target CRS using bilinear resampling.

    If the source and target CRS are the same, a copy of the dataset is
    returned without performing any reprojection.

    Parameters
    ----------
    raster_dataset : RasterDataset
        Source raster to reproject.
    target_crs : str
        Target coordinate reference system as an EPSG string (e.g. "EPSG:4326").

    Returns
    -------
    RasterDataset
        New RasterDataset with the target CRS, updated transform, and
        reprojected array values.
    """
    from rasterio.crs import CRS as RasterioCRS

    src_crs = raster_dataset.crs
    dst_crs = RasterioCRS.from_string(target_crs)

    # Normalise source CRS to a rasterio CRS for comparison
    if hasattr(src_crs, "to_wkt"):
        src_crs_obj = RasterioCRS.from_wkt(src_crs.to_wkt())
    else:
        src_crs_obj = RasterioCRS.from_string(str(src_crs))

    # If source and target CRS are the same, return a copy
    if src_crs_obj == dst_crs:
        return RasterDataset(
            array=raster_dataset.array.copy(),
            transform=raster_dataset.transform,
            crs=raster_dataset.crs,
            nodata=raster_dataset.nodata,
            source=raster_dataset.source,
        )

    src_array = raster_dataset.array
    src_height, src_width = src_array.shape

    # Calculate the default transform and shape for the reprojected raster
    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs_obj,
        dst_crs,
        src_width,
        src_height,
        left=raster_dataset.transform.c,
        bottom=raster_dataset.transform.f + src_height * raster_dataset.transform.e,
        right=raster_dataset.transform.c + src_width * raster_dataset.transform.a,
        top=raster_dataset.transform.f,
    )

    dst_array = np.empty((dst_height, dst_width), dtype=np.float32)

    nodata_val = raster_dataset.nodata if raster_dataset.nodata is not None else np.nan

    rasterio.warp.reproject(
        source=src_array,
        destination=dst_array,
        src_transform=raster_dataset.transform,
        src_crs=src_crs_obj,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata_val,
        dst_nodata=nodata_val,
    )

    return RasterDataset(
        array=dst_array,
        transform=dst_transform,
        crs=dst_crs,
        nodata=raster_dataset.nodata,
        source=raster_dataset.source,
    )


def reproject_geodataframe(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to the target CRS.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Source GeoDataFrame to reproject.
    target_crs : str
        Target coordinate reference system (e.g. "EPSG:4326").

    Returns
    -------
    gpd.GeoDataFrame
        New GeoDataFrame reprojected to the target CRS.
    """
    return gdf.to_crs(target_crs)
