"""Sentinel-1 flood observation implementations (raster and vector)."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import shape

from flood_risk_zonation.satellite.provider import Sentinel1Provider
from flood_risk_zonation.satellite.result import Sentinel1ObservationResult

logger = logging.getLogger(__name__)


class RasterFloodMaskProvider(Sentinel1Provider):
    """
    Load Sentinel-1-derived flood observations from GeoTIFF raster flood masks.

    Expected raster properties:
    - Binary (0 = no flood, 1 = flood) or float [0–1] (inundation fraction)
    - Georeferenced (GeoTIFF with geotransform)
    - CRS metadata present
    - NoData value documented

    This provider does NOT perform raw SAR processing.
    It assumes the input is a pre-processed flood mask.
    """

    provider_type = "raster_geotiff"

    def __init__(self, geotiff_path: str | Path):
        """
        Initialize with path to GeoTIFF flood mask.

        Parameters
        ----------
        geotiff_path : str | Path
            Path to GeoTIFF file
        """
        self.geotiff_path = Path(geotiff_path)
        if not self.geotiff_path.exists():
            logger.warning("GeoTIFF file does not exist: %s", geotiff_path)

    def load_observation(
        self,
        bbox: tuple[float, float, float, float],
        acquisition_date: str | None = None,
        max_days_old: int = 365,
    ) -> Sentinel1ObservationResult:
        """
        Load flood observation from GeoTIFF raster.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            (min_lon, min_lat, max_lon, max_lat)
        acquisition_date : str | None
            Ignored (raster is single timestamp)
        max_days_old : int
            Ignored (raster timestamp not checked)

        Returns
        -------
        Sentinel1ObservationResult
        """
        try:
            if not self.geotiff_path.exists():
                logger.warning("GeoTIFF file not found: %s", self.geotiff_path)
                from flood_risk_zonation.satellite.result import create_unknown_sentinel1_result
                return create_unknown_sentinel1_result(bbox, "GeoTIFF file not found")

            with rasterio.open(self.geotiff_path) as src:
                # Extract flood mask for bbox
                window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
                flood_data = src.read(1, window=window)

                # Validate data
                if flood_data.size == 0:
                    logger.warning("GeoTIFF contains no data for bbox")
                    from flood_risk_zonation.satellite.result import create_unknown_sentinel1_result
                    return create_unknown_sentinel1_result(bbox, "GeoTIFF has no data for bbox")

                # Handle nodata
                nodata = src.nodata
                valid_mask = np.ones_like(flood_data, dtype=bool)
                if nodata is not None:
                    valid_mask = flood_data != nodata

                # Calculate statistics
                no_data_fraction = float(np.sum(~valid_mask) / flood_data.size) if flood_data.size > 0 else 1.0

                if np.sum(valid_mask) == 0:
                    logger.warning("All raster values are nodata")
                    from flood_risk_zonation.satellite.result import create_unknown_sentinel1_result
                    return create_unknown_sentinel1_result(bbox, "GeoTIFF all nodata for bbox")

                valid_data = flood_data[valid_mask]

                # Infer if binary (0/1) or continuous [0–1]
                is_binary = np.all(np.isin(valid_data, [0, 1]))
                if is_binary:
                    flood_observed = float(np.mean(valid_data)) > 0.5
                    inundation_fraction = float(np.mean(valid_data))
                else:
                    # Assume continuous [0–1]
                    inundation_fraction = float(np.mean(valid_data))
                    flood_observed = inundation_fraction > 0.5

                # Estimate area (simple: bbox area × inundation fraction)
                bbox_width_deg = bbox[2] - bbox[0]
                bbox_height_deg = bbox[3] - bbox[1]
                bbox_area_deg2 = bbox_width_deg * bbox_height_deg
                # At equator: 1 deg ≈ 111 km
                bbox_area_km2 = bbox_area_deg2 * (111.0 ** 2)
                flooded_area_km2 = bbox_area_km2 * inundation_fraction

                # Get CRS
                crs = str(src.crs) if src.crs else "EPSG:4326"

                # Get spatial resolution (nominal)
                spatial_resolution_m = float(abs(src.transform.a)) * 111000.0 if src.transform else 10.0

                return Sentinel1ObservationResult(
                    observation_status="OBSERVED",
                    flood_observed=flood_observed,
                    inundation_fraction=inundation_fraction,
                    flooded_area_km2=flooded_area_km2,
                    no_data_fraction=no_data_fraction,
                    confidence=0.85 if no_data_fraction < 0.2 else 0.70,  # Lower confidence if lots of nodata
                    coverage_fraction=float(np.sum(valid_mask) / flood_data.size),
                    source="sentinel1_geotiff",
                    provider="Local",
                    platform="Sentinel-1A/1B",
                    sensor="SAR",
                    acquisition_time=datetime.now(),  # Should be from TIFF metadata
                    processing_time=datetime.now(),
                    method="DERIVED_FLOOD_MASK",
                    spatial_resolution_m=spatial_resolution_m,
                    crs=crs,
                    bbox=bbox,
                    input_format="GeoTIFF",
                    limitations=[
                        "Derived flood mask; processing method not tracked in TIFF.",
                        "Acquisition time from file metadata not available.",
                        "Spatial resolution estimated from transform.",
                    ],
                )

        except Exception as exc:
            logger.exception("Error loading GeoTIFF: %s", exc)
            from flood_risk_zonation.satellite.result import create_unavailable_sentinel1_result
            return create_unavailable_sentinel1_result(
                bbox, f"GeoTIFF loading failed: {exc}"
            )


class VectorFloodPolygonProvider(Sentinel1Provider):
    """
    Load Sentinel-1-derived flood observations from GeoJSON flood polygons.

    Expected GeoJSON properties:
    - MultiPolygon or Polygon features
    - CRS metadata (must be EPSG:4326 or transformable)
    - Optional properties: flood_type, confidence, etc.
    """

    provider_type = "vector_geojson"

    def __init__(self, geojson_path: str | Path):
        """
        Initialize with path to GeoJSON flood polygons.

        Parameters
        ----------
        geojson_path : str | Path
            Path to GeoJSON file
        """
        self.geojson_path = Path(geojson_path)
        if not self.geojson_path.exists():
            logger.warning("GeoJSON file does not exist: %s", geojson_path)

    def load_observation(
        self,
        bbox: tuple[float, float, float, float],
        acquisition_date: str | None = None,
        max_days_old: int = 365,
    ) -> Sentinel1ObservationResult:
        """
        Load flood observation from GeoJSON polygons.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            (min_lon, min_lat, max_lon, max_lat)
        acquisition_date : str | None
            Ignored
        max_days_old : int
            Ignored

        Returns
        -------
        Sentinel1ObservationResult
        """
        try:
            if not self.geojson_path.exists():
                logger.warning("GeoJSON file not found: %s", self.geojson_path)
                from flood_risk_zonation.satellite.result import create_unknown_sentinel1_result
                return create_unknown_sentinel1_result(bbox, "GeoJSON file not found")

            gdf = gpd.read_file(self.geojson_path)

            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")

            # Reproject to WGS84 if needed
            if gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            # Create bbox geometry
            from shapely.geometry import box

            bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])

            # Spatial filter
            gdf_intersect = gdf[gdf.geometry.intersects(bbox_geom)]

            if len(gdf_intersect) == 0:
                logger.info("No flood polygons intersect bbox")
                from flood_risk_zonation.satellite.result import create_unknown_sentinel1_result
                return create_unknown_sentinel1_result(bbox, "No flood polygons in bbox")

            # Calculate intersection area
            flooded_area_km2 = 0.0
            for _, row in gdf_intersect.iterrows():
                intersection = row.geometry.intersection(bbox_geom)
                if intersection.is_valid and not intersection.is_empty:
                    # Convert to Web Mercator for area calculation
                    intersection_proj = gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326").to_crs(
                        "EPSG:3857"
                    )
                    flooded_area_km2 += intersection_proj.geometry.area.sum() / 1e6

            # Calculate bbox area
            bbox_gdf = gpd.GeoDataFrame([{"geometry": bbox_geom}], crs="EPSG:4326").to_crs("EPSG:3857")
            bbox_area_km2 = bbox_gdf.geometry.area.sum() / 1e6

            inundation_fraction = min(1.0, flooded_area_km2 / bbox_area_km2) if bbox_area_km2 > 0 else 0.0
            flood_observed = inundation_fraction > 0.05  # Threshold

            return Sentinel1ObservationResult(
                observation_status="OBSERVED",
                flood_observed=flood_observed,
                inundation_fraction=inundation_fraction,
                flooded_area_km2=flooded_area_km2,
                no_data_fraction=0.0,
                confidence=0.80,
                coverage_fraction=1.0,
                source="sentinel1_geojson",
                provider="Local",
                platform="Sentinel-1A/1B",
                sensor="SAR",
                acquisition_time=datetime.now(),
                processing_time=datetime.now(),
                method="VECTOR_POLYGONS",
                spatial_resolution_m=10.0,
                crs="EPSG:4326",
                bbox=bbox,
                input_format="GeoJSON",
                limitations=[
                    "Derived from flood polygons; original SAR method not tracked.",
                    "Acquisition time from file metadata not available.",
                ],
            )

        except Exception as exc:
            logger.exception("Error loading GeoJSON: %s", exc)
            from flood_risk_zonation.satellite.result import create_unavailable_sentinel1_result
            return create_unavailable_sentinel1_result(
                bbox, f"GeoJSON loading failed: {exc}"
            )
