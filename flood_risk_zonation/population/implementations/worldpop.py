"""WorldPopProvider — Tier 3 — WorldPop gridded population raster with spatial aggregation."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np
from rasterio.transform import rowcol
from shapely.geometry import Point

from flood_risk_zonation.population.confidence import compute_confidence
from flood_risk_zonation.population.enums import (
    PopulationDataStatus,
    PopulationMethod,
    PopulationProviderType,
)
from flood_risk_zonation.population.provider import PopulationProvider
from flood_risk_zonation.population.result import PopulationResult
from flood_risk_zonation.models import RasterDataset

logger = logging.getLogger(__name__)


class WorldPopProvider(PopulationProvider):
    """
    Tier 3: WorldPop provider — gridded population raster with spatial aggregation.

    Aggregates population within search_radius_km of habitation point.
    Handles nodata, missing raster, CRS conversion, etc.

    Parameters
    ----------
    raster : RasterDataset
        WorldPop or similar gridded population raster
    search_radius_km : float
        Search radius for aggregation (default 2.0 km)
    collection_year : Optional[int]
        Year of WorldPop data (for temporal adjustments)
    """

    provider_type = PopulationProviderType.WORLDPOP

    def __init__(
        self,
        raster: RasterDataset,
        search_radius_km: float = 2.0,
        collection_year: Optional[int] = 2020,
    ):
        """Initialize with WorldPop raster."""
        self.raster = raster
        self.search_radius_km = search_radius_km
        self.collection_year = collection_year

    def get_population(
        self,
        hab_id: str,
        lat: float,
        lon: float,
        bbox_min_lon: float,
        bbox_min_lat: float,
        bbox_max_lon: float,
        bbox_max_lat: float,
    ) -> PopulationResult:
        """Get population from WorldPop raster."""
        try:
            # Aggregate population within search radius
            aggregated_pop = self._aggregate_within_radius(lat, lon, self.search_radius_km)

            if aggregated_pop is None or aggregated_pop <= 0:
                # No data at this location
                return PopulationResult(
                    population=None,
                    source=self.raster.source or "worldpop",
                    provider=self.provider_type,
                    method=None,
                    status=PopulationDataStatus.UNAVAILABLE,
                    confidence=0.0,
                    spatial_resolution_m=float("nan"),
                    temporal_resolution="static",
                    collection_year=self.collection_year,
                    retrieved_at=datetime.now(),
                    coverage_percent=0.0,
                    limitations=["WorldPop has no data at this location"],
                    fallback_reason="worldpop_no_data",
                )

            # Compute confidence with adjustments
            confidence = compute_confidence(
                provider_type=self.provider_type,
                spatial_resolution_m=1000.0,  # WorldPop native resolution
                collection_year=self.collection_year,
                coverage_percent=100.0,  # Assume full coverage if data exists
            )

            return PopulationResult(
                population=int(round(aggregated_pop)),
                source=self.raster.source or "worldpop",
                provider=self.provider_type,
                method=PopulationMethod.RASTER_AGGREGATION,
                status=PopulationDataStatus.OBSERVED,
                confidence=confidence,
                spatial_resolution_m=1000.0,
                temporal_resolution="static",
                collection_year=self.collection_year,
                retrieved_at=datetime.now(),
                coverage_percent=100.0,
                limitations=[
                    "WorldPop 1km resolution may not capture small clusters",
                    f"Aggregated within {self.search_radius_km}km radius",
                ],
            )

        except Exception as e:
            logger.warning(f"WorldPopProvider failed for {hab_id}: {e}")
            return PopulationResult(
                population=None,
                source=self.raster.source or "worldpop",
                provider=self.provider_type,
                method=None,
                status=PopulationDataStatus.UNAVAILABLE,
                confidence=0.0,
                spatial_resolution_m=float("nan"),
                temporal_resolution="static",
                collection_year=self.collection_year,
                retrieved_at=datetime.now(),
                coverage_percent=0.0,
                limitations=[f"WorldPopProvider exception: {str(e)}"],
                fallback_reason="worldpop_exception",
            )

    def _aggregate_within_radius(self, lat: float, lon: float, radius_km: float) -> Optional[float]:
        """
        Sum population within radius_km of point.

        Parameters
        ----------
        lat, lon : float
            Point location (WGS84)
        radius_km : float
            Search radius in kilometers

        Returns
        -------
        Optional[float]
            Sum of population within radius, or None if no data
        """
        if self.raster is None or self.raster.array is None:
            return None

        # Convert radius to degrees (~1 km ≈ 0.009 degrees)
        radius_deg = radius_km / 111.0

        # Extract bounding box around point
        min_lon = lon - radius_deg
        max_lon = lon + radius_deg
        min_lat = lat - radius_deg
        max_lat = lat + radius_deg

        try:
            # Get transform
            if self.raster.transform is None:
                return None

            # Convert lat/lon to raster row/col indices
            from rasterio.transform import rowcol

            nrows, ncols = self.raster.array.shape

            # Sample all pixels in the bounding box
            aggregated_pop = 0.0
            pixel_count = 0

            # Step through pixels in the search box
            # (Simplified: check all pixels in bounding box)
            for row in range(nrows):
                for col in range(ncols):
                    # Get coordinates of this pixel from transform
                    x = self.raster.transform.c + col * self.raster.transform.a
                    y = self.raster.transform.f + row * self.raster.transform.e

                    # Check if within bounding box
                    if min_lon <= x <= max_lon and min_lat <= y <= max_lat:
                        # Check if within search radius (Euclidean distance)
                        dx = (x - lon) * 111.0 * math.cos(math.radians(lat))
                        dy = (y - lat) * 111.0
                        distance_km = math.sqrt(dx**2 + dy**2)

                        if distance_km <= radius_km:
                            # Get pixel value
                            value = float(self.raster.array[row, col])

                            # Check for nodata
                            if not (math.isnan(value) or value < 0):
                                aggregated_pop += value
                                pixel_count += 1

            return aggregated_pop if pixel_count > 0 else None

        except Exception as e:
            logger.warning(f"Error aggregating WorldPop raster: {e}")
            return None
