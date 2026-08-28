"""PopulationProvider — abstract base class and terminal UNKNOWN provider."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from flood_risk_zonation.population.enums import (
    PopulationDataStatus,
    PopulationMethod,
    PopulationProviderType,
)
from flood_risk_zonation.population.result import PopulationResult

logger = logging.getLogger(__name__)


class PopulationProvider(ABC):
    """
    Abstract base class for population data providers.

    Each provider implements a single tier of the fallback chain.
    Providers are deterministic — same input always produces same output.

    Providers NEVER fabricate data. If unable to provide, return UNKNOWN or UNAVAILABLE.
    """

    provider_type: PopulationProviderType

    @abstractmethod
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
        """
        Get population for a single habitation.

        Parameters
        ----------
        hab_id : str
            Habitation ID for tracking/caching
        lat, lon : float
            Habitation location (WGS84)
        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat : float
            Analysis bounding box

        Returns
        -------
        PopulationResult
            Always returns a result. Never raises exception.
            If provider unavailable/fails, returns UNAVAILABLE or UNKNOWN result.
        """
        pass

    def get_population_batch(
        self,
        habitations: list[dict],
        bbox: tuple,
    ) -> list[PopulationResult]:
        """
        Get population for multiple habitations.

        Default: calls get_population per item. Override for bulk efficiency.

        Parameters
        ----------
        habitations : list[dict]
            Each dict: {"hab_id", "lat", "lon"}
        bbox : tuple
            (min_lon, min_lat, max_lon, max_lat)

        Returns
        -------
        list[PopulationResult]
        """
        results = []
        for hab in habitations:
            result = self.get_population(
                hab_id=hab.get("hab_id"),
                lat=hab.get("lat"),
                lon=hab.get("lon"),
                bbox_min_lon=bbox[0],
                bbox_min_lat=bbox[1],
                bbox_max_lon=bbox[2],
                bbox_max_lat=bbox[3],
            )
            results.append(result)
        return results


class UnknownProvider(PopulationProvider):
    """
    Terminal provider — returns UNKNOWN when no other source available.

    Always returns same result: population=None, status=UNKNOWN, confidence=0.0.
    """

    provider_type = PopulationProviderType.UNKNOWN

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
        """Always return UNKNOWN."""
        import numpy as np

        return PopulationResult(
            population=None,
            source="unknown",
            provider=self.provider_type,
            method=None,
            status=PopulationDataStatus.UNKNOWN,
            confidence=0.0,
            spatial_resolution_m=float("nan"),
            temporal_resolution="unknown",
            collection_year=None,
            retrieved_at=datetime.now(),
            coverage_percent=0.0,
            limitations=["No population data source available for this location"],
        )
