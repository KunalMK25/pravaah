"""Sentinel-1 flood observation provider abstraction."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from flood_risk_zonation.satellite.result import (
    Sentinel1ObservationResult,
    create_unknown_sentinel1_result,
    create_unavailable_sentinel1_result,
)

logger = logging.getLogger(__name__)


class Sentinel1Provider(ABC):
    """
    Abstract base class for Sentinel-1 flood observation providers.

    Each provider ingests Sentinel-1-derived flood observations from a specific source.
    Providers are deterministic — same input always produces same output.

    Providers NEVER fabricate satellite observations.
    If unable to provide, return UNKNOWN or UNAVAILABLE.

    Expected implementations:
    - RasterFloodMaskProvider (GeoTIFF)
    - VectorFloodPolygonProvider (GeoJSON)
    - CachedObservationProvider (cached previous results)
    - UnknownSentinel1Provider (terminal fallback)
    """

    provider_type: str

    @abstractmethod
    def load_observation(
        self,
        bbox: tuple[float, float, float, float],
        acquisition_date: str | None = None,
        max_days_old: int = 365,
    ) -> Sentinel1ObservationResult:
        """
        Load Sentinel-1 flood observation for a bounding box.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            (min_lon, min_lat, max_lon, max_lat) analysis area
        acquisition_date : str | None
            Target acquisition date (ISO-8601, e.g., "2024-08-26")
            If None, use most recent available observation.
        max_days_old : int
            Reject observations older than this many days.
            Default: 365 (1 year)

        Returns
        -------
        Sentinel1ObservationResult
            Always returns a result. Never raises exception.
            If provider unavailable/fails, returns UNAVAILABLE or UNKNOWN result.
        """
        pass


class UnknownSentinel1Provider(Sentinel1Provider):
    """
    Terminal provider — returns UNKNOWN when no other source available.

    Always returns same result: flood_observed=None, status=UNKNOWN, confidence=0.0.
    """

    provider_type = "unknown"

    def load_observation(
        self,
        bbox: tuple[float, float, float, float],
        acquisition_date: str | None = None,
        max_days_old: int = 365,
    ) -> Sentinel1ObservationResult:
        """Always return UNKNOWN."""
        logger.debug("UnknownSentinel1Provider: no data available")
        return create_unknown_sentinel1_result(bbox, reason="No Sentinel-1 provider available")
