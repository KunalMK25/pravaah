"""AuthoritativeProvider — Tier 1 — local census / government data."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from flood_risk_zonation.population.confidence import compute_confidence
from flood_risk_zonation.population.enums import (
    PopulationDataStatus,
    PopulationMethod,
    PopulationProviderType,
)
from flood_risk_zonation.population.provider import PopulationProvider
from flood_risk_zonation.population.result import PopulationResult

logger = logging.getLogger(__name__)


class AuthoritativeProvider(PopulationProvider):
    """
    Tier 1: Authoritative population provider (local census / government data).

    When configured and available, provides the highest-confidence population data.
    If not configured, returns UNAVAILABLE for fallback to next tier.

    Parameters
    ----------
    config : Optional[dict]
        Configuration dict with keys:
        - enabled: bool (default False)
        - data_path: str (path to census data file/database)
        - data_format: str (e.g., "shapefile", "geojson", "csv")
        - year: int (census year)
    """

    provider_type = PopulationProviderType.AUTHORITATIVE

    def __init__(self, config: Optional[dict] = None):
        """Initialize with authoritative data configuration."""
        self.config = config or {}
        self.data = None
        self._loaded = False

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
        """Get population from authoritative source."""
        # Check if provider is enabled
        if not self.config.get("enabled", False):
            return PopulationResult(
                population=None,
                source="authoritative",
                provider=self.provider_type,
                method=None,
                status=PopulationDataStatus.UNAVAILABLE,
                confidence=0.0,
                spatial_resolution_m=float("nan"),
                temporal_resolution="unknown",
                collection_year=None,
                retrieved_at=datetime.now(),
                coverage_percent=0.0,
                limitations=["Authoritative provider not configured"],
                fallback_reason="authoritative_not_configured",
            )

        # In this implementation, authoritative data is not bundled.
        # Users must configure their own census data.
        # For now, we return UNAVAILABLE as placeholder.
        logger.debug(
            "Authoritative provider configured but no data loaded (placeholder implementation)"
        )
        return PopulationResult(
            population=None,
            source="authoritative",
            provider=self.provider_type,
            method=None,
            status=PopulationDataStatus.UNAVAILABLE,
            confidence=0.0,
            spatial_resolution_m=float("nan"),
            temporal_resolution="point_in_time",
            collection_year=self.config.get("year"),
            retrieved_at=datetime.now(),
            coverage_percent=0.0,
            limitations=["Authoritative data not available at this location"],
            fallback_reason="authoritative_no_coverage",
        )
