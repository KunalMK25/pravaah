"""DerivedProvider — Tier 5 — derived population estimates."""
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


class DerivedProvider(PopulationProvider):
    """
    Tier 5: Derived provider — derived population estimates.

    Currently returns UNAVAILABLE (placeholder for future building-count estimation).
    When implemented, estimates population from building footprints or landuse area.

    Parameters
    ----------
    config : Optional[dict]
        Configuration dict (currently unused)
    """

    provider_type = PopulationProviderType.DERIVED

    def __init__(self, config: Optional[dict] = None):
        """Initialize derived estimate provider."""
        self.config = config or {}

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
        """Get derived population estimate."""
        # Derived estimates not yet implemented
        # When implemented, this would:
        # 1. Count OSM buildings within search radius
        # 2. Multiply by average occupancy (e.g., 5 persons/building)
        # 3. Return ESTIMATED status with low confidence

        return PopulationResult(
            population=None,
            source="derived",
            provider=self.provider_type,
            method=None,
            status=PopulationDataStatus.UNAVAILABLE,
            confidence=0.0,
            spatial_resolution_m=float("nan"),
            temporal_resolution="unknown",
            collection_year=None,
            retrieved_at=datetime.now(),
            coverage_percent=0.0,
            limitations=["Derived population estimation not yet implemented"],
            fallback_reason="derived_not_implemented",
        )
