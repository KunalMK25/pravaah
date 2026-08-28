"""OSMProvider — Tier 4 — OpenStreetMap population tags."""
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


class OSMProvider(PopulationProvider):
    """
    Tier 4: OSM provider — OpenStreetMap population tags.

    Looks up population from OSM habitation data ingested earlier.

    Parameters
    ----------
    habitations : dict
        Map of hab_id → Habitation dataclass with population field
    """

    provider_type = PopulationProviderType.OSM

    def __init__(self, habitations: Optional[dict] = None):
        """Initialize with OSM habitation data."""
        self.habitations = habitations or {}

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
        """Get population from OSM tag."""
        try:
            # Lookup habitation in OSM data
            hab = self.habitations.get(hab_id)

            if hab is None or hab.population is None:
                return PopulationResult(
                    population=None,
                    source="osm_tag",
                    provider=self.provider_type,
                    method=None,
                    status=PopulationDataStatus.UNAVAILABLE,
                    confidence=0.0,
                    spatial_resolution_m=0.0,  # Point-level
                    temporal_resolution="point_in_time",
                    collection_year=None,  # OSM doesn't track vintage
                    retrieved_at=datetime.now(),
                    coverage_percent=0.0,
                    limitations=["OSM population tag not available"],
                    fallback_reason="osm_population_missing",
                )

            # Compute confidence (OSM has inherent uncertainty)
            confidence = compute_confidence(
                provider_type=self.provider_type,
                spatial_resolution_m=0.0,  # Point-level = no penalty
                collection_year=None,  # Unknown age = no temporal penalty
                coverage_percent=100.0,  # Tag exists = 100% coverage for this point
            )

            return PopulationResult(
                population=hab.population,
                source="osm_tag",
                provider=self.provider_type,
                method=PopulationMethod.OSM_TAG_DIRECT,
                status=PopulationDataStatus.OBSERVED,
                confidence=confidence,
                spatial_resolution_m=0.0,  # Point-level
                temporal_resolution="point_in_time",
                collection_year=None,
                retrieved_at=datetime.now(),
                coverage_percent=100.0,
                limitations=[
                    "OSM population tags are volunteer-edited and may be outdated",
                    "Coverage is sparse, especially in rural areas",
                ],
            )

        except Exception as e:
            logger.warning(f"OSMProvider failed for {hab_id}: {e}")
            return PopulationResult(
                population=None,
                source="osm_tag",
                provider=self.provider_type,
                method=None,
                status=PopulationDataStatus.UNAVAILABLE,
                confidence=0.0,
                spatial_resolution_m=float("nan"),
                temporal_resolution="unknown",
                collection_year=None,
                retrieved_at=datetime.now(),
                coverage_percent=0.0,
                limitations=[f"OSMProvider exception: {str(e)}"],
                fallback_reason="osm_exception",
            )
