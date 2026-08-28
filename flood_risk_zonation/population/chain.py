"""PopulationProviderChain — deterministic multi-tier fallback orchestrator."""
from __future__ import annotations

import logging
from typing import Optional

from flood_risk_zonation.population.enums import (
    PopulationDataStatus,
    PopulationProviderType,
)
from flood_risk_zonation.population.provider import PopulationProvider, UnknownProvider
from flood_risk_zonation.population.result import PopulationResult

logger = logging.getLogger(__name__)


class PopulationProviderChain:
    """
    Deterministic multi-tier population provider chain.

    Tries each provider in order: Authoritative → Regional → WorldPop → OSM → Derived → UNKNOWN.
    Uses first successful provider; records fallback reasons.

    Parameters
    ----------
    authoritative : Optional[PopulationProvider]
        Tier 1: Local census/government data
    regional : Optional[PopulationProvider]
        Tier 2: Regional public database
    worldpop : Optional[PopulationProvider]
        Tier 3: WorldPop gridded raster
    osm : Optional[PopulationProvider]
        Tier 4: OpenStreetMap tags
    derived : Optional[PopulationProvider]
        Tier 5: Derived estimates (building count, etc.)
    """

    def __init__(
        self,
        authoritative: Optional[PopulationProvider] = None,
        regional: Optional[PopulationProvider] = None,
        worldpop: Optional[PopulationProvider] = None,
        osm: Optional[PopulationProvider] = None,
        derived: Optional[PopulationProvider] = None,
    ):
        """Initialize providers for each tier."""
        self.providers = [
            ("authoritative", authoritative),
            ("regional", regional),
            ("worldpop", worldpop),
            ("osm", osm),
            ("derived", derived),
        ]
        # Always have UNKNOWN as fallback
        self.unknown_provider = UnknownProvider()

    def get_population(
        self,
        hab_id: str,
        lat: float,
        lon: float,
        bbox: tuple,
    ) -> PopulationResult:
        """
        Get population via provider chain.

        Tries each provider in order. Returns first OBSERVED/ESTIMATED result.
        If all UNAVAILABLE, returns UNKNOWN.

        Parameters
        ----------
        hab_id : str
            Habitation ID
        lat, lon : float
            Habitation location (WGS84)
        bbox : tuple
            (min_lon, min_lat, max_lon, max_lat)

        Returns
        -------
        PopulationResult
        """
        previous_fallback_reason = None

        for tier_name, provider in self.providers:
            if provider is None:
                logger.debug(f"Provider chain: {tier_name} not configured, skipping")
                continue

            try:
                result = provider.get_population(
                    hab_id=hab_id,
                    lat=lat,
                    lon=lon,
                    bbox_min_lon=bbox[0],
                    bbox_min_lat=bbox[1],
                    bbox_max_lon=bbox[2],
                    bbox_max_lat=bbox[3],
                )

                # If data found (OBSERVED or ESTIMATED), return it
                if result.status in (PopulationDataStatus.OBSERVED, PopulationDataStatus.ESTIMATED):
                    logger.debug(f"Provider chain: {tier_name} succeeded for {hab_id}")
                    if previous_fallback_reason:
                        result.fallback_reason = previous_fallback_reason
                    return result

                # If UNAVAILABLE, try next provider
                if result.status == PopulationDataStatus.UNAVAILABLE:
                    previous_fallback_reason = f"{tier_name}_unavailable"
                    logger.debug(f"Provider chain: {tier_name} unavailable for {hab_id}, trying next")
                    continue

                # If UNKNOWN, try next provider
                if result.status == PopulationDataStatus.UNKNOWN:
                    previous_fallback_reason = f"{tier_name}_unknown"
                    logger.debug(f"Provider chain: {tier_name} returned UNKNOWN for {hab_id}, trying next")
                    continue

            except Exception as e:
                previous_fallback_reason = f"{tier_name}_exception"
                logger.warning(f"Provider chain: {tier_name} failed for {hab_id}: {e}. Trying next tier.")
                continue

        # All providers exhausted or skipped; return UNKNOWN
        result = self.unknown_provider.get_population(hab_id, lat, lon, *bbox)
        if previous_fallback_reason:
            result.fallback_reason = previous_fallback_reason
        logger.debug(f"Provider chain: all providers exhausted for {hab_id}, returning UNKNOWN")
        return result

    def get_population_batch(
        self,
        habitations: list[dict],
        bbox: tuple,
    ) -> list[PopulationResult]:
        """
        Get population for multiple habitations.

        Parameters
        ----------
        habitations : list[dict]
            Each dict: {"hab_id": str, "lat": float, "lon": float}
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
                bbox=bbox,
            )
            results.append(result)
        return results
