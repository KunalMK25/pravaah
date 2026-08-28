"""Factory function to build population provider chain from configuration."""
from __future__ import annotations

import logging
from typing import Optional

from flood_risk_zonation.models import RasterDataset
from flood_risk_zonation.population.chain import PopulationProviderChain
from flood_risk_zonation.population.implementations import (
    AuthoritativeProvider,
    DerivedProvider,
    OSMProvider,
    RegionalProvider,
    WorldPopProvider,
)

logger = logging.getLogger(__name__)


def create_population_provider_chain(
    config: Optional[dict] = None,
    worldpop_raster: Optional[RasterDataset] = None,
    habitations_dict: Optional[dict] = None,
) -> PopulationProviderChain:
    """
    Factory function to build provider chain from configuration.

    Parameters
    ----------
    config : Optional[dict]
        Configuration dict with keys:
        - "authoritative": dict (enable, data_path, year, etc.)
        - "regional": dict (enable, data_source, etc.)
        - "worldpop": dict (enable, search_radius_km, etc.)
        - "osm": dict (enable)
        - "derived": dict (enable)
    worldpop_raster : Optional[RasterDataset]
        WorldPop raster data (required if worldpop enabled)
    habitations_dict : Optional[dict]
        Map of hab_id → Habitation (required if osm enabled)

    Returns
    -------
    PopulationProviderChain
    """
    config = config or {}

    authoritative = None
    regional = None
    worldpop = None
    osm = None
    derived = None

    # Tier 1: Authoritative
    if config.get("authoritative", {}).get("enabled", False):
        authoritative = AuthoritativeProvider(config["authoritative"])
        logger.info("Population provider chain: authoritative provider enabled")

    # Tier 2: Regional
    if config.get("regional", {}).get("enabled", False):
        regional = RegionalProvider(config["regional"])
        logger.info("Population provider chain: regional provider enabled")

    # Tier 3: WorldPop
    if config.get("worldpop", {}).get("enabled", False):
        if worldpop_raster is None:
            logger.warning(
                "Population provider chain: worldpop enabled but no raster provided, skipping"
            )
        else:
            search_radius_km = config["worldpop"].get("search_radius_km", 2.0)
            collection_year = config["worldpop"].get("collection_year", 2020)
            worldpop = WorldPopProvider(
                raster=worldpop_raster,
                search_radius_km=search_radius_km,
                collection_year=collection_year,
            )
            logger.info("Population provider chain: worldpop provider enabled")

    # Tier 4: OSM
    if config.get("osm", {}).get("enabled", False):
        if habitations_dict is None:
            logger.warning(
                "Population provider chain: osm enabled but no habitations provided, skipping"
            )
        else:
            osm = OSMProvider(habitations=habitations_dict)
            logger.info("Population provider chain: osm provider enabled")

    # Tier 5: Derived
    if config.get("derived", {}).get("enabled", False):
        derived = DerivedProvider(config["derived"])
        logger.info("Population provider chain: derived provider enabled")

    # Build chain
    chain = PopulationProviderChain(
        authoritative=authoritative,
        regional=regional,
        worldpop=worldpop,
        osm=osm,
        derived=derived,
    )

    logger.info("Population provider chain created successfully")
    return chain
