"""
PRAVAAH-AI — Emergency Response Orchestrator

Orchestrates emergency facility loading and evacuation route computation.

This module is the main entry point for emergency response features:
1. Load emergency facilities (hospitals, shelters) from OSM
2. Compute hazard-aware evacuation routes for vulnerable habitations
3. Return structured EvacuationRoute objects for visualization and decision support

All operations are DECISION-SUPPORT ONLY. Authority decision-makers must verify
facility capacity and issue official evacuation orders.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx

from flood_risk_zonation.capacity.assessment import _load_healthcare, _load_shelters
from flood_risk_zonation.capacity.emergency import EmergencyFacility, EvacuationRoute
from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.models import SIHAnalysisResult
from flood_risk_zonation.spatial_zones.classifier import (
    ZONE_RED, ZONE_YELLOW, ZONE_GREEN, ZONE_WATER
)
from flood_risk_zonation.utils.evacuation_routing import (
    add_hazard_weights_to_graph, find_safest_facility
)
from flood_risk_zonation.utils.routing import build_road_graph

logger = logging.getLogger(__name__)


def load_emergency_facilities(
    bbox: BoundingBox,
    cache_dir: str | Path,
    allow_network: bool = True,
) -> dict[str, list[EmergencyFacility]]:
    """
    Load emergency facilities (hospitals, shelters) from OSM.

    Queries OpenStreetMap Overpass API for:
    - Hospitals, clinics, health centres
    - Shelters, community centres

    Facilities are cached locally to avoid repeated queries.

    Parameters
    ----------
    bbox : BoundingBox
        Bounding box for the study area.
    cache_dir : str | Path
        Directory for OSM query cache.
    allow_network : bool
        If True, allow network queries. If False, use cache only.

    Returns
    -------
    dict[str, list[EmergencyFacility]]
        Dictionary with keys "hospitals" and "shelters", each containing
        a list of EmergencyFacility objects.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    facilities_dict = {
        "hospitals": [],
        "shelters": [],
    }

    try:
        # Load hospitals
        hospital_points = _load_healthcare(bbox, cache_dir, allow_network)
        for i, (lat, lon) in enumerate(hospital_points):
            fac = EmergencyFacility(
                facility_id=f"hospital_osm_{i}",
                name=f"Hospital {i+1}",
                facility_type="hospital",
                latitude=lat,
                longitude=lon,
                source="osm_overpass",
                osm_id=None,
                metadata={},
            )
            facilities_dict["hospitals"].append(fac)

        logger.info("Loaded %d hospital/clinic facilities", len(facilities_dict["hospitals"]))
    except Exception as e:
        logger.warning("Failed to load hospitals: %s", e)

    try:
        # Load shelters
        shelter_points = _load_shelters(bbox, cache_dir, allow_network)
        for i, (lat, lon) in enumerate(shelter_points):
            fac = EmergencyFacility(
                facility_id=f"shelter_osm_{i}",
                name=f"Shelter {i+1}",
                facility_type="shelter",
                latitude=lat,
                longitude=lon,
                source="osm_overpass",
                osm_id=None,
                metadata={},
            )
            facilities_dict["shelters"].append(fac)

        logger.info("Loaded %d shelter facilities", len(facilities_dict["shelters"]))
    except Exception as e:
        logger.warning("Failed to load shelters: %s", e)

    return facilities_dict


def compute_evacuation_routes(
    sih_result: SIHAnalysisResult,
    zoned_grid: gpd.GeoDataFrame,
    facilities_dict: dict[str, list[EmergencyFacility]],
    road_graph: Optional[nx.MultiGraph] = None,
    priority_filter: Optional[list[str]] = None,
) -> list[EvacuationRoute]:
    """
    Compute hazard-aware evacuation routes for vulnerable habitations.

    For each habitation in the priority filter (default: CRITICAL, HIGH):
    1. Gather all candidate facilities (hospitals + shelters)
    2. Use find_safest_facility() to compute the lowest-RED-exposure route
    3. Return structured EvacuationRoute objects

    Parameters
    ----------
    sih_result : SIHAnalysisResult
        Output from the SIH pipeline with habitations and relocation priorities.
    zoned_grid : gpd.GeoDataFrame
        Grid with spatial_zone column (RED/YELLOW/GREEN/WATER).
    facilities_dict : dict[str, list[EmergencyFacility]]
        Dictionary with "hospitals" and "shelters" keys.
    road_graph : nx.MultiGraph | None
        Road network for routing. If None, straight-line fallback is used.
    priority_filter : list[str] | None
        Only compute routes for habitations with these priority classes.
        Default: ["CRITICAL", "HIGH"].

    Returns
    -------
    list[EvacuationRoute]
        List of EvacuationRoute objects (status may be FOUND, NO_SAFE_ROUTE, etc.).
    """
    if priority_filter is None:
        priority_filter = ["CRITICAL", "HIGH"]

    if not sih_result.relocation_results:
        logger.warning("No relocation results; no evacuation routes to compute")
        return []

    if zoned_grid is None or len(zoned_grid) == 0 or "spatial_zone" not in zoned_grid.columns:
        logger.warning("No zoned grid; evacuation routes will use fallback hazard exposure")

    # Build hazard-weighted graph if road graph is available
    hazard_graph = None
    if road_graph is not None and zoned_grid is not None:
        try:
            hazard_graph = add_hazard_weights_to_graph(road_graph, zoned_grid)
            logger.debug("Built hazard-weighted graph for evacuation routing")
        except Exception as e:
            logger.warning("Failed to build hazard-weighted graph: %s", e)
            hazard_graph = road_graph

    # Combine all facilities
    all_facilities = facilities_dict.get("hospitals", []) + facilities_dict.get("shelters", [])
    if not all_facilities:
        logger.warning("No facilities available; returning empty routes")
        return []

    # Convert facilities to dict format for routing engine
    facility_dicts = [
        {
            "facility_id": f.facility_id,
            "name": f.name,
            "facility_type": f.facility_type,
            "latitude": f.latitude,
            "longitude": f.longitude,
        }
        for f in all_facilities
    ]

    # Compute routes for priority habitations
    evacuation_routes = []

    for rel_result in sih_result.relocation_results:
        if rel_result.priority_class not in priority_filter:
            continue

        # Get corresponding exposure result for coordinates
        exp_result = sih_result.get_exposure_by_id(rel_result.hab_id)
        if exp_result is None:
            logger.debug("No exposure result for habitation %s", rel_result.hab_id)
            continue

        # Find safest facility
        route = find_safest_facility(
            origin_hab_id=rel_result.hab_id,
            origin_lat=exp_result.lat,
            origin_lon=exp_result.lon,
            candidate_facilities=facility_dicts,
            hazard_graph=hazard_graph,
            standard_graph=road_graph,
            zoned_grid=zoned_grid,
        )

        # Populate habitation name
        route.hab_name = rel_result.hab_name or exp_result.name or "Unnamed"

        evacuation_routes.append(route)

        if route.status == "FOUND":
            logger.info(
                "Evacuation route: %s → %s (%.2f km, RED=%.1f%%)",
                route.hab_name, route.facility_name, route.distance_km,
                route.hazard_exposure.get(ZONE_RED, 0.0)
            )
        else:
            logger.warning("Evacuation route failed for %s: %s", route.hab_name, route.status)

    logger.info("Computed %d evacuation routes (%d successful)",
                len(evacuation_routes),
                sum(1 for r in evacuation_routes if r.status == "FOUND"))

    return evacuation_routes
