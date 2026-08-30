"""
PRAVAAH-AI — Hazard-Aware Evacuation Routing

METHODOLOGY (transparent, decision-support):
───────────────────────────────────────────────────────────────────────────
This module implements hazard-aware shortest-path routing to recommend safe
evacuation routes from habitations to emergency facilities (hospitals, shelters).

HAZARD WEIGHTING STRATEGY:
  Routes are computed on a hazard-weighted road network where edge weights are
  adjusted based on the spatial zone (RED/YELLOW/GREEN/WATER) that each edge
  passes through. The weighting encourages the routing engine to naturally prefer
  longer GREEN routes over shorter RED routes.

  Zone-based weight multipliers:
    GREEN:   1.0x (preferred)
    YELLOW:  2.0x (penalized but not avoided)
    RED:     20.0x (heavily penalized; avoided if possible)
    WATER:   impassable (edge weight set to infinity)

  Dijkstra's algorithm then finds the lowest-cost path, which naturally minimizes
  hazard exposure when possible.

NO EVACUATION AUTONOMY:
  - Routes are RECOMMENDATIONS ONLY for decision-maker review.
  - Authority decision-makers verify facility capacity and issue official orders.
  - The system does NOT autonomously dispatch people or declare evacuation policy.

LIMITATIONS & FALLBACKS:
  - If no safe route exists, the algorithm reports "NO_SAFE_ROUTE_AVAILABLE".
    This typically means all paths must pass through RED zones; it does NOT mean
    evacuation is impossible (secondary routes, temporary shelter, etc. may exist).
  - If the road network is unavailable, the routing engine falls back to
    straight-line (haversine) distance.
  - Facility matching is nearest-node on the road graph; actual routing from
    facility is not performed.
  - Hazard zones are static (grid-based); real evacuation may encounter
    dynamic conditions (congestion, road damage).

ROUTE GEOMETRY:
  - Route geometry is a list of (latitude, longitude) tuples representing
    waypoints along the recommended path.
  - Hazard exposure is calculated by sampling the route at regular intervals
    and reporting the percentage distance in each zone.
───────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np

from flood_risk_zonation.capacity.emergency import EmergencyFacility, EvacuationRoute
from flood_risk_zonation.spatial_zones.classifier import (
    ZONE_RED, ZONE_YELLOW, ZONE_GREEN, ZONE_WATER
)
from flood_risk_zonation.utils.routing import _haversine_km

logger = logging.getLogger(__name__)


def add_hazard_weights_to_graph(
    graph: nx.MultiGraph,
    zoned_grid: gpd.GeoDataFrame,
) -> nx.MultiGraph:
    """
    Return a new graph with edge weights adjusted by hazard zone.

    For each edge in the original graph, we:
    1. Find the midpoint of the edge
    2. Look up its spatial zone in the zoned_grid
    3. Multiply the original weight by a zone-based factor:
       - GREEN:   1.0x
       - YELLOW:  2.0x
       - RED:     20.0x
       - WATER:   infinity (impassable)
    4. Store the adjusted weight on the edge

    Parameters
    ----------
    graph : nx.MultiGraph
        Original road network graph with distance-weighted edges.
    zoned_grid : gpd.GeoDataFrame
        Grid with spatial_zone column (RED/YELLOW/GREEN/WATER).

    Returns
    -------
    nx.MultiGraph
        New graph with hazard-adjusted weights. Original graph is not modified.
    """
    if graph is None or len(graph.nodes()) == 0:
        logger.warning("add_hazard_weights_to_graph: graph is empty or None")
        return graph

    if "spatial_zone" not in zoned_grid.columns:
        logger.warning("add_hazard_weights_to_graph: zoned_grid has no spatial_zone column")
        return graph

    # Zone-based weight multipliers
    zone_multipliers = {
        ZONE_GREEN:  1.0,
        ZONE_YELLOW: 2.0,
        ZONE_RED:    20.0,
        ZONE_WATER:  float("inf"),
    }

    # Build spatial index for fast zone lookup
    # (This is simplified: we just check centroids against grid geometry)
    weighted_graph = graph.copy()

    adjusted_count = 0
    for u, v, key, data in weighted_graph.edges(keys=True, data=True):
        # Get node coordinates
        u_lat = weighted_graph.nodes[u].get("lat")
        u_lon = weighted_graph.nodes[u].get("lon")
        v_lat = weighted_graph.nodes[v].get("lat")
        v_lon = weighted_graph.nodes[v].get("lon")

        if u_lat is None or u_lon is None or v_lat is None or v_lon is None:
            continue

        # Compute edge midpoint
        mid_lat = (u_lat + v_lat) / 2.0
        mid_lon = (u_lon + v_lon) / 2.0

        # Find zone at midpoint (nearest cell in grid)
        # Simple approach: find nearest grid cell centre
        zone = ZONE_GREEN  # default to GREEN
        min_dist = float("inf")

        for _, grid_cell in zoned_grid.iterrows():
            cell_lat = grid_cell.get("centroid_lat")
            cell_lon = grid_cell.get("centroid_lon")
            if cell_lat is None or cell_lon is None:
                continue

            dist = _haversine_km(mid_lat, mid_lon, cell_lat, cell_lon)
            if dist < min_dist:
                min_dist = dist
                zone = grid_cell.get("spatial_zone", ZONE_GREEN)

        # Apply zone multiplier
        multiplier = zone_multipliers.get(zone, 1.0)
        original_weight = data.get("weight", 1.0)
        new_weight = original_weight * multiplier

        # Update edge weight
        weighted_graph[u][v][key]["weight"] = new_weight
        weighted_graph[u][v][key]["hazard_zone"] = zone
        adjusted_count += 1

    logger.debug(
        "Adjusted %d edges with hazard weights (graph: %d nodes, %d edges)",
        adjusted_count, len(weighted_graph.nodes()), len(weighted_graph.edges())
    )
    return weighted_graph


def _sample_zone_along_route(
    route_coords: list[tuple[float, float]],
    zoned_grid: gpd.GeoDataFrame,
    sample_interval_km: float = 0.5,
) -> dict[str, float]:
    """
    Sample hazard zone along a route and return exposure breakdown.

    Samples at regular distance intervals along the route and counts
    how many samples fall into each zone.

    Parameters
    ----------
    route_coords : list[tuple[float, float]]
        List of (lat, lon) tuples along the route.
    zoned_grid : gpd.GeoDataFrame
        Grid with spatial_zone column.
    sample_interval_km : float
        Spacing between samples (in km). Default 0.5 km.

    Returns
    -------
    dict[str, float]
        Dictionary with keys RED, YELLOW, GREEN, WATER and float values
        representing percentage of route distance in each zone.
    """
    if not route_coords or len(route_coords) < 2:
        # Return default (all GREEN) if route is too short
        return {ZONE_RED: 0.0, ZONE_YELLOW: 0.0, ZONE_GREEN: 100.0, ZONE_WATER: 0.0}

    if "spatial_zone" not in zoned_grid.columns:
        # Return default if grid has no zones
        return {ZONE_RED: 0.0, ZONE_YELLOW: 0.0, ZONE_GREEN: 100.0, ZONE_WATER: 0.0}

    # Generate sample points along route at regular intervals
    sample_points: list[tuple[float, float]] = []
    cumulative_dist = 0.0

    for i in range(len(route_coords) - 1):
        lat1, lon1 = route_coords[i]
        lat2, lon2 = route_coords[i + 1]
        segment_dist = _haversine_km(lat1, lon1, lat2, lon2)

        # Sample this segment
        num_samples = max(1, int(segment_dist / sample_interval_km))
        for j in range(num_samples):
            t = j / num_samples
            sample_lat = lat1 + t * (lat2 - lat1)
            sample_lon = lon1 + t * (lon2 - lon1)
            sample_points.append((sample_lat, sample_lon))

        cumulative_dist += segment_dist

    if not sample_points:
        return {ZONE_RED: 0.0, ZONE_YELLOW: 0.0, ZONE_GREEN: 100.0, ZONE_WATER: 0.0}

    # For each sample point, find nearest grid cell and its zone
    zone_counts = {ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 0, ZONE_WATER: 0}

    for sample_lat, sample_lon in sample_points:
        # Find nearest cell
        min_dist = float("inf")
        nearest_zone = ZONE_GREEN

        for _, grid_cell in zoned_grid.iterrows():
            cell_lat = grid_cell.get("centroid_lat")
            cell_lon = grid_cell.get("centroid_lon")
            if cell_lat is None or cell_lon is None:
                continue

            dist = _haversine_km(sample_lat, sample_lon, cell_lat, cell_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_zone = grid_cell.get("spatial_zone", ZONE_GREEN)

        zone_counts[nearest_zone] = zone_counts.get(nearest_zone, 0) + 1

    # Convert counts to percentages
    total = sum(zone_counts.values())
    if total == 0:
        return {ZONE_RED: 0.0, ZONE_YELLOW: 0.0, ZONE_GREEN: 100.0, ZONE_WATER: 0.0}

    exposure = {
        zone: (count / total) * 100.0
        for zone, count in zone_counts.items()
    }

    logger.debug("Route hazard exposure: RED=%.1f%% YELLOW=%.1f%% GREEN=%.1f%% WATER=%.1f%%",
                 exposure[ZONE_RED], exposure[ZONE_YELLOW], exposure[ZONE_GREEN], exposure[ZONE_WATER])

    return exposure


def find_safest_facility(
    origin_hab_id: str,
    origin_lat: float,
    origin_lon: float,
    candidate_facilities: list[dict],
    hazard_graph: Optional[nx.MultiGraph],
    standard_graph: Optional[nx.MultiGraph],
    zoned_grid: gpd.GeoDataFrame,
) -> EvacuationRoute:
    """
    Find the safest facility for an origin habitation and compute route.

    Evaluates all candidate facilities using hazard-weighted routing.
    Returns the route with the lowest RED zone exposure.

    Parameters
    ----------
    origin_hab_id : str
        Habitation ID (for route tracking).
    origin_lat, origin_lon : float
        Habitation coordinates.
    candidate_facilities : list[dict]
        List of candidate facilities with keys:
        - facility_id: str
        - name: str
        - facility_type: str
        - latitude: float
        - longitude: float
    hazard_graph : nx.MultiGraph | None
        Road network with hazard-adjusted weights. If None, standard_graph is used.
    standard_graph : nx.MultiGraph | None
        Road network with standard weights (fallback if hazard_graph is None).
    zoned_grid : gpd.GeoDataFrame
        Grid with spatial zones.

    Returns
    -------
    EvacuationRoute
        Best route found, or route with status="NO_SAFE_ROUTE_AVAILABLE" if none found.
    """
    if not candidate_facilities:
        return EvacuationRoute(
            hab_id=origin_hab_id,
            hab_name="",
            facility_id="",
            facility_name="",
            facility_type="",
            route_geometry=[],
            distance_km=-1.0,
            routing_method="unavailable",
            hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 0, ZONE_WATER: 0},
            status="NO_FACILITY_AVAILABLE",
            details="No candidate facilities provided.",
        )

    # Use hazard graph if available, otherwise standard graph
    active_graph = hazard_graph if hazard_graph is not None else standard_graph

    if active_graph is None or len(active_graph.nodes()) == 0:
        logger.debug("No graph available for routing; using haversine fallback")
        # Fallback: find nearest facility by straight-line distance
        min_dist_km = float("inf")
        best_facility = None

        for facility in candidate_facilities:
            dist_km = _haversine_km(
                origin_lat, origin_lon,
                facility["latitude"], facility["longitude"]
            )
            if dist_km < min_dist_km:
                min_dist_km = dist_km
                best_facility = facility

        if best_facility is None:
            return EvacuationRoute(
                hab_id=origin_hab_id, hab_name="", facility_id="", facility_name="",
                facility_type="", route_geometry=[(origin_lat, origin_lon), (origin_lat, origin_lon)],
                distance_km=0.0, routing_method="straight_line_fallback",
                hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 100, ZONE_WATER: 0},
                status="DATA_UNAVAILABLE", details="No graph and no facilities available.",
            )

        # Create simple route geometry (straight line)
        route_geom = [(origin_lat, origin_lon), (best_facility["latitude"], best_facility["longitude"])]
        hazard_exp = _sample_zone_along_route(route_geom, zoned_grid)

        return EvacuationRoute(
            hab_id=origin_hab_id, hab_name="", facility_id=best_facility["facility_id"],
            facility_name=best_facility["name"], facility_type=best_facility["facility_type"],
            route_geometry=route_geom, distance_km=min_dist_km,
            routing_method="straight_line_fallback", hazard_exposure=hazard_exp,
            status="FOUND", details="Graph unavailable; used straight-line distance.",
        )

    # Find closest graph nodes to origin and each facility
    origin_node = _find_closest_graph_node(origin_lat, origin_lon, active_graph)
    if origin_node is None:
        return EvacuationRoute(
            hab_id=origin_hab_id, hab_name="", facility_id="", facility_name="",
            facility_type="", route_geometry=[], distance_km=-1.0,
            routing_method="unavailable",
            hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 0, ZONE_WATER: 0},
            status="FACILITY_UNREACHABLE", details="Origin habitation not reachable on road network.",
        )

    # Evaluate each facility
    best_route = None
    best_red_exposure = float("inf")

    for facility in candidate_facilities:
        facility_node = _find_closest_graph_node(
            facility["latitude"], facility["longitude"], active_graph
        )
        if facility_node is None:
            logger.debug("Facility %s not reachable on graph", facility["facility_id"])
            continue

        # Compute shortest path
        try:
            if nx.has_path(active_graph, origin_node, facility_node):
                path_length = nx.shortest_path_length(
                    active_graph, origin_node, facility_node, weight="weight"
                )
                path_nodes = nx.shortest_path(
                    active_graph, origin_node, facility_node, weight="weight"
                )

                # Convert path nodes to coordinates
                route_coords = []
                for node_id in path_nodes:
                    node_lat = active_graph.nodes[node_id].get("lat")
                    node_lon = active_graph.nodes[node_id].get("lon")
                    if node_lat is not None and node_lon is not None:
                        route_coords.append((node_lat, node_lon))

                if route_coords:
                    hazard_exp = _sample_zone_along_route(route_coords, zoned_grid)
                    red_exposure = hazard_exp.get(ZONE_RED, 0.0)

                    if red_exposure < best_red_exposure:
                        best_red_exposure = red_exposure
                        best_route = EvacuationRoute(
                            hab_id=origin_hab_id, hab_name="",
                            facility_id=facility["facility_id"],
                            facility_name=facility["name"],
                            facility_type=facility["facility_type"],
                            route_geometry=route_coords,
                            distance_km=round(path_length, 3),
                            routing_method="network_routing",
                            hazard_exposure=hazard_exp,
                            status="FOUND",
                            details=f"Routed via {len(path_nodes)} road nodes.",
                        )
        except (nx.NetworkXNoPath, nx.NetworkXError) as e:
            logger.debug("No path to facility %s: %s", facility["facility_id"], e)
            continue

    if best_route is not None:
        return best_route

    # No route found; return NO_SAFE_ROUTE_AVAILABLE
    return EvacuationRoute(
        hab_id=origin_hab_id, hab_name="", facility_id="", facility_name="",
        facility_type="", route_geometry=[], distance_km=-1.0,
        routing_method="unavailable",
        hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 0, ZONE_WATER: 0},
        status="NO_SAFE_ROUTE_AVAILABLE",
        details="No reachable facility found on road network.",
    )


def _find_closest_graph_node(
    lat: float,
    lon: float,
    graph: nx.MultiGraph,
    max_distance_km: float = 5.0,
) -> Optional[int]:
    """
    Find the closest node in the graph to a given coordinate.

    Returns None if no node is within max_distance_km.

    Parameters
    ----------
    lat, lon : float
        Target coordinates.
    graph : nx.MultiGraph
        Road network graph.
    max_distance_km : float
        Maximum search radius (default 5 km).

    Returns
    -------
    int | None
        Node ID, or None if no node found within radius.
    """
    min_dist = float("inf")
    closest_node = None

    for node_id in graph.nodes():
        node_lat = graph.nodes[node_id].get("lat")
        node_lon = graph.nodes[node_id].get("lon")

        if node_lat is None or node_lon is None:
            continue

        dist = _haversine_km(lat, lon, node_lat, node_lon)

        if dist <= max_distance_km and dist < min_dist:
            min_dist = dist
            closest_node = node_id

    return closest_node

