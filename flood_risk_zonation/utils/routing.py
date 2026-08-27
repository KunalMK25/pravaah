"""
PRAVAAH-AI — Routing-aware network distance calculation.

METHODOLOGY (transparent, fallback-safe):

When OSM road network data is available, this module constructs an undirected
graph from road geometries and calculates shortest-path distances between
a habitation and the network (or healthcare facilities on/near the network).

If routing fails or network data is unavailable, it gracefully falls back to
straight-line (haversine) distance. The fallback is explicitly tracked via
provenance so downstream consumers know which distance calculation was used.

Routing is computed lazily and cached per origin to avoid repeated calculations.

LIMITATIONS (documented):
  - Road network may be incomplete in OSM for remote areas
  - Healthcare facilities are matched to nearest road node; actual routing
    from facility to habitation is not performed
  - Routing uses simplified undirected graph (ignores one-way restrictions,
    turn restrictions, etc.)
  - Disconnected habitations or facilities (outside the network) use fallback

FALLBACK BEHAVIOR:
  If routing is unavailable or fails:
    - Use haversine (straight-line) distance
    - Mark the result with method='straight_line_fallback'
  Consumers MUST check the method field to distinguish routed vs fallback.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import networkx as nx
from math import cos, radians, sqrt

logger = logging.getLogger(__name__)


@dataclass
class NetworkDistance:
    """Result of a distance calculation with provenance."""

    distance_km: float
    """The calculated distance in kilometers."""

    method: str
    """One of: 'network_routing', 'straight_line_fallback'."""

    details: str = ""
    """Optional explanation (e.g., 'node not reachable' or 'graph unavailable')."""


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute straight-line distance in km using Haversine formula."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * (dlon / 2) ** 2
    return R * 2 * sqrt(max(a, 0))


def build_road_graph(road_points: list[tuple[float, float]]) -> Optional[nx.MultiGraph]:
    """
    Build a road network graph from OSM road geometry points.

    This is a simplified approach: each road point (lat, lon) becomes a node.
    Nodes are connected to form a spatial graph. No topological ordering is
    assumed; edges represent approximate network connectivity.

    Parameters
    ----------
    road_points : list[tuple[float, float]]
        List of (lat, lon) tuples representing road network nodes.

    Returns
    -------
    nx.MultiGraph or None
        Undirected multigraph, or None if insufficient points to build graph.
    """
    if not road_points or len(road_points) < 2:
        return None

    try:
        G = nx.MultiGraph()

        # Add nodes with coordinates
        for i, (lat, lon) in enumerate(road_points):
            G.add_node(i, lat=lat, lon=lon)

        # Connect nearby nodes (within ~2 km) to form edges
        # This simulates road connectivity without explicit topology
        for i, (lat1, lon1) in enumerate(road_points):
            for j, (lat2, lon2) in enumerate(road_points):
                if i < j:  # avoid duplicate edges
                    dist_km = _haversine_km(lat1, lon1, lat2, lon2)
                    if dist_km <= 2.0 and dist_km > 0:
                        G.add_edge(i, j, weight=dist_km)

        if len(G.edges()) == 0:
            logger.warning("Road graph has no edges; fallback to haversine")
            return None

        logger.debug("Built road graph: %d nodes, %d edges", len(G.nodes()), len(G.edges()))
        return G

    except Exception as e:
        logger.warning("Failed to build road graph: %s", e)
        return None


def shortest_network_distance(
    hab_lat: float,
    hab_lon: float,
    target_points: list[tuple[float, float]],
    graph: Optional[nx.MultiGraph] = None,
    allow_fallback: bool = True,
) -> NetworkDistance:
    """
    Calculate shortest-path distance from habitation to nearest target point.

    If a graph is available and connected, use shortest-path routing.
    Otherwise, fall back to haversine distance (if allow_fallback=True).

    Parameters
    ----------
    hab_lat : float
        Habitation latitude.
    hab_lon : float
        Habitation longitude.
    target_points : list[tuple[float, float]]
        List of (lat, lon) target points (e.g., healthcare facilities, road network).
    graph : nx.MultiGraph, optional
        Pre-built road network graph. If None or routing fails, uses fallback.
    allow_fallback : bool
        If True, use haversine on failure. If False, raise exception.

    Returns
    -------
    NetworkDistance
        Distance value and method identifier (routed or fallback).
    """
    if not target_points:
        if allow_fallback:
            return NetworkDistance(distance_km=-1.0, method="straight_line_fallback",
                                    details="No target points available")
        raise ValueError("No target points available")

    # Attempt routing if graph is available
    if graph is not None and len(graph.nodes()) > 0:
        try:
            # Find closest graph node to habitation
            min_dist_to_hab = float("inf")
            closest_hab_node = None
            for node_id in graph.nodes():
                node_lat = graph.nodes[node_id].get("lat")
                node_lon = graph.nodes[node_id].get("lon")
                if node_lat is not None and node_lon is not None:
                    dist = _haversine_km(hab_lat, hab_lon, node_lat, node_lon)
                    if dist < min_dist_to_hab:
                        min_dist_to_hab = dist
                        closest_hab_node = node_id

            if closest_hab_node is None:
                logger.debug("Could not find closest node to habitation")
                raise ValueError("No graph nodes with coordinates")

            # Find distances from each target to the graph, then shortest path
            min_total_distance = float("inf")

            for target_lat, target_lon in target_points:
                # Find closest graph node to this target
                min_dist_to_target = float("inf")
                closest_target_node = None
                for node_id in graph.nodes():
                    node_lat = graph.nodes[node_id].get("lat")
                    node_lon = graph.nodes[node_id].get("lon")
                    if node_lat is not None and node_lon is not None:
                        dist = _haversine_km(target_lat, target_lon, node_lat, node_lon)
                        if dist < min_dist_to_target:
                            min_dist_to_target = dist
                            closest_target_node = node_id

                if closest_target_node is None:
                    continue

                # Compute shortest path on graph
                try:
                    if nx.has_path(graph, closest_hab_node, closest_target_node):
                        path_length = nx.shortest_path_length(
                            graph, closest_hab_node, closest_target_node, weight="weight"
                        )
                        # Add distances from endpoints to graph
                        total_distance = min_dist_to_hab + path_length + min_dist_to_target
                        if total_distance < min_total_distance:
                            min_total_distance = total_distance
                except nx.NetworkXError:
                    continue

            if min_total_distance < float("inf"):
                logger.debug("Routed distance: %.3f km", min_total_distance)
                return NetworkDistance(
                    distance_km=round(min_total_distance, 3),
                    method="network_routing",
                    details=f"Shortest-path via {len(graph.nodes())} road nodes"
                )
            else:
                logger.debug("No path found in graph; falling back to haversine")

        except Exception as e:
            logger.debug("Routing failed (%s); falling back to haversine", e)

    # Fallback: haversine distance to nearest target
    if allow_fallback:
        haversine_distances = [_haversine_km(hab_lat, hab_lon, lat, lon) for lat, lon in target_points]
        min_haversine = min(haversine_distances)
        return NetworkDistance(
            distance_km=round(min_haversine, 3),
            method="straight_line_fallback",
            details="Graph unavailable or routing failed; using haversine"
        )
    else:
        raise ValueError("Routing failed and fallback disabled")
