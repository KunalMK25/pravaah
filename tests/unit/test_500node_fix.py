"""
Regression tests for 500-node routing cutoff removal.

Verifies that the arbitrary 500-node threshold has been removed
and that network routing is attempted regardless of graph size.
"""
import pytest
import networkx as nx
from flood_risk_zonation.utils.routing import build_road_graph, shortest_network_distance


class TestRemoval500NodeCutoff:
    """Verify 500-node cutoff is removed and routing works for all sizes."""

    def test_small_graph_under_500_nodes(self):
        """Verify routing works for small graph (<500 nodes)."""
        # Create a small graph with 50 nodes
        road_points = [(12.0 + i*0.01, 77.0 + i*0.01) for i in range(50)]
        graph = build_road_graph(road_points)
        
        assert graph is not None
        assert len(graph.nodes()) == 50
        
        # Perform routing from habitation to targets
        result = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.5, 77.5)],
            graph=graph,
            allow_fallback=True
        )
        
        assert result.distance_km > 0
        assert result.method == "network_routing"

    def test_exactly_500_nodes_still_routes(self):
        """Verify routing works for exactly 500 nodes."""
        # Create a graph with exactly 500 nodes
        road_points = [(12.0 + (i % 22)*0.01, 77.0 + (i // 22)*0.01) for i in range(500)]
        graph = build_road_graph(road_points)
        
        assert graph is not None
        assert len(graph.nodes()) == 500
        
        # Should use network routing, not fallback
        result = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.2, 77.2)],
            graph=graph,
            allow_fallback=True
        )
        
        assert result.distance_km > 0
        assert result.method == "network_routing"

    def test_over_500_nodes_still_routes(self):
        """Verify routing works for >500 nodes (the fixed case)."""
        # Create a graph with 600 nodes (previously would have triggered fallback)
        road_points = [(12.0 + (i % 25)*0.01, 77.0 + (i // 25)*0.01) for i in range(600)]
        graph = build_road_graph(road_points)
        
        assert graph is not None
        assert len(graph.nodes()) > 500
        
        # Should use network routing, NOT straight-line fallback
        result = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.2, 77.2)],
            graph=graph,
            allow_fallback=True
        )
        
        # Key assertion: result should indicate network routing
        assert result.method == "network_routing", \
            f"Expected network_routing but got {result.method} for 600-node graph"
        assert result.distance_km > 0

    def test_large_graph_1000_nodes(self):
        """Verify routing works for 1000 nodes."""
        # Create a large graph
        road_points = [(12.0 + (i % 32)*0.01, 77.0 + (i // 32)*0.01) for i in range(1000)]
        graph = build_road_graph(road_points)
        
        assert graph is not None
        assert len(graph.nodes()) == 1000
        
        result = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.3, 77.3)],
            graph=graph,
            allow_fallback=True
        )
        
        # Should use network routing
        assert result.method == "network_routing"
        assert result.distance_km > 0

    def test_fallback_only_on_genuine_failure(self):
        """Verify fallback is used ONLY for genuine routing failures, not graph size."""
        # Test 1: Graph available → use routing
        road_points = [(12.0 + i*0.01, 77.0 + i*0.01) for i in range(600)]
        graph = build_road_graph(road_points)
        
        result = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.6, 77.6)],
            graph=graph,
            allow_fallback=True
        )
        assert result.method == "network_routing"
        
        # Test 2: No graph available → use fallback
        result_no_graph = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.6, 77.6)],
            graph=None,
            allow_fallback=True
        )
        assert result_no_graph.method == "straight_line_fallback"
        
        # Test 3: Empty graph → use fallback
        empty_graph = nx.MultiGraph()
        result_empty = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.6, 77.6)],
            graph=empty_graph,
            allow_fallback=True
        )
        assert result_empty.method == "straight_line_fallback"

    def test_missing_node_handling_after_fix(self):
        """Verify nodes outside network still fall back correctly."""
        # Create small network
        road_points = [(12.1, 77.1), (12.2, 77.2), (12.3, 77.3)]
        graph = build_road_graph(road_points)
        
        # Query from point far outside network
        result = shortest_network_distance(
            hab_lat=15.0,  # Far away
            hab_lon=80.0,
            target_points=[(12.1, 77.1)],
            graph=graph,
            allow_fallback=True
        )
        
        # Should fall back (habitation too far from network)
        assert result.method == "straight_line_fallback"

    def test_disconnected_graph_fallback(self):
        """Verify fallback works for disconnected graph."""
        # Create disconnected graph (two islands)
        G = nx.MultiGraph()
        # Island 1
        G.add_node(0, lat=12.1, lon=77.1)
        G.add_node(1, lat=12.2, lon=77.2)
        G.add_edge(0, 1, weight=10.0)
        
        # Island 2 (disconnected)
        G.add_node(2, lat=13.1, lon=78.1)
        G.add_node(3, lat=13.2, lon=78.2)
        G.add_edge(2, 3, weight=10.0)
        
        # Try to route from island 1 to island 2 target
        result = shortest_network_distance(
            hab_lat=12.15,
            hab_lon=77.15,
            target_points=[(13.1, 78.1)],
            graph=G,
            allow_fallback=True
        )
        
        # Should fall back because no path exists
        assert result.method == "straight_line_fallback"

    def test_routing_result_correctness(self):
        """Verify routed distances are correct and > fallback in many cases."""
        # Create a simple triangle
        road_points = [(12.0, 77.0), (12.01, 77.0), (12.005, 77.01)]
        graph = build_road_graph(road_points)
        
        # Route from one corner to another
        routed = shortest_network_distance(
            hab_lat=12.0,
            hab_lon=77.0,
            target_points=[(12.005, 77.01)],
            graph=graph,
            allow_fallback=False  # Force routing
        )
        
        # Fallback distance
        from flood_risk_zonation.utils.routing import _haversine_km
        fallback_dist = _haversine_km(12.0, 77.0, 12.005, 77.01)
        
        assert routed.method == "network_routing"
        # Routed distance may be >= fallback (depends on graph path)
        assert routed.distance_km >= fallback_dist * 0.95  # Allow small variance
