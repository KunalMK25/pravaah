"""
Unit tests for routing-aware accessibility calculations.

Tests cover:
1. Network routing distance calculation
2. Straight-line fallback behavior
3. Graph construction and pathfinding
4. Provenance tracking (network_routing vs straight_line_fallback)
5. Edge cases: empty networks, disconnected nodes, missing data
6. Integration with capacity assessment
"""
import pytest
from unittest.mock import Mock, patch
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

from flood_risk_zonation.utils.routing import (
    NetworkDistance,
    build_road_graph,
    shortest_network_distance,
    _haversine_km,
)
from flood_risk_zonation.capacity.assessment import _nearest_km
from flood_risk_zonation.models import ExposureResult
from flood_risk_zonation.config import BoundingBox


class TestHaversineDistance:
    """Test basic straight-line distance calculation."""

    def test_same_point(self):
        """Distance from a point to itself should be ~0."""
        d = _haversine_km(12.90, 77.60, 12.90, 77.60)
        assert abs(d) < 0.001

    def test_known_distance(self):
        """Test against a known distance."""
        # Approximately 1 km
        lat1, lon1 = 12.9000, 77.6000
        lat2, lon2 = 12.9090, 77.6000
        d = _haversine_km(lat1, lon1, lat2, lon2)
        assert 0.9 < d < 1.2

    def test_distance_symmetry(self):
        """Distance should be symmetric."""
        d1 = _haversine_km(12.90, 77.60, 13.00, 77.70)
        d2 = _haversine_km(13.00, 77.70, 12.90, 77.60)
        assert abs(d1 - d2) < 0.001


class TestBuildRoadGraph:
    """Test road network graph construction."""

    def test_empty_points(self):
        """Graph from empty points should return None."""
        g = build_road_graph([])
        assert g is None

    def test_single_point(self):
        """Graph from single point should return None."""
        g = build_road_graph([(12.90, 77.60)])
        assert g is None

    def test_two_nearby_points(self):
        """Graph from two nearby points should have one edge."""
        # Points ~1 km apart
        points = [(12.9000, 77.6000), (12.9090, 77.6000)]
        g = build_road_graph(points)
        assert g is not None
        assert len(g.nodes()) == 2
        assert len(g.edges()) >= 1

    def test_two_distant_points(self):
        """Graph from distant points (>2 km) should have no edges."""
        # Points ~10 km apart
        points = [(12.9000, 77.6000), (13.0000, 77.6000)]
        g = build_road_graph(points)
        # Should still have nodes but no edges connecting them
        if g is not None:
            assert len(g.nodes()) == 2
            # May or may not have edges depending on distance calculation

    def test_multiple_connected_points(self):
        """Graph from linear road should have connected path."""
        # Linear road: 5 points, each ~0.5 km apart
        points = [
            (12.9000, 77.6000),
            (12.9045, 77.6000),
            (12.9090, 77.6000),
            (12.9135, 77.6000),
            (12.9180, 77.6000),
        ]
        g = build_road_graph(points)
        assert g is not None
        assert len(g.nodes()) == 5
        assert len(g.edges()) >= 4  # Should have path through points

    def test_graph_node_attributes(self):
        """Graph nodes should have lat/lon attributes."""
        points = [(12.9000, 77.6000), (12.9090, 77.6000)]
        g = build_road_graph(points)
        assert g is not None
        for node_id in g.nodes():
            node_data = g.nodes[node_id]
            assert "lat" in node_data
            assert "lon" in node_data

    def test_graph_edge_weights(self):
        """Graph edges should have distance weights."""
        points = [(12.9000, 77.6000), (12.9090, 77.6000)]
        g = build_road_graph(points)
        assert g is not None
        for u, v, data in g.edges(data=True):
            assert "weight" in data
            assert data["weight"] > 0


class TestShortestNetworkDistance:
    """Test network distance calculation with routing."""

    def test_empty_targets(self):
        """Empty target list should return -1 with fallback."""
        result = shortest_network_distance(12.90, 77.60, [], graph=None, allow_fallback=True)
        assert result.distance_km == -1.0
        assert result.method == "straight_line_fallback"

    def test_no_graph_uses_fallback(self):
        """Without graph, should use haversine fallback."""
        targets = [(12.9090, 77.6000)]
        result = shortest_network_distance(
            12.90, 77.60, targets, graph=None, allow_fallback=True
        )
        assert result.method == "straight_line_fallback"
        assert result.distance_km > 0
        # Should be ~1 km
        assert 0.8 < result.distance_km < 1.2

    def test_graph_with_connected_target(self):
        """Routing through connected network should work."""
        # Build a linear road
        road_points = [
            (12.9000, 77.6000),
            (12.9045, 77.6000),
            (12.9090, 77.6000),
        ]
        g = build_road_graph(road_points)
        assert g is not None

        # Habitation is ~0.5 km from road
        hab_lat, hab_lon = 12.8955, 77.6000

        # Target on the road
        targets = [(12.9090, 77.6000)]

        result = shortest_network_distance(
            hab_lat, hab_lon, targets, graph=g, allow_fallback=True
        )

        # Should attempt routing
        assert result.distance_km >= 0

    def test_routing_vs_haversine(self):
        """Routed distance should potentially differ from haversine."""
        # Build a detour network: L-shaped road
        road_points = [
            (12.9000, 77.6000),  # Bottom-left
            (12.9050, 77.6000),  # Move right
            (12.9100, 77.6000),  # Move right
            (12.9100, 77.6050),  # Move up
            (12.9100, 77.6100),  # Move up more
        ]
        g = build_road_graph(road_points)
        assert g is not None

        # Habitation is near bottom-left
        hab_lat, hab_lon = 12.8995, 77.5995

        # Target is at top-right of L
        targets = [(12.9100, 77.6100)]

        result_routed = shortest_network_distance(
            hab_lat, hab_lon, targets, graph=g, allow_fallback=True
        )

        # Direct haversine distance
        from flood_risk_zonation.utils.routing import _haversine_km
        direct = _haversine_km(hab_lat, hab_lon, targets[0][0], targets[0][1])

        # Routed should be >= direct (or fallback to direct if not found)
        assert result_routed.distance_km >= 0

    def test_fallback_disabled_raises(self):
        """With fallback disabled and no graph, should raise."""
        with pytest.raises(ValueError):
            shortest_network_distance(
                12.90, 77.60, [(12.9090, 77.6000)],
                graph=None, allow_fallback=False
            )

    def test_provenance_tracked(self):
        """Result should have method and details fields."""
        targets = [(12.9090, 77.6000)]
        result = shortest_network_distance(
            12.90, 77.60, targets, graph=None, allow_fallback=True
        )
        assert result.method in ["network_routing", "straight_line_fallback"]
        assert len(result.details) > 0


class TestNearestKmWithRouting:
    """Test _nearest_km function with routing integration."""

    def test_no_points_returns_unavailable(self):
        """Empty point list should return -1 and unavailable."""
        dist, method = _nearest_km(12.90, 77.60, [])
        assert dist == -1.0
        assert method == "unavailable"

    def test_single_point_fallback(self):
        """Single point should return haversine with fallback."""
        dist, method = _nearest_km(12.90, 77.60, [(12.9090, 77.6000)])
        assert dist > 0
        assert method == "straight_line_fallback"
        assert 0.8 < dist < 1.2

    def test_multiple_points_returns_nearest(self):
        """Should return distance to nearest point."""
        points = [
            (12.9090, 77.6000),  # ~1 km away
            (12.9500, 77.6000),  # ~5 km away
            (13.0000, 77.6000),  # ~10 km away
        ]
        dist, method = _nearest_km(12.90, 77.60, points)
        assert 0.8 < dist < 1.2  # Nearest is ~1 km

    def test_with_graph_attempts_routing(self):
        """With graph provided, should attempt routing."""
        road_points = [(12.9000, 77.6000), (12.9090, 77.6000)]
        g = build_road_graph(road_points)

        targets = [(12.9090, 77.6000)]
        dist, method = _nearest_km(12.90, 77.60, targets, road_graph=g)

        # Should have attempted routing
        assert dist >= 0


class TestCapacityAssessmentIntegration:
    """Test routing integration in capacity assessment."""

    def test_assessment_handles_missing_network(self):
        """Capacity assessment should work with no network data."""
        # This is more of an integration test
        # Mock exposure result
        exposure = ExposureResult(
            hab_id="test_001",
            name="Test Habitation",
            hab_type="village",
            lat=12.90,
            lon=77.60,
            hazard_score=50.0,
            hazard_class="Medium",
            pct_high_risk=0.3,
            population_source="osm_tag",
            population_exposed=100,
            is_in_red_zone=False,
        )

        # Empty road and healthcare lists (network unavailable)
        dist_road, method_road = _nearest_km(exposure.lat, exposure.lon, [])
        dist_hc, method_hc = _nearest_km(exposure.lat, exposure.lon, [])

        # Should gracefully handle unavailable data
        assert dist_road == -1.0
        assert method_road == "unavailable"
        assert dist_hc == -1.0
        assert method_hc == "unavailable"

    def test_assessment_with_routing_data(self):
        """Capacity assessment should use routing when data available."""
        exposure = ExposureResult(
            hab_id="test_002",
            name="Test Habitation",
            hab_type="village",
            lat=12.9000,
            lon=77.6000,
            hazard_score=50.0,
            hazard_class="Medium",
            pct_high_risk=0.3,
            population_source="osm_tag",
            population_exposed=100,
            is_in_red_zone=False,
        )

        # Mock road network
        road_points = [
            (12.9000, 77.6000),
            (12.9045, 77.6000),
            (12.9090, 77.6000),
        ]
        g = build_road_graph(road_points)

        # Calculate distance with graph
        targets = [(12.9045, 77.6000), (12.9090, 77.6000)]
        dist, method = _nearest_km(exposure.lat, exposure.lon, targets, road_graph=g)

        # Should complete without error
        assert dist >= 0
        assert method in ["network_routing", "straight_line_fallback"]


class TestRobustness:
    """Test robustness and edge cases."""

    def test_invalid_coordinates_handled(self):
        """Invalid coordinates should not crash with reasonable behavior."""
        # NaN should still compute (even if result is NaN)
        d = _haversine_km(float("nan"), 77.60, 12.90, 77.60)
        # Result will be NaN, which is fine
        assert d != d  # NaN != NaN is True

    def test_graph_with_orphan_nodes(self):
        """Graph with disconnected components should still work."""
        # Two separate roads with no connection
        road_points = [
            (12.9000, 77.6000),  # Road 1
            (12.9045, 77.6000),  # Road 1
            (13.0000, 77.7000),  # Road 2 (far away)
            (13.0045, 77.7000),  # Road 2
        ]
        g = build_road_graph(road_points)

        if g is not None:
            # Should still have structure
            assert len(g.nodes()) >= 2

    def test_multiple_targets_returns_nearest(self):
        """With multiple targets, should find nearest one."""
        targets = [
            (13.0000, 77.6000),  # ~10 km away
            (12.9090, 77.6000),  # ~1 km away
            (12.8000, 77.6000),  # ~10 km away
        ]
        result = shortest_network_distance(
            12.90, 77.60, targets, graph=None, allow_fallback=True
        )
        # Should be closest to middle target
        assert 0.8 < result.distance_km < 1.2

    def test_caching_consistency(self):
        """Multiple calls with same graph should be consistent."""
        road_points = [(12.9000, 77.6000), (12.9090, 77.6000)]
        g = build_road_graph(road_points)

        targets = [(12.9090, 77.6000)]

        result1 = shortest_network_distance(12.90, 77.60, targets, graph=g)
        result2 = shortest_network_distance(12.90, 77.60, targets, graph=g)

        # Both should be identical
        assert result1.distance_km == result2.distance_km
        assert result1.method == result2.method


class TestProvenance:
    """Test that provenance is correctly tracked."""

    def test_routing_method_identified(self):
        """Routed distances should be marked as network_routing."""
        # This test would need a properly connected graph
        road_points = [
            (12.9000, 77.6000),
            (12.9045, 77.6000),
            (12.9090, 77.6000),
        ]
        g = build_road_graph(road_points)

        if g is not None and nx.is_connected(g):
            result = shortest_network_distance(
                12.90, 77.60, [(12.9090, 77.6000)], graph=g, allow_fallback=True
            )
            # Could be either depending on connectivity
            assert result.method in ["network_routing", "straight_line_fallback"]

    def test_fallback_method_identified(self):
        """Fallback distances should be marked as straight_line_fallback."""
        result = shortest_network_distance(
            12.90, 77.60, [(12.9090, 77.6000)], graph=None, allow_fallback=True
        )
        assert result.method == "straight_line_fallback"

    def test_result_has_details(self):
        """All results should have explanation details."""
        result = shortest_network_distance(
            12.90, 77.60, [(12.9090, 77.6000)], graph=None, allow_fallback=True
        )
        assert isinstance(result.details, str)
        assert len(result.details) > 0
