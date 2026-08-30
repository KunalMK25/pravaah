"""
Unit tests for emergency facility models and layer creation.

Tests facility data structures, validation, and visualization layer creation.
"""
import pytest
from flood_risk_zonation.capacity.emergency import EmergencyFacility, EvacuationRoute
from flood_risk_zonation.spatial_zones.classifier import (
    ZONE_RED, ZONE_YELLOW, ZONE_GREEN, ZONE_WATER
)


class TestEmergencyFacility:
    """Test EmergencyFacility data model."""

    def test_create_hospital(self):
        """Test creating a hospital facility."""
        fac = EmergencyFacility(
            facility_id="hospital_001",
            name="City General Hospital",
            facility_type="hospital",
            latitude=12.9716,
            longitude=77.5946,
            source="osm_overpass",
            osm_id=123456,
            metadata={"operator": "Public Health", "contact:phone": "+91-80-123456"}
        )
        assert fac.facility_id == "hospital_001"
        assert fac.facility_type == "hospital"
        assert fac.name == "City General Hospital"
        assert fac.latitude == 12.9716

    def test_create_shelter(self):
        """Test creating a shelter facility."""
        fac = EmergencyFacility(
            facility_id="shelter_001",
            name="Community Shelter A",
            facility_type="shelter",
            latitude=12.9700,
            longitude=77.5900,
            source="osm_overpass",
        )
        assert fac.facility_type == "shelter"
        assert fac.osm_id is None

    def test_invalid_facility_type(self):
        """Test that invalid facility type raises ValueError."""
        with pytest.raises(ValueError):
            EmergencyFacility(
                facility_id="bad_001",
                name="Bad Facility",
                facility_type="invalid_type",
                latitude=12.0,
                longitude=77.0,
                source="test",
            )

    def test_facility_repr(self):
        """Test string representation."""
        fac = EmergencyFacility(
            facility_id="test_001",
            name="Test Facility",
            facility_type="clinic",
            latitude=12.5,
            longitude=77.5,
            source="test",
        )
        repr_str = repr(fac)
        assert "test_001" in repr_str
        assert "Test Facility" in repr_str
        assert "clinic" in repr_str


class TestEvacuationRoute:
    """Test EvacuationRoute data model."""

    def test_create_successful_route(self):
        """Test creating a successful evacuation route."""
        route = EvacuationRoute(
            hab_id="hab_001",
            hab_name="Settlement A",
            facility_id="hospital_001",
            facility_name="City Hospital",
            facility_type="hospital",
            route_geometry=[(12.97, 77.59), (12.98, 77.60)],
            distance_km=2.5,
            routing_method="network_routing",
            hazard_exposure={ZONE_RED: 5.0, ZONE_YELLOW: 15.0, ZONE_GREEN: 80.0, ZONE_WATER: 0.0},
            status="FOUND",
        )
        assert route.status == "FOUND"
        assert route.distance_km == 2.5
        assert route.hazard_exposure[ZONE_RED] == 5.0

    def test_route_hazard_exposure_validation(self):
        """Test that hazard_exposure percentages must sum to 100%."""
        # Should pass with ~100%
        route = EvacuationRoute(
            hab_id="hab_001",
            hab_name="Test",
            facility_id="fac_001",
            facility_name="Test Facility",
            facility_type="hospital",
            route_geometry=[(12.0, 77.0)],
            distance_km=1.0,
            routing_method="straight_line_fallback",
            hazard_exposure={ZONE_RED: 25.0, ZONE_YELLOW: 25.0, ZONE_GREEN: 50.0, ZONE_WATER: 0.0},
            status="FOUND",
        )
        assert route.hazard_exposure[ZONE_RED] == 25.0

        # Should fail if percentages don't sum to ~100%
        with pytest.raises(ValueError):
            EvacuationRoute(
                hab_id="hab_001",
                hab_name="Test",
                facility_id="fac_001",
                facility_name="Test Facility",
                facility_type="hospital",
                route_geometry=[(12.0, 77.0)],
                distance_km=1.0,
                routing_method="straight_line_fallback",
                hazard_exposure={ZONE_RED: 50.0, ZONE_YELLOW: 50.0, ZONE_GREEN: 50.0, ZONE_WATER: 0.0},
                status="FOUND",
            )

    def test_route_status_validation(self):
        """Test that only valid status values are accepted."""
        # Valid status
        route = EvacuationRoute(
            hab_id="hab_001",
            hab_name="Test",
            facility_id="fac_001",
            facility_name="Test Facility",
            facility_type="hospital",
            route_geometry=[],
            distance_km=-1.0,
            routing_method="unavailable",
            hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 0, ZONE_WATER: 100},
            status="NO_FACILITY_AVAILABLE",
        )
        assert route.status == "NO_FACILITY_AVAILABLE"

        # Invalid status
        with pytest.raises(ValueError):
            EvacuationRoute(
                hab_id="hab_001",
                hab_name="Test",
                facility_id="fac_001",
                facility_name="Test Facility",
                facility_type="hospital",
                route_geometry=[],
                distance_km=-1.0,
                routing_method="unavailable",
                hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 100, ZONE_WATER: 0},
                status="INVALID_STATUS",
            )

    def test_is_safe_check(self):
        """Test is_safe convenience method."""
        # Safe route (RED <= 10%)
        safe_route = EvacuationRoute(
            hab_id="hab_001",
            hab_name="Test",
            facility_id="fac_001",
            facility_name="Test Facility",
            facility_type="hospital",
            route_geometry=[(12.0, 77.0)],
            distance_km=1.0,
            routing_method="network_routing",
            hazard_exposure={ZONE_RED: 5.0, ZONE_YELLOW: 15.0, ZONE_GREEN: 80.0, ZONE_WATER: 0.0},
            status="FOUND",
        )
        assert safe_route.is_safe(red_threshold=10.0) is True

        # Unsafe route (RED > 10%)
        unsafe_route = EvacuationRoute(
            hab_id="hab_002",
            hab_name="Test 2",
            facility_id="fac_001",
            facility_name="Test Facility",
            facility_type="hospital",
            route_geometry=[(12.0, 77.0)],
            distance_km=1.0,
            routing_method="network_routing",
            hazard_exposure={ZONE_RED: 25.0, ZONE_YELLOW: 25.0, ZONE_GREEN: 50.0, ZONE_WATER: 0.0},
            status="FOUND",
        )
        assert unsafe_route.is_safe(red_threshold=10.0) is False

        # Failed route (status != FOUND)
        failed_route = EvacuationRoute(
            hab_id="hab_003",
            hab_name="Test 3",
            facility_id="fac_001",
            facility_name="Test Facility",
            facility_type="hospital",
            route_geometry=[],
            distance_km=-1.0,
            routing_method="unavailable",
            hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 100, ZONE_WATER: 0},
            status="NO_SAFE_ROUTE_AVAILABLE",
        )
        assert failed_route.is_safe() is False

    def test_hazard_summary(self):
        """Test hazard_summary convenience method."""
        route = EvacuationRoute(
            hab_id="hab_001",
            hab_name="Test",
            facility_id="fac_001",
            facility_name="Test Facility",
            facility_type="hospital",
            route_geometry=[(12.0, 77.0)],
            distance_km=1.0,
            routing_method="network_routing",
            hazard_exposure={ZONE_RED: 5.5, ZONE_YELLOW: 14.5, ZONE_GREEN: 80.0, ZONE_WATER: 0.0},
            status="FOUND",
        )
        summary = route.hazard_summary()
        assert "RED:5.5%" in summary
        assert "YELLOW:14.5%" in summary
        assert "GREEN:80.0%" in summary

    def test_route_repr(self):
        """Test string representation."""
        route = EvacuationRoute(
            hab_id="hab_001",
            hab_name="Settlement A",
            facility_id="fac_001",
            facility_name="Hospital X",
            facility_type="hospital",
            route_geometry=[(12.0, 77.0)],
            distance_km=3.25,
            routing_method="network_routing",
            hazard_exposure={ZONE_RED: 8.0, ZONE_YELLOW: 12.0, ZONE_GREEN: 80.0, ZONE_WATER: 0.0},
            status="FOUND",
        )
        repr_str = repr(route)
        assert "hab_001" in repr_str
        assert "fac_001" in repr_str
        assert "3.25km" in repr_str
        assert "RED:8.0%" in repr_str


class TestFacilityLayerCreation:
    """Test emergency facility visualization layer creation."""

    def test_add_emergency_facilities_layer_with_hospitals(self):
        """Test adding hospital markers to a folium map."""
        try:
            import folium
        except ImportError:
            pytest.skip("folium not installed")

        from flood_risk_zonation.visualization.layers import add_emergency_facilities_layer

        hospitals = [
            EmergencyFacility(
                facility_id="h1",
                name="Hospital 1",
                facility_type="hospital",
                latitude=12.97,
                longitude=77.59,
                source="osm_overpass",
            ),
            EmergencyFacility(
                facility_id="h2",
                name="Hospital 2",
                facility_type="clinic",
                latitude=12.98,
                longitude=77.60,
                source="osm_overpass",
            ),
        ]

        m = folium.Map(location=[12.97, 77.59], zoom_start=12)
        result_map = add_emergency_facilities_layer(m, hospitals=hospitals, shelters=None)

        # Verify map has layers
        assert result_map is not None
        assert len(result_map._children) > 0

    def test_add_emergency_facilities_layer_with_shelters(self):
        """Test adding shelter markers to a folium map."""
        try:
            import folium
        except ImportError:
            pytest.skip("folium not installed")

        from flood_risk_zonation.visualization.layers import add_emergency_facilities_layer

        shelters = [
            EmergencyFacility(
                facility_id="s1",
                name="Shelter A",
                facility_type="shelter",
                latitude=12.96,
                longitude=77.58,
                source="osm_overpass",
            ),
        ]

        m = folium.Map(location=[12.96, 77.58], zoom_start=12)
        result_map = add_emergency_facilities_layer(m, hospitals=None, shelters=shelters)

        assert result_map is not None
        assert len(result_map._children) > 0

    def test_add_evacuation_routes_layer(self):
        """Test adding evacuation route polylines to a folium map."""
        try:
            import folium
        except ImportError:
            pytest.skip("folium not installed")

        from flood_risk_zonation.visualization.layers import add_evacuation_routes_layer

        routes = [
            EvacuationRoute(
                hab_id="hab_001",
                hab_name="Settlement A",
                facility_id="fac_001",
                facility_name="Hospital",
                facility_type="hospital",
                route_geometry=[(12.97, 77.59), (12.98, 77.60)],
                distance_km=2.5,
                routing_method="network_routing",
                hazard_exposure={ZONE_RED: 5.0, ZONE_YELLOW: 15.0, ZONE_GREEN: 80.0, ZONE_WATER: 0.0},
                status="FOUND",
            ),
        ]

        m = folium.Map(location=[12.97, 77.59], zoom_start=12)
        result_map = add_evacuation_routes_layer(m, evacuation_routes=routes)

        assert result_map is not None
        assert len(result_map._children) > 0

    def test_add_evacuation_routes_layer_empty(self):
        """Test that empty routes list doesn't crash."""
        try:
            import folium
        except ImportError:
            pytest.skip("folium not installed")

        from flood_risk_zonation.visualization.layers import add_evacuation_routes_layer

        m = folium.Map(location=[12.97, 77.59], zoom_start=12)
        result_map = add_evacuation_routes_layer(m, evacuation_routes=[])

        assert result_map is not None

    def test_add_evacuation_routes_layer_failed_status(self):
        """Test that routes with non-FOUND status are skipped."""
        try:
            import folium
        except ImportError:
            pytest.skip("folium not installed")

        from flood_risk_zonation.visualization.layers import add_evacuation_routes_layer

        routes = [
            EvacuationRoute(
                hab_id="hab_001",
                hab_name="Settlement A",
                facility_id="",
                facility_name="",
                facility_type="",
                route_geometry=[],
                distance_km=-1.0,
                routing_method="unavailable",
                hazard_exposure={ZONE_RED: 0, ZONE_YELLOW: 0, ZONE_GREEN: 100, ZONE_WATER: 0},
                status="NO_SAFE_ROUTE_AVAILABLE",
            ),
        ]

        m = folium.Map(location=[12.97, 77.59], zoom_start=12)
        result_map = add_evacuation_routes_layer(m, evacuation_routes=routes)

        # Should not crash; just skip failed routes
        assert result_map is not None
