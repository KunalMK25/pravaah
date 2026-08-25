"""Unit tests for carrying capacity assessment."""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from unittest.mock import patch

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.models import ExposureResult
from flood_risk_zonation.capacity.assessment import (
    _capacity_status,
    _compute_safe_area,
    _nearest_km,
    _haversine_km,
    assess_capacity,
    CAPACITY_WEIGHTS,
)


def _make_grid(n: int = 16, risk_class: str = "Low"):
    """Build a small grid for testing safe-area computation."""
    lats = [12.84 + i * 0.003 for i in range(n)]
    lons = [77.55 + j * 0.003 for j in range(n)]
    geoms = [box(lon - 0.0015, lat - 0.0015, lon + 0.0015, lat + 0.0015)
             for lat, lon in zip(lats, lons)]
    df = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(n)],
        "centroid_lat": lats,
        "centroid_lon": lons,
        "risk_class": [risk_class] * n,
        "risk_score": [20.0 if risk_class == "Low" else 80.0] * n,
    })
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


def _make_exposure(lat=12.848, lon=77.558):
    return ExposureResult(
        hab_id="h1",
        name="Test",
        hab_type="village",
        lat=lat,
        lon=lon,
        hazard_score=50.0,
        hazard_class="Medium",
        pct_high_risk=0.3,
        population_source="UNKNOWN",
        population_exposed=None,
        is_in_red_zone=False,
    )


class TestCapacityWeights:
    def test_sum_to_one(self):
        assert abs(sum(CAPACITY_WEIGHTS.values()) - 1.0) < 1e-9


class TestCapacityStatus:
    def test_adequate(self):
        assert _capacity_status(0.9) == "ADEQUATE"
        assert _capacity_status(0.60) == "ADEQUATE"

    def test_stressed(self):
        assert _capacity_status(0.4) == "STRESSED"
        assert _capacity_status(0.35) == "STRESSED"

    def test_critical(self):
        assert _capacity_status(0.2) == "CRITICAL"
        assert _capacity_status(0.0) == "CRITICAL"


class TestHaversineKm:
    def test_zero(self):
        assert _haversine_km(12.9, 77.6, 12.9, 77.6) == pytest.approx(0.0, abs=0.001)

    def test_positive(self):
        d = _haversine_km(12.84, 77.55, 12.91, 77.62)
        assert d > 0


class TestNearestKm:
    def test_empty_points(self):
        assert _nearest_km(12.9, 77.6, []) == -1.0

    def test_single_point(self):
        d = _nearest_km(12.9, 77.6, [(12.9, 77.6)])
        assert d == pytest.approx(0.0, abs=0.001)

    def test_picks_closest(self):
        d = _nearest_km(0.0, 0.0, [(1.0, 0.0), (0.5, 0.0)])
        assert d == pytest.approx(55.6, rel=0.05)


class TestComputeSafeArea:
    def test_all_low_risk_returns_positive(self):
        grid = _make_grid(n=16, risk_class="Low")
        area = _compute_safe_area(12.844, 77.556, grid, radius_km=10.0)
        assert area > 0

    def test_all_high_risk_returns_zero(self):
        grid = _make_grid(n=16, risk_class="High")
        area = _compute_safe_area(12.844, 77.556, grid, radius_km=10.0)
        assert area == 0.0

    def test_returns_float(self):
        grid = _make_grid()
        area = _compute_safe_area(12.844, 77.556, grid)
        assert isinstance(area, float)


class TestAssessCapacity:
    def test_no_network_returns_result(self, tmp_path):
        exp = _make_exposure()
        grid = _make_grid(n=9, risk_class="Low")
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = assess_capacity(exp, grid, bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.hab_id == "h1"
        assert 0.0 <= result.capacity_score <= 1.0
        assert result.capacity_status in ("ADEQUATE", "STRESSED", "CRITICAL")

    def test_no_network_distances_are_unknown(self, tmp_path):
        exp = _make_exposure()
        grid = _make_grid(n=9, risk_class="Low")
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = assess_capacity(exp, grid, bbox, cache_dir=str(tmp_path), allow_network=False)
        # No network → no healthcare / road data
        assert result.nearest_healthcare_km == -1.0
        assert result.nearest_road_km == -1.0

    def test_safe_area_in_result(self, tmp_path):
        exp = _make_exposure()
        grid = _make_grid(n=16, risk_class="Low")
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = assess_capacity(exp, grid, bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.safe_area_km2 >= 0

    def test_shelter_unavailable_by_default(self, tmp_path):
        exp = _make_exposure()
        grid = _make_grid()
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = assess_capacity(exp, grid, bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.shelter_capacity is None
        assert result.shelter_source == "unavailable"

    @patch("flood_risk_zonation.capacity.assessment._fetch")
    def test_with_mock_healthcare(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {
            "elements": [
                {"type": "node", "id": 1, "lat": 12.87, "lon": 77.58,
                 "tags": {"amenity": "hospital", "name": "General Hospital"}}
            ]
        }
        exp = _make_exposure()
        grid = _make_grid(n=9, risk_class="Low")
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = assess_capacity(exp, grid, bbox, cache_dir=str(tmp_path), allow_network=True)
        assert result.nearest_healthcare_km >= 0
