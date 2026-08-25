"""Unit tests for habitation exposure analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box

from flood_risk_zonation.models import Habitation, HabitationDataset
from flood_risk_zonation.exposure.analysis import (
    analyse_exposure,
    _dominant_class,
    _haversine_km,
    _classify_hazard,
)


def _make_grid(n: int = 20, risk_classes=None, scores=None):
    """Build a minimal scored GeoDataFrame for tests."""
    if risk_classes is None:
        risk_classes = ["High"] * (n // 2) + ["Low"] * (n - n // 2)
    if scores is None:
        scores = [80.0] * (n // 2) + [20.0] * (n - n // 2)
    lats = [12.84 + i * 0.002 for i in range(n)]
    lons = [77.55 + i * 0.002 for i in range(n)]
    geoms = [box(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
             for lat, lon in zip(lats, lons)]
    df = pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n)],
        "centroid_lat": lats,
        "centroid_lon": lons,
        "risk_score": scores,
        "risk_class": risk_classes,
        "elevation_m": [30.0] * n,
        "dist_water_m": [500.0] * n,
        "drainage_capacity": [0.4] * n,
    })
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


class TestHaversine:
    def test_zero_distance(self):
        d = _haversine_km(12.9, 77.6, 12.9, 77.6)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        # ~111.2 km per degree latitude
        d = _haversine_km(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111.2, rel=0.01)


class TestDominantClass:
    def test_high_dominates(self):
        assert _dominant_class(["Low", "Medium", "High"]) == "High"

    def test_all_low(self):
        assert _dominant_class(["Low", "Low", "Low"]) == "Low"

    def test_empty(self):
        assert _dominant_class([]) == "Low"

    def test_water_lowest(self):
        assert _dominant_class(["Water", "Low"]) == "Low"


class TestClassifyHazard:
    def test_high_score(self):
        assert _classify_hazard(80.0, 0.6, 33.0, 66.0) == "High"

    def test_medium_score(self):
        assert _classify_hazard(50.0, 0.1, 33.0, 66.0) == "Medium"

    def test_low_score(self):
        assert _classify_hazard(20.0, 0.0, 33.0, 66.0) == "Low"

    def test_high_pct_overrides_medium_score(self):
        assert _classify_hazard(50.0, 0.6, 33.0, 66.0) == "High"


class TestAnalyseExposure:
    def test_empty_habitations(self):
        grid = _make_grid()
        ds = HabitationDataset(habitations=[], source="fallback", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results == []

    def test_single_habitation_high_risk(self):
        grid = _make_grid(n=10, risk_classes=["High"] * 10, scores=[80.0] * 10)
        hab = Habitation("h1", "Test", "village", 12.848, 77.558, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert len(results) == 1
        assert results[0].is_in_red_zone is True
        assert results[0].hazard_class == "High"

    def test_single_habitation_low_risk(self):
        grid = _make_grid(n=10, risk_classes=["Low"] * 10, scores=[20.0] * 10)
        hab = Habitation("h1", "Safe", "hamlet", 12.848, 77.558, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results[0].is_in_red_zone is False
        assert results[0].hazard_class == "Low"

    def test_population_from_osm_tag(self):
        grid = _make_grid(n=6)
        hab = Habitation("h1", "Pop Village", "village", 12.848, 77.558,
                         "osm_overpass", population=3000)
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results[0].population_source == "osm_tag"
        assert results[0].population_exposed == 3000

    def test_population_unknown_when_not_in_osm(self):
        grid = _make_grid(n=6)
        hab = Habitation("h1", "NoPopVillage", "village", 12.848, 77.558, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results[0].population_source == "UNKNOWN"
        assert results[0].population_exposed is None

    def test_multiple_habitations(self):
        grid = _make_grid(n=20)
        habs = [
            Habitation(f"h{i}", f"Hab {i}", "village", 12.84 + i * 0.003, 77.55 + i * 0.003, "osm_overpass")
            for i in range(5)
        ]
        ds = HabitationDataset(habitations=habs, source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert len(results) == 5

    def test_water_cells_excluded_from_score(self):
        grid = _make_grid(n=4, risk_classes=["Water"] * 4, scores=[0.0] * 4)
        hab = Habitation("h1", "Water Hab", "village", 12.842, 77.552, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert len(results) == 1
        # Should not crash; hazard score from water cells = 0
        assert results[0].hazard_score == pytest.approx(0.0, abs=0.01)
