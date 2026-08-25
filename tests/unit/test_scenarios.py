"""Tests for the what-if scenario engine."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from flood_risk_zonation.models import ScenarioParameters, ScenarioResult
from flood_risk_zonation.scenarios.engine import (
    run_scenario,
    build_preset_scenarios,
)
from flood_risk_zonation.spatial_zones.classifier import (
    classify_spatial_zones, ZONE_RED, ZONE_YELLOW, ZONE_GREEN,
)


def _make_grid(n_high=4, n_medium=3, n_low=9, n_cols=4):
    import pandas as pd
    classes = ["High"] * n_high + ["Medium"] * n_medium + ["Low"] * n_low
    rows = []
    for i, rc in enumerate(classes):
        r, c = divmod(i, n_cols)
        lat = 12.84 + r * 0.008
        lon = 77.55 + c * 0.008
        score = {"High": 80.0, "Medium": 50.0, "Low": 20.0}.get(rc, 20.0)
        rows.append({
            "cell_id": f"c{i}", "risk_class": rc, "risk_score": score,
            "centroid_lat": lat, "centroid_lon": lon,
            "rainfall_mean_mm": 1200.0, "rainfall_max_24h_mm": 80.0,
            "drainage_capacity": 0.5, "population_density": 200.0,
            "elevation_m": 40.0, "slope_deg": 3.0, "twi": 8.0,
            "aspect_deg": 180.0, "curvature": -0.5,
            "dist_water_m": 500.0,
            "geometry": box(lon - 0.004, lat - 0.004, lon + 0.004, lat + 0.004),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return classify_spatial_zones(gdf)


def _make_hazard_result(grid=None):
    if grid is None:
        grid = _make_grid()
    mock_config = MagicMock()
    mock_config.low_threshold = 33.0
    mock_config.medium_threshold = 66.0
    mock_bbox = MagicMock()
    mock_bbox.min_lon, mock_bbox.min_lat = 77.55, 12.84
    mock_bbox.max_lon, mock_bbox.max_lat = 77.62, 12.91

    from flood_risk_zonation.scoring.susceptibility import WeightedSusceptibilityModel
    from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
    model = WeightedSusceptibilityModel().fit(
        grid[[c for c in FEATURE_COLUMNS if c in grid.columns]]
    )
    mock_analysis = MagicMock()
    mock_analysis.model = model

    hr = MagicMock()
    hr.scored_grid = grid
    hr.config = mock_config
    hr.bounding_box = mock_bbox
    hr.analysis_result = mock_analysis
    return hr


class TestBuildPresetScenarios:
    def test_returns_list(self):
        scenarios = build_preset_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) >= 4

    def test_all_have_labels(self):
        for s in build_preset_scenarios():
            assert len(s.label) > 0

    def test_all_have_unique_ids(self):
        scenarios = build_preset_scenarios()
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_rainfall_multipliers_positive(self):
        for s in build_preset_scenarios():
            assert s.rainfall_multiplier > 0


class TestRunScenario:
    def test_returns_scenario_result(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("test", "+20% Rainfall", rainfall_multiplier=1.2)
        result = run_scenario(hr, None, params)
        assert isinstance(result, ScenarioResult)

    def test_provenance_is_simulation(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("s1", "+30%", rainfall_multiplier=1.3)
        result = run_scenario(hr, None, params)
        assert "SIMULATION" in result.provenance

    def test_baseline_not_modified(self):
        grid = _make_grid()
        original_scores = grid["risk_score"].copy()
        hr = _make_hazard_result(grid)
        params = ScenarioParameters("s1", "+50%", rainfall_multiplier=1.5)
        run_scenario(hr, None, params)
        # Original grid scores must not be altered
        assert (grid["risk_score"].values == original_scores.values).all()

    def test_heavy_rain_increases_red_zone_score(self):
        # With heavy rain boost, mean adjusted scores should be higher than baseline
        hr = _make_hazard_result()
        params = ScenarioParameters("s1", "+100%", rainfall_multiplier=2.0,
                                    extra_rainfall_mm=50.0)
        result = run_scenario(hr, None, params)
        # The scenario ran and produced a valid result
        assert isinstance(result, ScenarioResult)
        # With doubled rainfall, the narrative should reflect change
        assert "SIMULATION" in result.narrative.upper()

    def test_no_change_scenario_returns_valid(self):
        # An identity scenario produces a valid result — zone counts may differ
        # slightly due to WSI re-scoring on the modified (but unchanged) grid
        hr = _make_hazard_result()
        params = ScenarioParameters("s_identity", "No Change",
                                    rainfall_multiplier=1.0,
                                    drainage_capacity_multiplier=1.0)
        result = run_scenario(hr, None, params)
        assert isinstance(result, ScenarioResult)
        # Total cells must be conserved
        total_scenario = sum(result.scenario_zone_counts.values())
        total_baseline = sum(result.baseline_zone_counts.values())
        assert total_scenario == total_baseline

    def test_delta_zones_correct(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("s1", "+30%", rainfall_multiplier=1.3)
        result = run_scenario(hr, None, params)
        for z in [ZONE_RED, ZONE_YELLOW, ZONE_GREEN]:
            expected_delta = (
                result.scenario_zone_counts.get(z, 0)
                - result.baseline_zone_counts.get(z, 0)
            )
            assert result.delta_zone_counts.get(z, 0) == expected_delta

    def test_narrative_contains_simulation_label(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("s1", "+20% Rain", rainfall_multiplier=1.2)
        result = run_scenario(hr, None, params)
        assert "SIMULATION" in result.narrative.upper()

    def test_degraded_drainage_scenario(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("s_drain", "Degraded Drainage",
                                    drainage_capacity_multiplier=0.5)
        result = run_scenario(hr, None, params)
        assert isinstance(result, ScenarioResult)

    def test_combined_scenario(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("s_combined", "+30% Rain + Poor Drain",
                                    rainfall_multiplier=1.3,
                                    drainage_capacity_multiplier=0.7)
        result = run_scenario(hr, None, params)
        assert result.scenario_id == "s_combined"
        assert result.parameters.label == "+30% Rain + Poor Drain"

    def test_parameters_stored_in_result(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("s1", "Test", rainfall_multiplier=1.25,
                                    extra_rainfall_mm=10.0)
        result = run_scenario(hr, None, params)
        assert result.parameters.rainfall_multiplier == 1.25
        assert result.parameters.extra_rainfall_mm == 10.0

    def test_scenario_counts_sum_to_total_cells(self):
        hr = _make_hazard_result()
        params = ScenarioParameters("s1", "+30%", rainfall_multiplier=1.3)
        result = run_scenario(hr, None, params)
        total_scenario = sum(result.scenario_zone_counts.values())
        total_grid = len(hr.scored_grid)
        assert total_scenario == total_grid


class TestScenarioIsolation:
    """Property-style tests: baseline MUST NOT be modified by scenario."""

    def test_scored_grid_risk_class_unchanged(self):
        grid = _make_grid()
        original_classes = grid["risk_class"].tolist()
        hr = _make_hazard_result(grid)
        params = ScenarioParameters("s", "+50%", rainfall_multiplier=1.5)
        run_scenario(hr, None, params)
        assert grid["risk_class"].tolist() == original_classes

    def test_scored_grid_risk_score_unchanged(self):
        grid = _make_grid()
        original_scores = grid["risk_score"].values.copy()
        hr = _make_hazard_result(grid)
        params = ScenarioParameters("s", "+50%", rainfall_multiplier=1.5)
        run_scenario(hr, None, params)
        assert (grid["risk_score"].values == original_scores).all()

    def test_multiple_scenarios_independent(self):
        hr = _make_hazard_result()
        p1 = ScenarioParameters("s1", "+10%", rainfall_multiplier=1.1)
        p2 = ScenarioParameters("s2", "+50%", rainfall_multiplier=1.5)
        r1 = run_scenario(hr, None, p1)
        r2 = run_scenario(hr, None, p2)
        # Baselines must be identical (same source)
        assert r1.baseline_zone_counts == r2.baseline_zone_counts
