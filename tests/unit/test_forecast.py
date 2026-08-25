"""Tests for the short-term flood-risk forecast engine."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from flood_risk_zonation.models import (
    WeatherData, WeatherObservation, ForecastResult, ForecastPoint,
)
from flood_risk_zonation.forecast.engine import (
    generate_forecast,
    _rainfall_boost,
    _project_zone,
    _sum_forecast_mm,
    _MAX_BOOST_POINTS,
    _BOOST_THRESHOLD_MM,
)
from flood_risk_zonation.spatial_zones.classifier import ZONE_RED, ZONE_YELLOW, ZONE_GREEN


def _make_hazard_result(n_high=3, n_low=7, low_t=33.0, med_t=66.0):
    """Minimal FloodRiskResult mock."""
    import pandas as pd
    n = n_high + n_low
    rows = []
    for i in range(n):
        rc = "High" if i < n_high else "Low"
        score = 80.0 if rc == "High" else 20.0
        lat = 12.84 + i * 0.005
        lon = 77.55 + i * 0.005
        rows.append({
            "cell_id": f"c{i}", "risk_class": rc, "risk_score": score,
            "centroid_lat": lat, "centroid_lon": lon,
            "rainfall_mean_mm": 1200.0, "rainfall_max_24h_mm": 80.0,
            "geometry": box(lon-0.002, lat-0.002, lon+0.002, lat+0.002),
        })
    grid = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    from flood_risk_zonation.spatial_zones.classifier import classify_spatial_zones
    grid = classify_spatial_zones(grid)

    mock_config = MagicMock()
    mock_config.low_threshold = low_t
    mock_config.medium_threshold = med_t

    mock_bbox = MagicMock()
    mock_bbox.min_lon = 77.55; mock_bbox.min_lat = 12.84
    mock_bbox.max_lon = 77.62; mock_bbox.max_lat = 12.91

    mock_analysis = MagicMock()
    mock_analysis.model = MagicMock()
    mock_analysis.model.predict_proba = MagicMock(
        return_value=np.column_stack([np.zeros(n), np.ones(n)])
    )

    hr = MagicMock()
    hr.scored_grid = grid
    hr.config = mock_config
    hr.bounding_box = mock_bbox
    hr.analysis_result = mock_analysis
    return hr


def _make_weather(status="LIVE", curr_mm=10.0, fc_mm=20.0, n_fc=8):
    current = WeatherObservation("2024-01-01T00:00:00Z", rainfall_mm=curr_mm,
                                  temperature_c=28.0, humidity_pct=75.0)
    forecast = [
        WeatherObservation(f"2024-01-01T0{i}:00:00Z", rainfall_mm=fc_mm)
        for i in range(n_fc)
    ]
    return WeatherData(
        lat=12.87, lon=77.58, current=current, forecast=forecast,
        source="openweather_live", fetched_at="2024-01-01T00:00:00Z",
        data_status=status, location_name="Bangalore",
        dynamic_risk_adjustment=min(1.0, max(curr_mm, fc_mm) / _BOOST_THRESHOLD_MM),
        dynamic_risk_reason="Test weather",
    )


class TestRainfallBoost:
    def test_zero_rainfall_zero_boost(self):
        assert _rainfall_boost(0.0) == 0.0

    def test_heavy_rainfall_max_boost(self):
        assert _rainfall_boost(_BOOST_THRESHOLD_MM) == pytest.approx(_MAX_BOOST_POINTS, abs=0.01)

    def test_boost_capped(self):
        assert _rainfall_boost(_BOOST_THRESHOLD_MM * 10) == pytest.approx(_MAX_BOOST_POINTS, abs=0.01)

    def test_partial_rainfall_proportional(self):
        boost = _rainfall_boost(_BOOST_THRESHOLD_MM / 2)
        assert boost == pytest.approx(_MAX_BOOST_POINTS / 2, abs=0.1)

    def test_negative_rainfall_zero_boost(self):
        assert _rainfall_boost(-5.0) == 0.0


class TestProjectZone:
    def test_high_score_is_red(self):
        assert _project_zone(90.0, 33.0, 66.0) == ZONE_RED

    def test_medium_score_is_yellow(self):
        assert _project_zone(50.0, 33.0, 66.0) == ZONE_YELLOW

    def test_low_score_is_green(self):
        assert _project_zone(10.0, 33.0, 66.0) == ZONE_GREEN

    def test_at_medium_threshold_is_red(self):
        assert _project_zone(66.1, 33.0, 66.0) == ZONE_RED


class TestSumForecastMm:
    def test_sums_first_8_steps_for_24h(self):
        weather = _make_weather(fc_mm=5.0, n_fc=8)
        total = _sum_forecast_mm(weather, 24)
        assert total == pytest.approx(40.0, abs=0.1)  # 8 steps × 5.0mm

    def test_zero_when_no_forecast(self):
        weather = WeatherData(lat=0, lon=0, data_status="UNAVAILABLE")
        assert _sum_forecast_mm(weather, 24) == 0.0

    def test_negative_mm_excluded(self):
        weather = _make_weather(fc_mm=-1.0, n_fc=4)
        assert _sum_forecast_mm(weather, 24) == 0.0


class TestGenerateForecast:
    def test_returns_forecast_result(self):
        hr = _make_hazard_result()
        weather = _make_weather(curr_mm=20.0, fc_mm=25.0)
        result = generate_forecast(hr, weather)
        assert isinstance(result, ForecastResult)

    def test_returns_three_default_horizons(self):
        hr = _make_hazard_result()
        weather = _make_weather()
        result = generate_forecast(hr, weather)
        assert len(result.horizons) == 3
        assert {h.horizon_h for h in result.horizons} == {24, 48, 72}

    def test_custom_horizons(self):
        hr = _make_hazard_result()
        weather = _make_weather()
        result = generate_forecast(hr, weather, horizons=[12, 24])
        assert {h.horizon_h for h in result.horizons} == {12, 24}

    def test_adjusted_score_gte_baseline_with_rain(self):
        hr = _make_hazard_result()
        weather = _make_weather(curr_mm=30.0, fc_mm=40.0)
        result = generate_forecast(hr, weather)
        for h in result.horizons:
            if h.forecast_rainfall_mm > 0:
                assert h.adjusted_risk_score >= h.baseline_risk_score

    def test_no_boost_when_weather_unavailable(self):
        hr = _make_hazard_result()
        weather = WeatherData(lat=0, lon=0, data_status="UNAVAILABLE")
        result = generate_forecast(hr, weather)
        for h in result.horizons:
            assert h.risk_change == pytest.approx(0.0, abs=0.01)
            assert h.provenance == "baseline_only"

    def test_provenance_labelled_forecast(self):
        hr = _make_hazard_result()
        weather = _make_weather()
        result = generate_forecast(hr, weather)
        assert "ESTIMATE" in result.methodology

    def test_get_horizon_helper(self):
        hr = _make_hazard_result()
        weather = _make_weather()
        result = generate_forecast(hr, weather)
        h24 = result.get_horizon(24)
        assert h24 is not None
        assert h24.horizon_h == 24

    def test_get_horizon_missing_returns_none(self):
        hr = _make_hazard_result()
        weather = _make_weather()
        result = generate_forecast(hr, weather)
        assert result.get_horizon(999) is None

    def test_horizon_confidence_live(self):
        hr = _make_hazard_result()
        weather = _make_weather(status="LIVE")
        result = generate_forecast(hr, weather)
        h24 = result.get_horizon(24)
        assert h24.confidence == "HIGH"

    def test_horizon_confidence_unavailable_is_low(self):
        hr = _make_hazard_result()
        weather = WeatherData(lat=0, lon=0, data_status="UNAVAILABLE")
        result = generate_forecast(hr, weather)
        for h in result.horizons:
            assert h.confidence == "LOW"
