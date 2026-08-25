"""
Tests for the live weather client.
All network calls are mocked — tests run offline.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from flood_risk_zonation.models import WeatherData, WeatherObservation
from flood_risk_zonation.weather.client import (
    fetch_weather,
    _parse_current,
    _parse_forecast_list,
    _compute_dynamic_adjustment,
    _HEAVY_RAIN_MM_3H,
)


# ── Fixture data ──────────────────────────────────────────────────────────────

def _owm_current_response(rain_3h=5.0, temp=28.0, humidity=80.0):
    return {
        "name": "Bangalore",
        "dt": int(time.time()),
        "main": {"temp": temp, "humidity": humidity},
        "wind": {"speed": 3.5},
        "rain": {"3h": rain_3h},
        "weather": [{"description": "light rain"}],
    }

def _owm_forecast_response(rain_per_step=2.0, n_steps=8):
    return {
        "list": [
            {
                "dt": int(time.time()) + i * 10800,
                "main": {"temp": 27.0, "humidity": 75.0},
                "wind": {"speed": 2.0},
                "rain": {"3h": rain_per_step},
                "weather": [{"description": "rain"}],
            }
            for i in range(n_steps)
        ]
    }


class TestParseCurrentWeather:
    def test_basic_parse(self):
        raw = _owm_current_response(rain_3h=12.0)
        obs = _parse_current(raw)
        assert isinstance(obs, WeatherObservation)
        assert obs.rainfall_mm == pytest.approx(12.0, abs=0.01)
        assert obs.temperature_c == pytest.approx(28.0, abs=0.1)
        assert obs.humidity_pct == 80.0
        assert obs.description == "light rain"

    def test_no_rain_field(self):
        raw = {"name": "X", "dt": int(time.time()), "main": {"temp": 30.0, "humidity": 60.0},
               "wind": {"speed": 1.0}, "weather": [{"description": "clear"}]}
        obs = _parse_current(raw)
        assert obs.rainfall_mm == 0.0

    def test_uses_1h_if_no_3h(self):
        raw = _owm_current_response()
        raw["rain"] = {"1h": 8.0}
        obs = _parse_current(raw)
        assert obs.rainfall_mm == pytest.approx(24.0, abs=0.01)  # 1h × 3


class TestParseForecastList:
    def test_basic_parse(self):
        raw = _owm_forecast_response(rain_per_step=3.0, n_steps=5)
        obs_list = _parse_forecast_list(raw["list"])
        assert len(obs_list) == 5
        assert all(o.rainfall_mm == pytest.approx(3.0, abs=0.01) for o in obs_list)

    def test_empty_list(self):
        assert _parse_forecast_list([]) == []

    def test_caps_at_24_steps(self):
        raw = _owm_forecast_response(n_steps=30)
        obs_list = _parse_forecast_list(raw["list"])
        assert len(obs_list) == 24  # capped at 24 steps


class TestDynamicAdjustment:
    def test_no_rain_zero_adjustment(self):
        curr = WeatherObservation("2024-01-01T00:00:00Z", rainfall_mm=0.0)
        adj, reason = _compute_dynamic_adjustment(curr, [])
        assert adj == 0.0

    def test_heavy_rain_max_adjustment(self):
        curr = WeatherObservation("2024-01-01T00:00:00Z", rainfall_mm=_HEAVY_RAIN_MM_3H)
        adj, reason = _compute_dynamic_adjustment(curr, [])
        assert adj == pytest.approx(1.0, abs=0.01)

    def test_adjustment_capped_at_1(self):
        curr = WeatherObservation("2024-01-01T00:00:00Z", rainfall_mm=_HEAVY_RAIN_MM_3H * 3)
        adj, _ = _compute_dynamic_adjustment(curr, [])
        assert adj <= 1.0

    def test_forecast_drives_adjustment_when_higher(self):
        curr = WeatherObservation("2024-01-01T00:00:00Z", rainfall_mm=5.0)
        fc = [WeatherObservation("t", rainfall_mm=40.0)]
        adj_with_fc, _ = _compute_dynamic_adjustment(curr, fc)
        adj_without_fc, _ = _compute_dynamic_adjustment(curr, [])
        assert adj_with_fc >= adj_without_fc

    def test_unknown_current_rainfall_treated_as_zero(self):
        curr = WeatherObservation("2024-01-01T00:00:00Z", rainfall_mm=-1.0)
        adj, _ = _compute_dynamic_adjustment(curr, [])
        assert adj == 0.0


class TestFetchWeather:
    def test_unavailable_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        result = fetch_weather(12.87, 77.58)
        assert result.data_status == "UNAVAILABLE"
        assert isinstance(result, WeatherData)

    @patch("flood_risk_zonation.weather.client.requests.get")
    def test_live_fetch_success(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key_123")
        monkeypatch.setenv("PRAVAAH_WEATHER_CACHE_DIR", str(tmp_path))

        mock_curr = MagicMock()
        mock_curr.status_code = 200
        mock_curr.json.return_value = _owm_current_response(rain_3h=15.0)

        mock_fc = MagicMock()
        mock_fc.status_code = 200
        mock_fc.json.return_value = _owm_forecast_response(rain_per_step=5.0)

        mock_get.side_effect = [mock_curr, mock_fc]

        result = fetch_weather(12.87, 77.58)
        assert result.data_status == "LIVE"
        assert result.current is not None
        assert result.current.rainfall_mm == pytest.approx(15.0, abs=0.01)
        assert result.dynamic_risk_adjustment > 0

    @patch("flood_risk_zonation.weather.client.requests.get")
    def test_api_error_returns_unavailable(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")
        monkeypatch.setenv("PRAVAAH_WEATHER_CACHE_DIR", str(tmp_path))

        mock_r = MagicMock()
        mock_r.status_code = 401
        mock_get.return_value = mock_r

        result = fetch_weather(12.87, 77.58)
        assert result.data_status == "UNAVAILABLE"

    @patch("flood_risk_zonation.weather.client.requests.get")
    def test_cache_hit_returns_cached(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")
        monkeypatch.setenv("PRAVAAH_WEATHER_CACHE_DIR", str(tmp_path))

        # Pre-populate cache
        cache_file = tmp_path / "weather_12.8700_77.5800.json"
        cache_data = {
            "_fetched_at": time.time(),  # fresh
            "current": _owm_current_response(rain_3h=8.0),
            "forecast": _owm_forecast_response()["list"],
            "location_name": "Bangalore",
        }
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = fetch_weather(12.87, 77.58)
        assert result.data_status == "CACHED"
        mock_get.assert_not_called()

    @patch("flood_risk_zonation.weather.client.requests.get", side_effect=Exception("Network down"))
    def test_network_exception_returns_unavailable(self, mock_get, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")
        monkeypatch.setenv("PRAVAAH_WEATHER_CACHE_DIR", str(tmp_path))
        result = fetch_weather(12.87, 77.58)
        assert result.data_status == "UNAVAILABLE"
        assert isinstance(result.dynamic_risk_adjustment, float)

    def test_weather_data_always_valid_object(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        monkeypatch.setenv("PRAVAAH_WEATHER_CACHE_DIR", str(tmp_path))
        result = fetch_weather(0.0, 0.0)
        assert isinstance(result, WeatherData)
        assert result.lat == 0.0
        assert result.lon == 0.0
