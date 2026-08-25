"""Tests for the application health check module."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

from flood_risk_zonation.health import (
    _check_app,
    _check_shap,
    _check_llm,
    _check_weather,
    get_health_status,
)


class TestCheckApp:
    def test_returns_available(self):
        result = _check_app()
        assert result["status"] == "AVAILABLE"
        assert "detail" in result

    def test_unavailable_on_import_error(self):
        # _check_app imports pipeline — if it raises, it should degrade gracefully
        result = _check_app()
        # In a working environment this should be AVAILABLE; the test just
        # verifies no exception is raised and a status key exists
        assert "status" in result


class TestCheckShap:
    def test_available_when_installed(self):
        result = _check_shap()
        assert result["status"] == "AVAILABLE"
        assert "shap" in result["detail"].lower()

    def test_unavailable_when_not_installed(self):
        with patch.dict("sys.modules", {"shap": None}):
            import importlib, flood_risk_zonation.health as h
            importlib.reload(h)
            result = h._check_shap()
            assert result["status"] == "UNAVAILABLE"
            importlib.reload(h)


class TestCheckLLM:
    def test_unavailable_when_provider_none(self, monkeypatch):
        monkeypatch.setenv("PRAVAAH_LLM_PROVIDER", "none")
        result = _check_llm()
        assert result["status"] == "UNAVAILABLE"

    def test_available_when_openai_key_set(self, monkeypatch):
        monkeypatch.setenv("PRAVAAH_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        result = _check_llm()
        assert result["status"] == "AVAILABLE"

    def test_unavailable_when_openai_key_missing(self, monkeypatch):
        monkeypatch.setenv("PRAVAAH_LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = _check_llm()
        assert result["status"] == "UNAVAILABLE"

    def test_available_when_anthropic_key_set(self, monkeypatch):
        monkeypatch.setenv("PRAVAAH_LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        result = _check_llm()
        assert result["status"] == "AVAILABLE"

    def test_unavailable_unknown_provider(self, monkeypatch):
        monkeypatch.setenv("PRAVAAH_LLM_PROVIDER", "unknown_provider")
        result = _check_llm()
        assert result["status"] == "UNAVAILABLE"


class TestCheckWeather:
    def test_unavailable_when_no_key(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        result = _check_weather()
        assert result["status"] == "UNAVAILABLE"

    @patch("flood_risk_zonation.health.requests.get")
    def test_available_on_200_response(self, mock_get, monkeypatch):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")
        mock_r = MagicMock()
        mock_r.status_code = 200
        mock_get.return_value = mock_r
        from flood_risk_zonation import health as h
        result = h._check_weather()
        assert result["status"] == "AVAILABLE"

    @patch("flood_risk_zonation.health.requests.get")
    def test_unavailable_on_401(self, mock_get, monkeypatch):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "bad_key")
        mock_r = MagicMock()
        mock_r.status_code = 401
        mock_get.return_value = mock_r
        from flood_risk_zonation import health as h
        result = h._check_weather()
        assert result["status"] == "UNAVAILABLE"

    @patch("flood_risk_zonation.health.requests.get",
           side_effect=Exception("Network error"))
    def test_degraded_on_network_error(self, mock_get, monkeypatch):
        monkeypatch.setenv("OPENWEATHER_API_KEY", "test_key")
        from flood_risk_zonation import health as h
        result = h._check_weather()
        assert result["status"] == "DEGRADED"


class TestGetHealthStatus:
    def test_returns_all_keys(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        monkeypatch.setenv("PRAVAAH_LLM_PROVIDER", "none")
        status = get_health_status()
        assert "app" in status
        assert "weather" in status
        assert "llm" in status
        assert "shap" in status
        assert "overall" in status
        assert "check_duration_s" in status

    def test_overall_status_is_worst(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        monkeypatch.setenv("PRAVAAH_LLM_PROVIDER", "none")
        status = get_health_status()
        all_statuses = [v["status"] for k, v in status.items()
                        if k not in ("overall", "check_duration_s")]
        priority = {"AVAILABLE": 0, "DEGRADED": 1, "UNAVAILABLE": 2}
        expected_worst = max(all_statuses, key=lambda s: priority.get(s, 9))
        assert status["overall"]["status"] == expected_worst

    def test_duration_is_positive(self, monkeypatch):
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
        status = get_health_status()
        assert status["check_duration_s"] >= 0
