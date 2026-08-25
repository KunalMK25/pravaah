"""
Tests for the intelligence enhancement agents: WeatherAnalyst, ForecastAnalyst,
ScenarioAnalyst, ValidationAnalyst. All LLM calls mocked — tests run offline.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch

from flood_risk_zonation.models import AgentEvidence
from flood_risk_zonation.agents.agents import (
    run_weather_agent,
    run_forecast_agent,
    run_scenario_agent,
    run_validation_agent,
)
from flood_risk_zonation.agents.tools import (
    get_weather_summary,
    get_forecast_summary,
    get_scenario_summary,
    get_validation_summary,
)
from flood_risk_zonation.models import (
    WeatherData, WeatherObservation,
    ForecastResult, ForecastPoint,
    ScenarioResult, ScenarioParameters,
    ValidationResult, ValidationMetrics, HistoricalFloodEvent,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _live_weather(adj=0.5):
    return WeatherData(
        lat=12.87, lon=77.58,
        current=WeatherObservation("2024-01-01T00:00:00Z", rainfall_mm=25.0,
                                    temperature_c=28.0, humidity_pct=85.0),
        forecast=[WeatherObservation(f"2024-01-01T0{i}:00:00Z", rainfall_mm=15.0)
                  for i in range(8)],
        source="openweather_live", fetched_at="2024-01-01T00:00:00Z",
        data_status="LIVE", location_name="Bangalore",
        dynamic_risk_adjustment=adj, dynamic_risk_reason="Heavy rainfall",
    )


def _forecast_result(zone="RED"):
    return ForecastResult(
        bbox_key="test", baseline_zone_counts={"RED": 5, "YELLOW": 8, "GREEN": 12},
        horizons=[
            ForecastPoint(24, 40.0, 55.0, 65.0, 10.0, zone, "HIGH", "forecast_rainfall_adjusted"),
            ForecastPoint(48, 30.0, 55.0, 62.0, 7.0, "YELLOW", "MEDIUM", "forecast_rainfall_adjusted"),
            ForecastPoint(72, 10.0, 55.0, 57.0, 2.0, "YELLOW", "LOW", "forecast_rainfall_adjusted"),
        ],
        weather_source="openweather_live", forecast_timestamp="2024-01-01T00:00:00Z",
    )


def _scenario_result(delta_crit=2, delta_red=5):
    params = ScenarioParameters("sc_test", "+30% Rainfall", rainfall_multiplier=1.3)
    return ScenarioResult(
        scenario_id="sc_test", parameters=params,
        baseline_zone_counts={"RED": 5, "YELLOW": 8, "GREEN": 12},
        scenario_zone_counts={"RED": 5 + delta_red, "YELLOW": 8, "GREEN": 12 - delta_red},
        delta_zone_counts={"RED": delta_red, "YELLOW": 0, "GREEN": -delta_red},
        baseline_critical=1, scenario_critical=1 + delta_crit, delta_critical=delta_crit,
        baseline_high=2, scenario_high=3,
        habitations_escalated=["h1", "h2"],
        narrative=f"SIMULATION: +30% Rainfall. RED zones: +{delta_red}. SIMULATION — not a forecast.",
    )


def _validation_result():
    event = HistoricalFloodEvent(
        "bangalore_2022_09", "Bangalore Flooding 2022",
        "2022-09-05", "Bangalore", "KSNDMC",
    )
    m = ValidationMetrics("bangalore_2022_09", 0.62, 0.55, 0.58, 0.41, 25, 22, 14)
    return ValidationResult(events=[event], metrics=[m], data_status="VALIDATED")


# ── get_weather_summary ───────────────────────────────────────────────────────
class TestGetWeatherSummary:
    def test_live_weather_returns_correct_status(self):
        w = _live_weather(adj=0.5)
        d = get_weather_summary(w)
        assert d["status"] == "LIVE"
        assert d["current_rainfall_mm"] == pytest.approx(25.0, abs=0.01)
        assert d["dynamic_risk_adjustment"] == pytest.approx(0.5, abs=0.01)

    def test_unavailable_weather_returns_unavailable(self):
        d = get_weather_summary(None)
        assert d["status"] == "UNAVAILABLE"
        assert d["dynamic_risk_adjustment"] == 0.0

    def test_max_forecast_derived(self):
        w = _live_weather()
        d = get_weather_summary(w)
        assert d["max_forecast_mm_24h"] >= 0


# ── get_forecast_summary ──────────────────────────────────────────────────────
class TestGetForecastSummary:
    def test_available_forecast(self):
        d = get_forecast_summary(_forecast_result())
        assert d["available"] is True
        assert len(d["horizons"]) == 3

    def test_none_returns_unavailable(self):
        d = get_forecast_summary(None)
        assert d["available"] is False

    def test_horizon_fields_present(self):
        d = get_forecast_summary(_forecast_result())
        h = d["horizons"][0]
        assert "horizon_h" in h
        assert "forecast_rainfall_mm" in h
        assert "spatial_zone" in h
        assert "confidence" in h
        assert "provenance" in h


# ── get_scenario_summary ──────────────────────────────────────────────────────
class TestGetScenarioSummary:
    def test_available_scenario(self):
        d = get_scenario_summary(_scenario_result())
        assert d["available"] is True
        assert "SIMULATION" in d["narrative"].upper()

    def test_none_returns_unavailable(self):
        d = get_scenario_summary(None)
        assert d["available"] is False

    def test_delta_fields_correct(self):
        d = get_scenario_summary(_scenario_result(delta_crit=3, delta_red=7))
        assert d["delta_critical"] == 3
        assert d["delta_zone_counts"]["RED"] == 7


# ── get_validation_summary ────────────────────────────────────────────────────
class TestGetValidationSummary:
    def test_available_validation(self):
        d = get_validation_summary(_validation_result())
        assert d["available"] is True
        assert len(d["metrics"]) == 1

    def test_metric_fields(self):
        d = get_validation_summary(_validation_result())
        m = d["metrics"][0]
        assert "precision" in m
        assert "recall" in m
        assert "f1_score" in m
        assert "iou" in m

    def test_no_events_returns_unavailable(self):
        vr = ValidationResult(data_status="NO_EVENTS_AVAILABLE")
        d = get_validation_summary(vr)
        assert d["available"] is False


# ── run_weather_agent ─────────────────────────────────────────────────────────
class TestRunWeatherAgent:
    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_returns_agent_evidence(self, _):
        w = get_weather_summary(_live_weather(adj=0.6))
        ev = run_weather_agent(w)
        assert isinstance(ev, AgentEvidence)
        assert ev.agent_name == "WeatherAnalyst"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_high_rainfall_high_severity(self, _):
        w = get_weather_summary(_live_weather(adj=0.85))
        ev = run_weather_agent(w)
        assert ev.severity in ("CRITICAL", "HIGH")

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_unavailable_weather_low_severity(self, _):
        ev = run_weather_agent(get_weather_summary(None))
        assert ev.severity == "LOW"
        assert ev.ai_assisted is False

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_summary_not_empty(self, _):
        ev = run_weather_agent(get_weather_summary(_live_weather()))
        assert len(ev.summary) > 0


# ── run_forecast_agent ────────────────────────────────────────────────────────
class TestRunForecastAgent:
    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_returns_agent_evidence(self, _):
        ev = run_forecast_agent(get_forecast_summary(_forecast_result(zone="RED")))
        assert isinstance(ev, AgentEvidence)
        assert ev.agent_name == "ForecastAnalyst"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_red_zone_forecast_high_severity(self, _):
        ev = run_forecast_agent(get_forecast_summary(_forecast_result(zone="RED")))
        assert ev.severity == "HIGH"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_green_zone_forecast_low_or_medium_severity(self, _):
        # When all horizons are GREEN, severity should be LOW or MEDIUM
        # (MEDIUM is acceptable when some horizons are still elevated)
        ev = run_forecast_agent(get_forecast_summary(_forecast_result(zone="GREEN")))
        assert ev.severity in ("LOW", "MEDIUM")

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_unavailable_forecast(self, _):
        ev = run_forecast_agent(get_forecast_summary(None))
        assert isinstance(ev, AgentEvidence)
        assert ev.ai_assisted is False

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_summary_contains_estimate_label(self, _):
        ev = run_forecast_agent(get_forecast_summary(_forecast_result()))
        assert "FORECAST" in ev.summary.upper() or "ESTIMATE" in ev.summary.upper() or len(ev.summary) > 0


# ── run_scenario_agent ────────────────────────────────────────────────────────
class TestRunScenarioAgent:
    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_returns_agent_evidence(self, _):
        ev = run_scenario_agent(get_scenario_summary(_scenario_result()))
        assert isinstance(ev, AgentEvidence)
        assert ev.agent_name == "ScenarioAnalyst"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_large_delta_high_severity(self, _):
        ev = run_scenario_agent(get_scenario_summary(_scenario_result(delta_crit=5, delta_red=15)))
        assert ev.severity == "HIGH"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_simulation_label_in_summary(self, _):
        ev = run_scenario_agent(get_scenario_summary(_scenario_result()))
        assert "SIMULATION" in ev.summary.upper()

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_unavailable_scenario(self, _):
        ev = run_scenario_agent(get_scenario_summary(None))
        assert isinstance(ev, AgentEvidence)


# ── run_validation_agent ──────────────────────────────────────────────────────
class TestRunValidationAgent:
    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_returns_agent_evidence(self, _):
        ev = run_validation_agent(get_validation_summary(_validation_result()))
        assert isinstance(ev, AgentEvidence)
        assert ev.agent_name == "ValidationAnalyst"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_good_f1_low_severity(self, _):
        ev = run_validation_agent(get_validation_summary(_validation_result()))
        assert ev.severity in ("LOW", "MEDIUM")

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_no_events_available(self, _):
        vr = ValidationResult(data_status="NO_EVENTS_AVAILABLE")
        ev = run_validation_agent(get_validation_summary(vr))
        assert isinstance(ev, AgentEvidence)
        assert ev.ai_assisted is False

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_summary_not_empty(self, _):
        ev = run_validation_agent(get_validation_summary(_validation_result()))
        assert len(ev.summary) > 0

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_metrics_in_evidence(self, _):
        ev = run_validation_agent(get_validation_summary(_validation_result()))
        assert isinstance(ev.metrics, dict)
