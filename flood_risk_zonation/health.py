"""
PRAVAAH-AI — Application Health / Status Check.

Reports the availability of each optional service without blocking startup.
Used by the UI status panel and deployment health checks.

Status values:
  AVAILABLE   — service responded or is configured
  DEGRADED    — partial functionality (cached/fallback)
  UNAVAILABLE — not configured or unreachable
"""
from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


def _check_weather() -> dict:
    key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not key:
        return {"status": "UNAVAILABLE", "detail": "OPENWEATHER_API_KEY not set"}
    try:
        import requests
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": 12.87, "lon": 77.58, "appid": key},
            timeout=5,
        )
        if r.status_code == 200:
            return {"status": "AVAILABLE", "detail": "OpenWeatherMap API reachable"}
        elif r.status_code == 401:
            return {"status": "UNAVAILABLE", "detail": "Invalid API key"}
        else:
            return {"status": "DEGRADED", "detail": f"HTTP {r.status_code}"}
    except Exception as exc:
        return {"status": "DEGRADED", "detail": f"Network error: {exc}"}


def _check_osm() -> dict:
    try:
        import requests
        r = requests.get(
            "https://overpass-api.de/api/interpreter",
            data="[out:json][timeout:5];node(12.84,77.55,12.85,77.56)[place=village];out 1;",
            timeout=8,
        )
        if r.status_code == 200:
            return {"status": "AVAILABLE", "detail": "OSM Overpass reachable"}
        return {"status": "DEGRADED", "detail": f"HTTP {r.status_code}"}
    except Exception as exc:
        return {"status": "DEGRADED", "detail": f"Network error: {exc}"}


def _check_llm() -> dict:
    provider = os.environ.get("PRAVAAH_LLM_PROVIDER", "none").lower()
    if provider == "none":
        return {"status": "UNAVAILABLE", "detail": "LLM disabled (rule-based mode active)"}
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        return (
            {"status": "AVAILABLE", "detail": "OpenAI key configured"}
            if key else
            {"status": "UNAVAILABLE", "detail": "OPENAI_API_KEY not set"}
        )
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        return (
            {"status": "AVAILABLE", "detail": "Anthropic key configured"}
            if key else
            {"status": "UNAVAILABLE", "detail": "ANTHROPIC_API_KEY not set"}
        )
    return {"status": "UNAVAILABLE", "detail": f"Unknown provider: {provider}"}


def _check_shap() -> dict:
    try:
        import shap
        return {"status": "AVAILABLE", "detail": f"SHAP {shap.__version__}"}
    except ImportError:
        return {"status": "UNAVAILABLE", "detail": "shap not installed"}


def _check_app() -> dict:
    try:
        from flood_risk_zonation.pipeline import FloodRiskPipeline
        from flood_risk_zonation.config import PipelineConfig
        return {"status": "AVAILABLE", "detail": "Core pipeline importable"}
    except Exception as exc:
        return {"status": "UNAVAILABLE", "detail": str(exc)}


def get_health_status() -> dict:
    """
    Return a structured health dict for all PRAVAAH services.

    Returns
    -------
    dict with keys: app, weather, osm, llm, shap
    Each value is {"status": str, "detail": str}.
    Overall "status" is the worst of all component statuses.
    """
    t0 = time.time()
    checks = {
        "app":     _check_app(),
        "weather": _check_weather(),
        "osm":     _check_osm(),
        "llm":     _check_llm(),
        "shap":    _check_shap(),
    }
    # Overall status: AVAILABLE > DEGRADED > UNAVAILABLE
    priority = {"AVAILABLE": 0, "DEGRADED": 1, "UNAVAILABLE": 2}
    worst = max(checks.values(), key=lambda c: priority.get(c["status"], 9))
    checks["overall"] = worst
    checks["check_duration_s"] = round(time.time() - t0, 2)
    return checks
