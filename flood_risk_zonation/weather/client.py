"""
PRAVAAH — Live Weather Client.

Fetches current conditions and short-term forecast from OpenWeatherMap.
Implements the same resilience architecture as water_bodies.py:
  cache → live API → fallback (UNAVAILABLE).

Provider: OpenWeatherMap One Call API 3.0 (free tier: 1000 calls/day).
Configuration (environment variables — never hard-coded):
  OPENWEATHER_API_KEY   — required for live data
  PRAVAAH_WEATHER_CACHE_DIR — optional, default data/cache/weather

Data provenance labels (always shown in UI):
  LIVE      — freshly fetched from API
  CACHED    — from local disk cache (< CACHE_TTL_SECONDS old)
  FALLBACK  — API unavailable; synthetic/zero values used
  UNAVAILABLE — no key configured

Dynamic hazard adjustment methodology (transparent):
  1. Fetch current 3h precipitation (mm) from API.
  2. Normalise against a reference heavy-rain threshold (50 mm/3h).
  3. Compute adjustment = min(1.0, precipitation / HEAVY_RAIN_THRESHOLD).
  4. This multiplier is exposed as weather.dynamic_risk_adjustment in [0, 1].
  5. The existing ML hazard model is NEVER retrained; the adjustment is an
     additive signal passed to the agentic layer for interpretation.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from flood_risk_zonation.models import WeatherData, WeatherObservation

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_API_KEY_ENV        = "OPENWEATHER_API_KEY"
_CACHE_DIR_ENV      = "PRAVAAH_WEATHER_CACHE_DIR"
_DEFAULT_CACHE_DIR  = Path("data/cache/weather")
_CACHE_TTL_SECONDS  = 900          # 15 minutes
_REQUEST_TIMEOUT    = 10           # seconds
_HEAVY_RAIN_MM_3H   = 50.0         # reference threshold for normalising risk adjustment
_OWM_CURRENT_URL    = "https://api.openweathermap.org/data/2.5/weather"
_OWM_FORECAST_URL   = "https://api.openweathermap.org/data/2.5/forecast"


def _api_key() -> str | None:
    return os.environ.get(_API_KEY_ENV, "").strip() or None


def _cache_dir() -> Path:
    d = Path(os.environ.get(_CACHE_DIR_ENV, str(_DEFAULT_CACHE_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(lat: float, lon: float) -> str:
    return f"weather_{lat:.4f}_{lon:.4f}"


def _load_cache(lat: float, lon: float) -> dict | None:
    path = _cache_dir() / f"{_cache_key(lat, lon)}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - data.get("_fetched_at", 0)
        if age < _CACHE_TTL_SECONDS:
            return data
        logger.debug("Weather cache expired (age %.0fs)", age)
    except Exception as exc:
        logger.debug("Weather cache read error: %s", exc)
    return None


def _save_cache(lat: float, lon: float, data: dict) -> None:
    path = _cache_dir() / f"{_cache_key(lat, lon)}.json"
    try:
        data["_fetched_at"] = time.time()
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception as exc:
        logger.debug("Weather cache write error: %s", exc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_current(raw: dict) -> WeatherObservation:
    """Parse OpenWeatherMap /weather JSON into WeatherObservation."""
    rain_3h = raw.get("rain", {}).get("3h", 0.0) or 0.0
    rain_1h = raw.get("rain", {}).get("1h", 0.0) or 0.0
    rainfall = max(rain_3h, rain_1h * 3)   # prefer 3h if available

    main = raw.get("main", {})
    wind = raw.get("wind", {})
    desc = raw.get("weather", [{}])[0].get("description", "")
    dt_unix = raw.get("dt", 0)
    ts = datetime.fromtimestamp(dt_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt_unix else _now_iso()

    return WeatherObservation(
        timestamp=ts,
        rainfall_mm=round(rainfall, 2),
        temperature_c=round(main.get("temp", -999.0), 1),
        humidity_pct=round(main.get("humidity", -1.0), 1),
        wind_speed_ms=round(wind.get("speed", -1.0), 2),
        description=desc,
    )


def _parse_forecast_list(raw_list: list) -> list[WeatherObservation]:
    """Parse OWM /forecast list (3-hour steps) into observations."""
    result = []
    for item in raw_list[:24]:   # up to 72 h (24 × 3h steps)
        rain = item.get("rain", {}).get("3h", 0.0) or 0.0
        main = item.get("main", {})
        wind = item.get("wind", {})
        desc = item.get("weather", [{}])[0].get("description", "")
        dt_unix = item.get("dt", 0)
        ts = datetime.fromtimestamp(dt_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if dt_unix else ""
        result.append(WeatherObservation(
            timestamp=ts,
            rainfall_mm=round(rain, 2),
            temperature_c=round(main.get("temp", -999.0), 1),
            humidity_pct=round(main.get("humidity", -1.0), 1),
            wind_speed_ms=round(wind.get("speed", -1.0), 2),
            description=desc,
        ))
    return result


def _compute_dynamic_adjustment(current: WeatherObservation, forecast: list[WeatherObservation]) -> tuple[float, str]:
    """
    Compute a normalised [0, 1] dynamic risk adjustment from precipitation.

    Uses current rainfall and max forecast rainfall.
    Returns (adjustment, reason_string).
    """
    current_mm = max(0.0, current.rainfall_mm) if current.rainfall_mm >= 0 else 0.0
    # Max 3h rainfall in next 24h forecast
    fc_mm = max((o.rainfall_mm for o in forecast[:8] if o.rainfall_mm >= 0), default=0.0)
    peak_mm = max(current_mm, fc_mm)
    adjustment = round(min(1.0, peak_mm / _HEAVY_RAIN_MM_3H), 4)

    if peak_mm <= 0:
        reason = "No significant rainfall — no dynamic hazard adjustment."
    elif peak_mm < 10:
        reason = f"Light rainfall ({peak_mm:.1f} mm) — minimal dynamic adjustment."
    elif peak_mm < 25:
        reason = f"Moderate rainfall ({peak_mm:.1f} mm) — low dynamic adjustment."
    elif peak_mm < 50:
        reason = f"Heavy rainfall ({peak_mm:.1f} mm) — elevated dynamic adjustment."
    else:
        reason = f"Very heavy/extreme rainfall ({peak_mm:.1f} mm) — maximum dynamic adjustment."

    return adjustment, reason


def _unavailable_result(lat: float, lon: float, reason: str) -> WeatherData:
    return WeatherData(
        lat=lat, lon=lon, current=None, forecast=[],
        source="unavailable", fetched_at=_now_iso(),
        data_status="UNAVAILABLE", location_name="",
        dynamic_risk_adjustment=0.0, dynamic_risk_reason=reason,
    )


def fetch_weather(lat: float, lon: float) -> WeatherData:
    """
    Fetch current conditions and short-term forecast for a lat/lon point.

    Resolution order:
      1. Local cache (< 15 min old)
      2. Live OpenWeatherMap API (if OPENWEATHER_API_KEY set)
      3. UNAVAILABLE (application continues)

    Parameters
    ----------
    lat, lon : float
        Geographic coordinates.

    Returns
    -------
    WeatherData
        Always returns a valid WeatherData object.
        Check .data_status for LIVE / CACHED / UNAVAILABLE.
    """
    key = _api_key()
    if not key:
        logger.info("OPENWEATHER_API_KEY not configured — weather unavailable.")
        return _unavailable_result(lat, lon, "OPENWEATHER_API_KEY not configured.")

    # Try cache first
    cached = _load_cache(lat, lon)
    if cached:
        try:
            current_raw = cached.get("current")
            forecast_raw = cached.get("forecast", [])
            current = _parse_current(current_raw) if current_raw else None
            forecast = _parse_forecast_list(forecast_raw)
            adj, reason = _compute_dynamic_adjustment(current or WeatherObservation(""), forecast)
            return WeatherData(
                lat=lat, lon=lon, current=current, forecast=forecast,
                source="openweather_cache",
                fetched_at=datetime.fromtimestamp(
                    cached.get("_fetched_at", 0), tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                data_status="CACHED",
                location_name=cached.get("location_name", ""),
                dynamic_risk_adjustment=adj,
                dynamic_risk_reason=reason,
            )
        except Exception as exc:
            logger.debug("Cache parse error: %s", exc)

    # Live API
    params = {"lat": lat, "lon": lon, "appid": key, "units": "metric"}
    try:
        r_curr = requests.get(_OWM_CURRENT_URL, params=params, timeout=_REQUEST_TIMEOUT)
        r_fc   = requests.get(_OWM_FORECAST_URL, params=params, timeout=_REQUEST_TIMEOUT)

        if r_curr.status_code != 200 or r_fc.status_code != 200:
            logger.warning("OWM API error: current=%d forecast=%d",
                           r_curr.status_code, r_fc.status_code)
            return _unavailable_result(lat, lon,
                f"API error (current={r_curr.status_code}, forecast={r_fc.status_code})")

        curr_json = r_curr.json()
        fc_json   = r_fc.json()

        current = _parse_current(curr_json)
        forecast_list = fc_json.get("list", [])
        forecast = _parse_forecast_list(forecast_list)
        location_name = curr_json.get("name", "")

        adj, reason = _compute_dynamic_adjustment(current, forecast)

        # Cache raw responses
        _save_cache(lat, lon, {
            "current": curr_json,
            "forecast": forecast_list,
            "location_name": location_name,
        })

        logger.info("Live weather fetched for (%.4f, %.4f) — %s", lat, lon, location_name)
        return WeatherData(
            lat=lat, lon=lon, current=current, forecast=forecast,
            source="openweather_live", fetched_at=_now_iso(),
            data_status="LIVE", location_name=location_name,
            dynamic_risk_adjustment=adj,
            dynamic_risk_reason=reason,
        )

    except requests.RequestException as exc:
        logger.warning("Weather API request failed: %s", exc)
        return _unavailable_result(lat, lon, f"Network error: {exc}")
    except Exception as exc:
        logger.warning("Weather fetch unexpected error: %s", exc)
        return _unavailable_result(lat, lon, f"Unexpected error: {exc}")
