"""
PRAVAAH — Short-term Flood-Risk Forecast Engine.

METHODOLOGY (transparent, documented):
───────────────────────────────────────────────────────────────────────────
This is a RISK PROJECTION / ESTIMATE, not a deterministic flood prediction.

Approach: Rainfall-adjusted baseline susceptibility.

1. BASELINE: use the existing ML hazard scores per grid cell as the
   starting susceptibility (range [0, 100]).

2. RAINFALL SIGNAL: obtain forecast precipitation for each horizon
   (24h, 48h, 72h) from the weather client.

3. ADJUSTMENT FORMULA:
     adjusted_score = baseline_score + rainfall_boost
     where:
       rainfall_boost = min(MAX_BOOST, (forecast_mm / BOOST_THRESHOLD) × MAX_BOOST)
       MAX_BOOST       = 15.0 points   (declared constant, not magic number)
       BOOST_THRESHOLD = 50.0 mm       (heavy rain reference)

4. ZONE PROJECTION: apply the same Low/Medium/High thresholds as the
   baseline to determine the projected spatial zone.

5. CONFIDENCE:
     HIGH   — live forecast data available, < 24h horizon
     MEDIUM — live forecast, 24–48h horizon
     LOW    — cached/fallback forecast, or > 48h horizon

The existing trained ML model is NEVER retrained or overridden.
The baseline scores come directly from a completed FloodRiskResult run.
Forecast horizons beyond available data are marked as ESTIMATE only.

LIMITATIONS (documented):
  - Spatial resolution of OWM forecast (point query) may not capture
    local orographic effects.
  - Rainfall-to-runoff relationship is simplified (no soil saturation,
    antecedent conditions, or hydrological routing).
  - Should be verified against independent observations before use in
    operational decision-making.
───────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from flood_risk_zonation.models import ForecastPoint, ForecastResult, WeatherData
from flood_risk_zonation.spatial_zones.classifier import (
    ZONE_GREEN, ZONE_RED, ZONE_WATER, ZONE_YELLOW,
)

logger = logging.getLogger(__name__)

# ── Declared constants ────────────────────────────────────────────────────────
_MAX_BOOST_POINTS   = 15.0   # max score boost from heavy rain
_BOOST_THRESHOLD_MM = 50.0   # rainfall (mm/3h) that triggers max boost
_FORECAST_HORIZONS  = [24, 48, 72]


def _project_zone(score: float, low_t: float, med_t: float) -> str:
    """Project spatial zone from a score using the baseline thresholds."""
    if score > med_t:
        return ZONE_RED
    if score > low_t:
        return ZONE_YELLOW
    return ZONE_GREEN


def _sum_forecast_mm(weather: WeatherData, up_to_hour: int) -> float:
    """Sum forecast precipitation up to `up_to_hour` hours from now."""
    if not weather.forecast:
        return 0.0
    n_steps = max(1, up_to_hour // 3)   # each OWM step is 3h
    total = sum(
        max(0.0, obs.rainfall_mm)
        for obs in weather.forecast[:n_steps]
        if obs.rainfall_mm >= 0
    )
    return round(total, 2)


def _rainfall_boost(forecast_mm: float) -> float:
    """Compute score boost from forecast rainfall. Bounded, declared formula."""
    if forecast_mm <= 0:
        return 0.0
    return round(min(_MAX_BOOST_POINTS, (forecast_mm / _BOOST_THRESHOLD_MM) * _MAX_BOOST_POINTS), 2)


def _confidence(weather: WeatherData, horizon_h: int) -> str:
    if weather.data_status == "UNAVAILABLE":
        return "LOW"
    if weather.data_status in ("LIVE", "CACHED"):
        if horizon_h <= 24:
            return "HIGH" if weather.data_status == "LIVE" else "MEDIUM"
        if horizon_h <= 48:
            return "MEDIUM"
    return "LOW"


def generate_forecast(
    hazard_result: Any,           # FloodRiskResult
    weather: WeatherData,
    horizons: list[int] | None = None,
) -> ForecastResult:
    """
    Generate a multi-horizon flood-risk forecast.

    Parameters
    ----------
    hazard_result : FloodRiskResult
        Completed baseline hazard pipeline result.
    weather : WeatherData
        Weather data from fetch_weather(); may have data_status=UNAVAILABLE.
    horizons : list[int] | None
        Forecast horizons in hours. Default: [24, 48, 72].

    Returns
    -------
    ForecastResult
        Always returns a valid result. Check horizon.provenance for data quality.
    """
    if horizons is None:
        horizons = _FORECAST_HORIZONS

    config = hazard_result.config
    low_t  = config.low_threshold
    med_t  = config.medium_threshold
    grid   = hazard_result.scored_grid

    # Baseline zone counts
    zone_counts = {}
    if "spatial_zone" in grid.columns:
        zone_counts = grid["spatial_zone"].value_counts().to_dict()
    else:
        for z in [ZONE_RED, ZONE_YELLOW, ZONE_GREEN, ZONE_WATER]:
            zone_counts[z] = 0

    bbox = hazard_result.bounding_box
    bbox_key = f"{bbox.min_lon:.4f}_{bbox.min_lat:.4f}_{bbox.max_lon:.4f}_{bbox.max_lat:.4f}"

    horizon_points: list[ForecastPoint] = []

    for h in horizons:
        fc_mm = _sum_forecast_mm(weather, h)
        boost = _rainfall_boost(fc_mm)

        # Per-cell adjusted scores → projected zone counts
        baseline_scores = grid["risk_score"].values.astype(float)
        risk_classes    = grid["risk_class"].values

        projected_red    = 0
        projected_yellow = 0
        projected_green  = 0
        projected_water  = 0

        for score, rc in zip(baseline_scores, risk_classes):
            if str(rc) == "Water":
                projected_water += 1
                continue
            adj = min(100.0, score + boost)
            zone = _project_zone(adj, low_t, med_t)
            if zone == ZONE_RED:
                projected_red += 1
            elif zone == ZONE_YELLOW:
                projected_yellow += 1
            else:
                projected_green += 1

        # Representative area-average scores
        land_mask = [str(rc) != "Water" for rc in risk_classes]
        land_scores = baseline_scores[[i for i, v in enumerate(land_mask) if v]]
        mean_baseline = float(land_scores.mean()) if len(land_scores) > 0 else 0.0
        mean_adjusted = round(min(100.0, mean_baseline + boost), 2)

        prov = (
            "baseline_only" if weather.data_status == "UNAVAILABLE"
            else "forecast_rainfall_adjusted"
        )

        horizon_points.append(ForecastPoint(
            horizon_h=h,
            forecast_rainfall_mm=fc_mm,
            baseline_risk_score=round(mean_baseline, 2),
            adjusted_risk_score=mean_adjusted,
            risk_change=round(mean_adjusted - mean_baseline, 2),
            spatial_zone=ZONE_RED if projected_red > len(baseline_scores) * 0.3
                         else ZONE_YELLOW if projected_yellow > projected_green
                         else ZONE_GREEN,
            confidence=_confidence(weather, h),
            provenance=prov,
        ))

        logger.debug(
            "Forecast h=%dh: fc_mm=%.1f boost=%.1f adj=%.2f RED=%d YELLOW=%d GREEN=%d",
            h, fc_mm, boost, mean_adjusted, projected_red, projected_yellow, projected_green,
        )

    methodology = (
        "Baseline ML hazard susceptibility adjusted by forecast precipitation. "
        f"Boost formula: min({_MAX_BOOST_POINTS}, (forecast_mm / {_BOOST_THRESHOLD_MM}) × {_MAX_BOOST_POINTS}). "
        "ESTIMATE only — not a deterministic flood prediction. "
        f"Weather source: {weather.data_status}."
    )

    return ForecastResult(
        bbox_key=bbox_key,
        baseline_zone_counts=zone_counts,
        horizons=horizon_points,
        weather_source=weather.source,
        forecast_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        methodology=methodology,
    )
