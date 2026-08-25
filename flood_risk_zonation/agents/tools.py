"""
PRAVAAH-AI — Agent Tool Definitions.

Clean, bounded tool functions that agents call to retrieve structured
PRAVAAH data.  Tools are pure functions: they take IDs/parameters and
return structured dicts.  They never accept free-form text or allow
agents to invent values — all data comes directly from the pipeline.

Design principle:
  - Tools return small, targeted dicts rather than entire GeoDataFrames.
  - Agents receive only the data they need for their reasoning step.
  - All returned values have explicit provenance labels.
  - Tools raise ValueError with a clear message if required data is missing,
    which the orchestrator handles gracefully.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_hazard_details(
    hab_id: str,
    exposure_results: list,
    zoned_grid: Any | None,
    habitation_zones: dict,
    scored_grid: Any,
) -> dict:
    """
    Return structured hazard metrics for a single habitation.

    Parameters
    ----------
    hab_id : str
    exposure_results : list[ExposureResult]
    zoned_grid : GeoDataFrame | None
    habitation_zones : dict   (hab_id → zone string)
    scored_grid : GeoDataFrame  (the raw hazard grid with risk_score, features)

    Returns
    -------
    dict with keys:
        hab_id, hazard_score, hazard_class, spatial_zone,
        pct_high_risk, dominant_features, is_coastal
    """
    exp = next((e for e in exposure_results if e.hab_id == hab_id), None)
    if exp is None:
        raise ValueError(f"No exposure data found for hab_id={hab_id!r}")

    zone = habitation_zones.get(hab_id, "UNKNOWN")

    # Extract feature values from nearby grid cells
    dominant_features: list[str] = []
    if scored_grid is not None and exp.intersecting_cell_ids:
        try:
            import pandas as pd
            if "cell_id" in scored_grid.columns:
                mask = scored_grid["cell_id"].isin(exp.intersecting_cell_ids)
                nearby = scored_grid[mask]
            else:
                nearby = scored_grid.head(4)
            # Identify the most impactful features (those with high/low values
            # relative to known risk directions)
            _risk_direction = {
                "elevation_m":       ("low",  30.0),
                "dist_water_m":      ("low",  500.0),
                "drainage_capacity": ("low",  0.45),
                "twi":               ("high", 10.0),
                "slope_deg":         ("low",  2.0),
            }
            for col, (direction, threshold) in _risk_direction.items():
                if col in nearby.columns:
                    val = float(nearby[col].mean())
                    if direction == "low" and val < threshold:
                        dominant_features.append(f"{col.replace('_', ' ').title()}: {val:.1f} (unfavourable)")
                    elif direction == "high" and val > threshold:
                        dominant_features.append(f"{col.replace('_', ' ').title()}: {val:.1f} (unfavourable)")
        except Exception as e:
            logger.debug("Feature extraction in tool: %s", e)

    return {
        "hab_id":          hab_id,
        "hazard_score":    exp.hazard_score,
        "hazard_class":    exp.hazard_class,
        "spatial_zone":    zone,
        "pct_high_risk":   round(exp.pct_high_risk, 3),
        "is_coastal":      False,   # updated by orchestrator if coastal cells found
        "dominant_features": dominant_features[:4],
        "provenance":      "ml_hazard_engine",
    }


def get_exposure_details(
    hab_id: str,
    exposure_results: list,
) -> dict:
    """
    Return structured exposure metrics for a single habitation.

    Population is labelled with its provenance:
      "osm_tag"   — directly from OSM population tag
      "UNKNOWN"   — not available, never fabricated
    """
    exp = next((e for e in exposure_results if e.hab_id == hab_id), None)
    if exp is None:
        raise ValueError(f"No exposure data found for hab_id={hab_id!r}")

    pop_label = (
        f"{exp.population_exposed:,} (OSM tag)"
        if exp.population_source == "osm_tag" and exp.population_exposed
        else "UNKNOWN (not in OSM data)"
    )

    return {
        "hab_id":            hab_id,
        "name":              exp.name or "Unnamed",
        "hab_type":          exp.hab_type,
        "is_in_red_zone":    exp.is_in_red_zone,
        "hazard_class":      exp.hazard_class,
        "population_label":  pop_label,
        "population_value":  exp.population_exposed,
        "population_source": exp.population_source,
        "pct_high_risk":     round(exp.pct_high_risk, 3),
        "provenance":        "exposure_analysis",
    }


def get_vulnerability_details(
    hab_id: str,
    vulnerability_results: list,
) -> dict:
    """Return structured vulnerability metrics for a single habitation."""
    vuln = next((v for v in vulnerability_results if v.hab_id == hab_id), None)
    if vuln is None:
        raise ValueError(f"No vulnerability data found for hab_id={hab_id!r}")

    return {
        "hab_id":               hab_id,
        "vulnerability_score":  vuln.vulnerability_score,
        "vulnerability_class":  vuln.vulnerability_class,
        "component_scores":     vuln.component_scores,
        "component_weights":    vuln.component_weights,
        "dominant_factors":     vuln.factors[:4],
        "provenance":           "vulnerability_scorer",
    }


def get_capacity_details(
    hab_id: str,
    capacity_results: list,
) -> dict:
    """Return structured capacity metrics for a single habitation."""
    cap = next((c for c in capacity_results if c.hab_id == hab_id), None)
    if cap is None:
        raise ValueError(f"No capacity data found for hab_id={hab_id!r}")

    hc_str = f"{cap.nearest_healthcare_km:.1f} km" if cap.nearest_healthcare_km >= 0 else "not found in area"
    road_str = f"{cap.nearest_road_km:.1f} km" if cap.nearest_road_km >= 0 else "not found in area"

    return {
        "hab_id":                hab_id,
        "capacity_score":        cap.capacity_score,
        "capacity_status":       cap.capacity_status,
        "safe_area_km2":         cap.safe_area_km2,
        "search_radius_km":      cap.search_radius_km,
        "nearest_healthcare":    hc_str,
        "nearest_road":          road_str,
        "shelter_capacity":      cap.shelter_capacity or "unavailable",
        "shelter_source":        cap.shelter_source,
        "notes":                 cap.notes,
        "provenance":            "capacity_assessment",
    }


def get_relocation_details(
    hab_id: str,
    relocation_results: list,
) -> dict:
    """Return structured relocation priority metrics for a single habitation."""
    rel = next((r for r in relocation_results if r.hab_id == hab_id), None)
    if rel is None:
        raise ValueError(f"No relocation data found for hab_id={hab_id!r}")

    return {
        "hab_id":              hab_id,
        "name":                rel.name or "Unnamed",
        "relocation_score":    rel.relocation_score,
        "priority_class":      rel.priority_class,
        "recommended_action":  rel.recommended_action,
        "contributing_factors":rel.contributing_factors,
        "component_scores":    rel.component_scores,
        "weights":             rel.weights,
        "is_coastal":          rel.is_coastal,
        "provenance":          "relocation_priority_engine",
    }


def find_relocation_candidates_tool(
    hab_id: str,
    candidates_map: dict,
) -> list[dict]:
    """
    Return pre-computed relocation candidate summaries for a habitation.

    Parameters
    ----------
    hab_id : str
    candidates_map : dict   hab_id → list[RelocationCandidate]

    Returns
    -------
    list[dict]  — serialised candidate records suitable for agent consumption
    """
    candidates = candidates_map.get(hab_id, [])
    return [
        {
            "candidate_id":        c.candidate_id,
            "distance_km":         c.distance_km,
            "area_km2":            c.area_km2,
            "candidate_score":     c.candidate_score,
            "mean_hazard_score":   c.mean_hazard_score,
            "nearest_road_km":     c.nearest_road_km,
            "nearest_healthcare_km": c.nearest_healthcare_km,
            "notes":               c.notes,
            "provenance":          c.data_provenance,
        }
        for c in candidates
    ]


def compare_relocation_candidates_tool(
    candidates: list[dict],
) -> dict:
    """
    Compare a list of candidate dicts and return a structured comparison.

    Returns
    -------
    dict with keys:
        best_candidate_id, ranking, comparison_narrative
    """
    if not candidates:
        return {
            "best_candidate_id": None,
            "ranking": [],
            "comparison_narrative": "No relocation candidates found in the search area.",
        }

    sorted_by_score = sorted(candidates, key=lambda c: c["candidate_score"], reverse=True)
    best = sorted_by_score[0]

    ranking = [
        {
            "rank":           i + 1,
            "candidate_id":   c["candidate_id"],
            "score":          c["candidate_score"],
            "distance_km":    c["distance_km"],
            "area_km2":       c["area_km2"],
        }
        for i, c in enumerate(sorted_by_score)
    ]

    # Build a plain comparison narrative (deterministic, no LLM needed)
    if len(sorted_by_score) == 1:
        narrative = (
            f"Only one candidate area found (ID: {best['candidate_id']}). "
            f"Distance: {best['distance_km']:.1f} km, "
            f"area: {best['area_km2']:.2f} km², "
            f"score: {best['candidate_score']:.3f}."
        )
    else:
        second = sorted_by_score[1]
        narrative = (
            f"Top candidate: {best['candidate_id']} "
            f"(score {best['candidate_score']:.3f}, "
            f"distance {best['distance_km']:.1f} km, "
            f"area {best['area_km2']:.2f} km²). "
            f"Second candidate: {second['candidate_id']} "
            f"(score {second['candidate_score']:.3f}, "
            f"distance {second['distance_km']:.1f} km). "
        )
        if best["candidate_score"] - second["candidate_score"] > 0.1:
            narrative += f"Top candidate is substantially better."
        else:
            narrative += f"Candidates are comparable in quality."

    return {
        "best_candidate_id":    best["candidate_id"],
        "ranking":              ranking,
        "comparison_narrative": narrative,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTELLIGENCE ENHANCEMENT TOOLS — Weather, Forecast, Scenario, Validation
# ─────────────────────────────────────────────────────────────────────────────

def get_weather_summary(weather_data: Any) -> dict:
    """
    Return a compact structured weather summary for agent consumption.

    Parameters
    ----------
    weather_data : WeatherData

    Returns
    -------
    dict with keys: status, current_rainfall_mm, max_forecast_mm_24h,
                    dynamic_risk_adjustment, reason, source, timestamp
    """
    if weather_data is None or weather_data.data_status == "UNAVAILABLE":
        return {
            "status":                   "UNAVAILABLE",
            "current_rainfall_mm":      -1.0,
            "max_forecast_mm_24h":      -1.0,
            "dynamic_risk_adjustment":  0.0,
            "reason":                   "Weather data not available.",
            "source":                   "unavailable",
            "timestamp":                "",
        }

    curr_mm = -1.0
    if weather_data.current and weather_data.current.rainfall_mm >= 0:
        curr_mm = weather_data.current.rainfall_mm

    fc_24h = [
        obs.rainfall_mm
        for obs in weather_data.forecast[:8]
        if obs.rainfall_mm >= 0
    ]
    max_fc = round(max(fc_24h), 2) if fc_24h else -1.0

    return {
        "status":                   weather_data.data_status,
        "current_rainfall_mm":      curr_mm,
        "max_forecast_mm_24h":      max_fc,
        "dynamic_risk_adjustment":  weather_data.dynamic_risk_adjustment,
        "reason":                   weather_data.dynamic_risk_reason,
        "source":                   weather_data.source,
        "timestamp":                weather_data.fetched_at,
    }


def get_forecast_summary(forecast_result: Any) -> dict:
    """
    Return a compact forecast summary for agent consumption.

    Parameters
    ----------
    forecast_result : ForecastResult | None

    Returns
    -------
    dict with keys: available, methodology, horizons (list of dicts)
    """
    if forecast_result is None:
        return {"available": False, "methodology": "No forecast available.", "horizons": []}

    horizon_dicts = [
        {
            "horizon_h":            h.horizon_h,
            "forecast_rainfall_mm": h.forecast_rainfall_mm,
            "baseline_risk_score":  h.baseline_risk_score,
            "adjusted_risk_score":  h.adjusted_risk_score,
            "risk_change":          h.risk_change,
            "spatial_zone":         h.spatial_zone,
            "confidence":           h.confidence,
            "provenance":           h.provenance,
        }
        for h in forecast_result.horizons
    ]
    return {
        "available":    True,
        "methodology":  forecast_result.methodology,
        "weather_source": forecast_result.weather_source,
        "timestamp":    forecast_result.forecast_timestamp,
        "horizons":     horizon_dicts,
    }


def get_scenario_summary(scenario_result: Any) -> dict:
    """
    Return a compact scenario comparison summary for agent consumption.

    Parameters
    ----------
    scenario_result : ScenarioResult | None
    """
    if scenario_result is None:
        return {"available": False, "narrative": "No scenario run.", "delta_zone_counts": {}}

    return {
        "available":              True,
        "scenario_label":         scenario_result.parameters.label,
        "narrative":              scenario_result.narrative,
        "delta_zone_counts":      scenario_result.delta_zone_counts,
        "delta_critical":         scenario_result.delta_critical,
        "baseline_critical":      scenario_result.baseline_critical,
        "scenario_critical":      scenario_result.scenario_critical,
        "habitations_escalated":  len(scenario_result.habitations_escalated),
        "habitations_escalated_ids": scenario_result.habitations_escalated,
        "provenance":             scenario_result.provenance,
    }


def get_validation_summary(validation_result: Any) -> dict:
    """
    Return a compact validation summary for agent consumption.

    Parameters
    ----------
    validation_result : ValidationResult | None
    """
    if validation_result is None or validation_result.data_status == "NO_EVENTS_AVAILABLE":
        return {
            "available":     False,
            "data_status":   "NO_EVENTS_AVAILABLE",
            "events":        [],
            "overall_notes": "No historical events available for this area.",
        }

    metric_dicts = [
        {
            "event_id":   m.event_id,
            "precision":  m.precision,
            "recall":     m.recall,
            "f1_score":   m.f1_score,
            "iou":        m.iou,
            "overlap":    m.overlap_count,
            "predicted":  m.predicted_high_count,
            "observed":   m.observed_flood_count,
        }
        for m in validation_result.metrics
    ]
    return {
        "available":     True,
        "data_status":   validation_result.data_status,
        "events":        [e.event_name for e in validation_result.events],
        "metrics":       metric_dicts,
        "overall_notes": validation_result.overall_notes,
    }
