"""
PRAVAAH-AI — What-If Scenario Engine.

DESIGN PRINCIPLE — BASELINE ISOLATION:
  A scenario NEVER modifies the baseline FloodRiskResult or SIHAnalysisResult.
  It creates a parameter-overridden copy of the relevant feature columns,
  re-runs the scoring logic on that copy, and returns a ScenarioResult
  alongside the baseline for comparison.

  The baseline is ALWAYS preserved intact.

SUPPORTED PARAMETERS:
  rainfall_multiplier          : scale rainfall_mean_mm and rainfall_max_24h_mm
  extra_rainfall_mm            : add absolute mm on top of the multiplier
  population_multiplier        : scale population_density
  drainage_capacity_multiplier : scale drainage_capacity (< 1 = degraded)

METHODOLOGY:
  1. Copy the scored_grid feature columns.
  2. Apply parameter overrides to the copy.
  3. Re-run WeightedSusceptibilityModel.predict_proba() on the modified features.
  4. Re-apply FloodRiskScorer to get scenario risk scores and classes.
  5. Apply spatial zone classification to the scenario grid.
  6. Compare zone counts and habitation priorities against baseline.
  7. Return ScenarioResult with provenance="SIMULATION — user-defined parameter override".

  The scenario result is always labelled SIMULATION.
  It must never be presented as a forecast or observation.
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
from flood_risk_zonation.models import ScenarioParameters, ScenarioResult
from flood_risk_zonation.spatial_zones.classifier import (
    classify_spatial_zones,
    ZONE_RED, ZONE_YELLOW, ZONE_GREEN, ZONE_WATER,
)

logger = logging.getLogger(__name__)

_PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _escalated(base_priority: str, scenario_priority: str) -> bool:
    return _PRIORITY_ORDER.get(scenario_priority, 9) < _PRIORITY_ORDER.get(base_priority, 9)


def _deescalated(base_priority: str, scenario_priority: str) -> bool:
    return _PRIORITY_ORDER.get(scenario_priority, 9) > _PRIORITY_ORDER.get(base_priority, 9)


def run_scenario(
    hazard_result: Any,        # FloodRiskResult
    sih_result: Any | None,    # SIHAnalysisResult | None
    params: ScenarioParameters,
) -> ScenarioResult:
    """
    Execute a what-if scenario on top of an existing hazard result.

    Parameters
    ----------
    hazard_result : FloodRiskResult
        Completed baseline hazard run.
    sih_result : SIHAnalysisResult | None
        Baseline habitation intelligence result (for priority comparison).
    params : ScenarioParameters
        Parameter overrides.

    Returns
    -------
    ScenarioResult
        Comparison of scenario vs baseline.
        provenance is always "SIMULATION — user-defined parameter override".
    """
    t0 = time.time()
    grid = hazard_result.scored_grid.copy()
    config = hazard_result.config

    # ── Apply parameter overrides to a copy of features ───────────────────────
    modified = grid.copy()

    # Rainfall scaling
    for col in ("rainfall_mean_mm", "rainfall_max_24h_mm"):
        if col in modified.columns:
            modified[col] = (
                modified[col] * params.rainfall_multiplier
                + params.extra_rainfall_mm
            ).clip(lower=0.0)

    # Population scaling
    if "population_density" in modified.columns:
        modified["population_density"] = (
            modified["population_density"] * params.population_multiplier
        ).clip(lower=0.0)

    # Drainage capacity scaling
    if "drainage_capacity" in modified.columns:
        modified["drainage_capacity"] = (
            modified["drainage_capacity"] * params.drainage_capacity_multiplier
        ).clip(0.0, 1.0)

    # ── Re-score using the fitted model from the baseline run ─────────────────
    try:
        from flood_risk_zonation.scoring.scorer import FloodRiskScorer
        model = hazard_result.analysis_result.model

        available_feats = [c for c in FEATURE_COLUMNS if c in modified.columns]
        X_mod = modified[available_feats].copy()

        # CRITICAL FIX: Preserve the baseline's probability calibration (p_min/p_max)
        # so that scenario risk scores are directly comparable to baseline scores.
        # Without this, the scenario recalibrates from its own probability distribution,
        # causing different normalization bounds and shifting cells across thresholds
        # even when rainfall increases (counterintuitive behavior).
        scorer = FloodRiskScorer()
        
        # Extract baseline probability bounds if available from the hazard_result
        if hasattr(hazard_result, 'analysis_result') and hasattr(hazard_result.analysis_result, 'scorer'):
            baseline_scorer = hazard_result.analysis_result.scorer
            scorer.p_min = baseline_scorer.p_min
            scorer.p_max = baseline_scorer.p_max
            logger.debug(
                "Using baseline calibration: p_min=%.4f, p_max=%.4f",
                scorer.p_min, scorer.p_max
            )
        else:
            # Fallback: calibrate from baseline grid to ensure consistency
            logger.warning(
                "Baseline scorer not available; recalibrating from baseline grid raw probabilities"
            )
            try:
                baseline_raw_probs = model.predict_proba(grid[available_feats].values)[:, -1]
                scorer.calibrate(baseline_raw_probs)
                logger.debug(
                    "Recalibrated from baseline: p_min=%.4f, p_max=%.4f",
                    scorer.p_min, scorer.p_max
                )
            except Exception as recal_exc:
                logger.warning("Recalibration failed (%s); using default bounds", recal_exc)
                scorer.p_min = 0.0
                scorer.p_max = 1.0
        
        thresholds = {"low_max": config.low_threshold, "medium_max": config.medium_threshold}
        scenario_grid = scorer.score_grid(modified, model, available_feats, thresholds, 
                                          use_provided_bounds=True)
        
        # CRITICAL PRESERVATION: Permanent WATER cells must remain WATER
        # A cell is permanent water if it was marked as Water in the baseline.
        # Scenario parameter changes (rainfall, drainage, etc.) should not reclassify
        # permanent water bodies as land-based risk classes.
        # This preserves the scientific integrity: water body extent is independent of
        # temporary rainfall scenarios.
        water_mask = grid["risk_class"] == "Water"
        scenario_grid.loc[water_mask, "risk_class"] = "Water"
        scenario_grid.loc[water_mask, "risk_score"] = 0.0
        
        # CRITICAL FIX: Apply proximity boost to scenario
        # The baseline grid was post-processed with water masking and proximity boost.
        # Scenarios must apply the SAME proximity boost (same water bodies, same distances)
        # for fair comparison. Otherwise scenarios artificially appear lower-risk.
        #
        # Solution: The proximity boost depends only on distance to water bodies (which don't change),
        # not on model probabilities. We can safely apply the baseline's proximity boost to scenario cells,
        # as long as we respect cell-by-cell distance calculations.
        #
        # Implementation: For each cell, apply the maximum of:
        #  1. Cell's raw scenario risk score
        #  2. Proximity boost strength (based on distance to water bodies, same as baseline)
        # This preserves the "boost can only increase, never decrease" semantics.
        try:
            if "water_proximity_score" in grid.columns:
                # Baseline has proximity boost information
                logger.debug("Applying water proximity boost to scenario grid...")
                
                for idx in range(len(scenario_grid)):
                    baseline_prox_score = float(grid["water_proximity_score"].iloc[idx])
                    
                    if baseline_prox_score > 0:
                        # Cell was boosted in baseline; apply same boost to scenario
                        current_scenario_score = float(scenario_grid["risk_score"].iloc[idx])
                        boosted_scenario_score = max(current_scenario_score, baseline_prox_score)
                        
                        scenario_grid.iloc[idx, scenario_grid.columns.get_loc("risk_score")] = boosted_scenario_score
                        
                        # Re-classify if boost changed the class
                        if boosted_scenario_score > config.medium_threshold:
                            scenario_grid.iloc[idx, scenario_grid.columns.get_loc("risk_class")] = "High"
                        elif boosted_scenario_score > config.low_threshold:
                            scenario_grid.iloc[idx, scenario_grid.columns.get_loc("risk_class")] = "Medium"
                        # else: no class change needed
                
                logger.info("Water proximity boost applied to scenario grid")
        except Exception as boost_exc:
            logger.warning("Applying proximity boost to scenario failed: %s", boost_exc)
        
        logger.info(
            "Scenario grid prepared: %d cells rescored, %d permanent WATER cells preserved",
            len(scenario_grid), water_mask.sum()
        )
    except Exception as exc:
        logger.warning("Scenario re-scoring failed (%s) — using original scores.", exc)
        scenario_grid = modified.copy()
        scenario_grid["risk_score"] = modified["risk_score"]
        scenario_grid["risk_class"] = modified["risk_class"]

    # ── Spatial zone classification on scenario grid ───────────────────────────
    scenario_grid = classify_spatial_zones(scenario_grid)

    # ── Zone count comparison ──────────────────────────────────────────────────
    baseline_zones = {}
    if "spatial_zone" in grid.columns:
        baseline_zones = grid["spatial_zone"].value_counts().to_dict()
    else:
        from flood_risk_zonation.spatial_zones.classifier import classify_spatial_zones as _csz
        _z = _csz(grid)
        baseline_zones = _z["spatial_zone"].value_counts().to_dict()

    for z in [ZONE_RED, ZONE_YELLOW, ZONE_GREEN, ZONE_WATER]:
        baseline_zones.setdefault(z, 0)

    scenario_zones = scenario_grid["spatial_zone"].value_counts().to_dict()
    for z in [ZONE_RED, ZONE_YELLOW, ZONE_GREEN, ZONE_WATER]:
        scenario_zones.setdefault(z, 0)

    delta_zones = {z: scenario_zones[z] - baseline_zones[z] for z in baseline_zones}

    # ── Habitation priority comparison ─────────────────────────────────────────
    escalated_habs: list[str] = []
    deescalated_habs: list[str] = []
    baseline_critical = 0
    scenario_critical = 0
    baseline_high     = 0
    scenario_high     = 0

    if sih_result is not None:
        from flood_risk_zonation.exposure.analysis import analyse_exposure
        from flood_risk_zonation.vulnerability.scorer import score_vulnerability
        from flood_risk_zonation.capacity.assessment import _capacity_status, _compute_safe_area
        from flood_risk_zonation.relocation.priority import score_relocation_priority
        from flood_risk_zonation.models import CarryingCapacityResult

        exp_results  = sih_result.exposure_results
        vuln_results = {v.hab_id: v for v in sih_result.vulnerability_results}
        cap_results  = {c.hab_id: c for c in sih_result.capacity_results}
        base_rel_map = {r.hab_id: r for r in sih_result.relocation_results}

        for r in sih_result.relocation_results:
            if r.priority_class == "CRITICAL":
                baseline_critical += 1
            if r.priority_class == "HIGH":
                baseline_high += 1

        # Re-derive exposure from scenario grid
        try:
            hab_ds = sih_result.habitation_dataset
            sc_exp = analyse_exposure(hab_ds, scenario_grid,
                                      low_threshold=config.low_threshold,
                                      medium_threshold=config.medium_threshold)
            for sc_e in sc_exp:
                vuln = vuln_results.get(sc_e.hab_id)
                cap  = cap_results.get(sc_e.hab_id)
                if vuln is None or cap is None:
                    continue

                # Re-score relocation under scenario conditions
                sc_rel = score_relocation_priority(sc_e, vuln, cap)

                base_rel = base_rel_map.get(sc_e.hab_id)
                if base_rel:
                    if _escalated(base_rel.priority_class, sc_rel.priority_class):
                        escalated_habs.append(sc_e.hab_id)
                    elif _deescalated(base_rel.priority_class, sc_rel.priority_class):
                        deescalated_habs.append(sc_e.hab_id)

                if sc_rel.priority_class == "CRITICAL":
                    scenario_critical += 1
                if sc_rel.priority_class == "HIGH":
                    scenario_high += 1
        except Exception as exc:
            logger.warning("Scenario habitation comparison failed: %s", exc)

    # ── Build narrative ────────────────────────────────────────────────────────
    delta_red  = delta_zones.get(ZONE_RED, 0)
    delta_yellow = delta_zones.get(ZONE_YELLOW, 0)
    delta_green = delta_zones.get(ZONE_GREEN, 0)
    delta_water = delta_zones.get(ZONE_WATER, 0)
    delta_crit = scenario_critical - baseline_critical
    
    parts = [f"Scenario: {params.label}."]
    
    # Describe simulated zone classification changes
    if params.rainfall_multiplier != 1.0:
        parts.append(f"Rainfall ×{params.rainfall_multiplier:.1f}" +
                     (f" + {params.extra_rainfall_mm:.0f} mm" if params.extra_rainfall_mm > 0 else "") + ".")
    
    # Use explicit "classification change" language to indicate these are model-derived
    zone_changes = []
    if delta_red != 0:
        zone_changes.append(f"RED {delta_red:+d}")
    if delta_yellow != 0:
        zone_changes.append(f"YELLOW {delta_yellow:+d}")
    if delta_green != 0:
        zone_changes.append(f"GREEN {delta_green:+d}")
    if zone_changes:
        parts.append(f"Simulated classification change: {', '.join(zone_changes)} cells.")
    else:
        parts.append("No simulated classification change in zone cells.")
    
    # Habitation priority language
    if delta_crit > 0:
        parts.append(f"{delta_crit} additional habitation(s) received a CRITICAL priority in this scenario.")
    elif delta_crit < 0:
        parts.append(f"{abs(delta_crit)} fewer habitation(s) received CRITICAL priority in this scenario.")
    if escalated_habs:
        parts.append(f"Scenario impact: {len(escalated_habs)} habitation(s) received a higher simulated priority.")
    
    parts.append("SIMULATION — not a forecast or observation.")
    narrative = " ".join(parts)

    duration = time.time() - t0
    logger.info(
        "Scenario '%s' complete in %.1fs. ΔRED=%+d ΔCRIT=%+d escalated=%d",
        params.label, duration, delta_red, delta_crit, len(escalated_habs),
    )

    return ScenarioResult(
        scenario_id=params.scenario_id,
        parameters=params,
        baseline_zone_counts=baseline_zones,
        scenario_zone_counts=scenario_zones,
        delta_zone_counts=delta_zones,
        baseline_critical=baseline_critical,
        scenario_critical=scenario_critical,
        delta_critical=delta_crit,
        baseline_high=baseline_high,
        scenario_high=scenario_high,
        habitations_escalated=escalated_habs,
        habitations_deescalated=deescalated_habs,
        narrative=narrative,
    )


def build_preset_scenarios() -> list[ScenarioParameters]:
    """Return the standard preset scenario parameter list."""
    return [
        ScenarioParameters("sc_rain_10", "+10% Rainfall",  rainfall_multiplier=1.10),
        ScenarioParameters("sc_rain_20", "+20% Rainfall",  rainfall_multiplier=1.20),
        ScenarioParameters("sc_rain_30", "+30% Rainfall",  rainfall_multiplier=1.30),
        ScenarioParameters("sc_rain_50", "+50% Rainfall",  rainfall_multiplier=1.50),
        ScenarioParameters("sc_rain_extreme", "+100% Rainfall (Extreme)", rainfall_multiplier=2.00,
                           description="Doubling of rainfall — extreme scenario."),
        ScenarioParameters("sc_drain_deg", "Degraded Drainage (−30%)", drainage_capacity_multiplier=0.70,
                           description="30% reduction in drainage capacity."),
        ScenarioParameters("sc_combined", "+30% Rain + Degraded Drainage",
                           rainfall_multiplier=1.30, drainage_capacity_multiplier=0.70,
                           description="Combined rainfall increase and drainage degradation."),
    ]
