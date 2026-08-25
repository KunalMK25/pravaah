"""
PRAVAAH — Relocation priority scorer.

METHODOLOGY (fully declared, transparent):
─────────────────────────────────────────────────────────────────────────────
Relocation priority answers: "Which habitations need intervention most
urgently, and why?"

Formula:
  relocation_score =
      w_hazard     × norm(hazard_score/100)
    + w_vuln       × vulnerability_score
    + w_cap_stress × (1 − capacity_score)
    + w_exposure   × exposure_component

  where exposure_component:
    = 1.0  if population is known and hazard_class == "High"
    = 0.7  if hazard_class == "High" but population unknown
    = 0.4  if hazard_class == "Medium"
    = 0.1  otherwise

Declared weights (visible in code and in the PRAVAAH UI):
  w_hazard     = 0.35
  w_vuln       = 0.30
  w_cap_stress = 0.20
  w_exposure   = 0.15
  (sum = 1.00)

Guardrails (deterministic, documented):
  • no_population_guardrail: if population_source == "UNKNOWN" AND
    hazard_class != "High", priority is capped at HIGH.
  • water_cell_guardrail: habitations classified as Water cannot be CRITICAL.
  • coastal_escalation: coastal habitations with HIGH priority are escalated
    to CRITICAL.
  • capacity_escalation: CRITICAL capacity status adds 0.10 to the score.

Action classes:
  [0.00 – 0.25) → LOW      — Routine monitoring
  [0.25 – 0.50) → MEDIUM   — Preparedness / monitoring
  [0.50 – 0.75) → HIGH     — Priority intervention / evacuation planning
  [0.75 – 1.00] → CRITICAL — Immediate relocation priority
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging

from flood_risk_zonation.models import (
    CarryingCapacityResult,
    ExposureResult,
    RelocationPriorityResult,
    VulnerabilityResult,
)

logger = logging.getLogger(__name__)

# ── Declared weights ──────────────────────────────────────────────────────────
RELOCATION_WEIGHTS: dict[str, float] = {
    "hazard":     0.35,
    "vulnerability": 0.30,
    "cap_stress": 0.20,
    "exposure":   0.15,
}
assert abs(sum(RELOCATION_WEIGHTS.values()) - 1.0) < 1e-9

# ── Classification thresholds ─────────────────────────────────────────────────
_THRESHOLDS = [
    ("LOW",      0.25),
    ("MEDIUM",   0.50),
    ("HIGH",     0.75),
    ("CRITICAL", 1.01),
]

# ── Recommended action text ───────────────────────────────────────────────────
_ACTIONS = {
    "LOW": (
        "Routine monitoring — include in seasonal flood preparedness plans. "
        "No immediate intervention required."
    ),
    "MEDIUM": (
        "Preparedness action required — conduct community awareness, "
        "identify evacuation routes, and maintain alert readiness."
    ),
    "HIGH": (
        "Priority intervention — initiate evacuation planning, pre-position "
        "relief materials, and engage local authorities for immediate review."
    ),
    "CRITICAL": (
        "IMMEDIATE RELOCATION PRIORITY — this habitation faces critical "
        "hazard exposure with inadequate carrying capacity. Authorities "
        "should initiate relocation procedures without delay."
    ),
}


def _classify(score: float) -> str:
    for label, upper in _THRESHOLDS:
        if score < upper:
            return label
    return "CRITICAL"


def score_relocation_priority(
    exposure: ExposureResult,
    vulnerability: VulnerabilityResult,
    capacity: CarryingCapacityResult,
    is_coastal: bool = False,
) -> RelocationPriorityResult:
    """
    Compute relocation priority for a single habitation.

    Parameters
    ----------
    exposure : ExposureResult
    vulnerability : VulnerabilityResult
    capacity : CarryingCapacityResult
    is_coastal : bool
        Whether the habitation is flagged as coastal/tsunami risk.

    Returns
    -------
    RelocationPriorityResult
    """
    # ── Component scores ──────────────────────────────────────────────────────

    # Hazard component: normalise from [0, 100] → [0, 1]
    c_hazard = round(exposure.hazard_score / 100.0, 4)

    # Vulnerability component: already in [0, 1]
    c_vuln = round(vulnerability.vulnerability_score, 4)

    # Capacity stress component: inverse of capacity score
    c_cap_stress = round(1.0 - capacity.capacity_score, 4)

    # Exposure component: accounts for population and hazard class
    if (
        exposure.population_source == "osm_tag"
        and exposure.population_exposed
        and exposure.hazard_class == "High"
    ):
        c_exposure = 1.0
    elif exposure.hazard_class == "High":
        c_exposure = 0.7
    elif exposure.hazard_class == "Medium":
        c_exposure = 0.4
    else:
        c_exposure = 0.1
    c_exposure = round(c_exposure, 4)

    components = {
        "hazard":       c_hazard,
        "vulnerability": c_vuln,
        "cap_stress":   c_cap_stress,
        "exposure":     c_exposure,
    }

    # ── Weighted composite ────────────────────────────────────────────────────
    score = (
        c_hazard      * RELOCATION_WEIGHTS["hazard"]
        + c_vuln      * RELOCATION_WEIGHTS["vulnerability"]
        + c_cap_stress* RELOCATION_WEIGHTS["cap_stress"]
        + c_exposure  * RELOCATION_WEIGHTS["exposure"]
    )

    # ── Capacity escalation guardrail ─────────────────────────────────────────
    if capacity.capacity_status == "CRITICAL":
        score = min(1.0, score + 0.10)

    score = round(max(0.0, min(1.0, score)), 4)
    priority = _classify(score)

    # ── Guardrail: water-cell habitations cannot be CRITICAL ─────────────────
    if exposure.hazard_class == "Water":
        priority = min(priority, "HIGH")   # water ≠ relocation priority

    # ── Coastal escalation: HIGH → CRITICAL for coastal hazard ───────────────
    if is_coastal and priority == "HIGH":
        priority = "CRITICAL"
        score = min(1.0, score + 0.05)

    # ── Guardrail: no pop + non-high hazard → cap at HIGH ────────────────────
    if (
        exposure.population_source == "UNKNOWN"
        and exposure.hazard_class not in ("High",)
        and priority == "CRITICAL"
    ):
        priority = "HIGH"

    # ── Build contributing factors narrative ──────────────────────────────────
    factor_parts: list[str] = []

    if c_hazard >= 0.5:
        factor_parts.append(
            f"High hazard score ({exposure.hazard_score:.1f}/100, class: {exposure.hazard_class})"
        )
    if c_vuln >= 0.5:
        factor_parts.append(
            f"High vulnerability ({vulnerability.vulnerability_class}, score {vulnerability.vulnerability_score:.2f})"
        )
    if c_cap_stress >= 0.6:
        factor_parts.append(
            f"Stressed carrying capacity (status: {capacity.capacity_status}, score {capacity.capacity_score:.2f})"
        )
    if capacity.safe_area_km2 < 0.5:
        factor_parts.append(
            f"Very limited nearby safe area ({capacity.safe_area_km2:.2f} km² within {capacity.search_radius_km:.0f}km)"
        )
    if capacity.nearest_road_km >= 3.0 or capacity.nearest_road_km < 0:
        factor_parts.append(
            "Poor road accessibility"
            if capacity.nearest_road_km < 0
            else f"Remote road access ({capacity.nearest_road_km:.1f}km to major road)"
        )
    if capacity.nearest_healthcare_km >= 10.0 or capacity.nearest_healthcare_km < 0:
        factor_parts.append(
            "No healthcare facility found in area"
            if capacity.nearest_healthcare_km < 0
            else f"Distant healthcare ({capacity.nearest_healthcare_km:.1f}km)"
        )
    if exposure.population_source == "osm_tag" and exposure.population_exposed:
        factor_parts.append(
            f"Known exposed population: {exposure.population_exposed:,}"
        )
    if is_coastal:
        factor_parts.append("Coastal / tsunami inundation risk")
    if c_exposure >= 0.7:
        factor_parts.append(
            f"High population exposure (hazard class: {exposure.hazard_class})"
        )

    if not factor_parts:
        if priority in ("LOW", "MEDIUM"):
            factor_parts.append(f"Moderate hazard ({exposure.hazard_class}) with adequate capacity")
        else:
            factor_parts.append("Combined hazard, vulnerability, and capacity stress")

    # ── Full explanation narrative (for detail panel) ─────────────────────────
    pop_str = (
        f"{exposure.population_exposed:,} (OSM data)"
        if exposure.population_source == "osm_tag" and exposure.population_exposed
        else "UNKNOWN (not in OSM)"
    )
    explanation = (
        f"WHY THIS HABITATION IS {priority}\n\n"
        f"  Hazard score:          {exposure.hazard_score:.1f}/100 → class {exposure.hazard_class}\n"
        f"  Vulnerability score:   {vulnerability.vulnerability_score:.3f} → {vulnerability.vulnerability_class}\n"
        f"  Capacity score:        {capacity.capacity_score:.3f} → {capacity.capacity_status}\n"
        f"  Population exposed:    {pop_str}\n"
        f"  Safe area nearby:      {capacity.safe_area_km2:.2f} km²\n"
        f"  Nearest road:          "
        + (f"{capacity.nearest_road_km:.1f} km" if capacity.nearest_road_km >= 0 else "not found")
        + f"\n"
        f"  Nearest healthcare:    "
        + (f"{capacity.nearest_healthcare_km:.1f} km" if capacity.nearest_healthcare_km >= 0 else "not found")
        + f"\n"
        f"  Coastal risk:          {'Yes' if is_coastal else 'No'}\n\n"
        f"Component weights used:\n"
        f"  Hazard weight:        {RELOCATION_WEIGHTS['hazard']:.0%}\n"
        f"  Vulnerability weight: {RELOCATION_WEIGHTS['vulnerability']:.0%}\n"
        f"  Capacity weight:      {RELOCATION_WEIGHTS['cap_stress']:.0%}\n"
        f"  Exposure weight:      {RELOCATION_WEIGHTS['exposure']:.0%}\n\n"
        f"Key factors:\n"
        + "\n".join(f"  • {f}" for f in factor_parts)
    )

    logger.debug(
        "Relocation: %s → score=%.3f priority=%s", exposure.hab_id, score, priority
    )

    return RelocationPriorityResult(
        hab_id=exposure.hab_id,
        name=exposure.name,
        relocation_score=score,
        priority_class=priority,
        recommended_action=_ACTIONS[priority],
        contributing_factors=factor_parts,
        component_scores=components,
        weights=RELOCATION_WEIGHTS,
        hazard_score=exposure.hazard_score,
        vulnerability_score=vulnerability.vulnerability_score,
        capacity_score=capacity.capacity_score,
        population_exposed=exposure.population_exposed,
        population_source=exposure.population_source,
        is_coastal=is_coastal,
        explanation=explanation,
    )
