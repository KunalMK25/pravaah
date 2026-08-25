"""
PRAVAAH — Bounded Agent Implementations.

Each agent receives a structured context dict (pre-populated by the orchestrator
from pipeline outputs), calls at most one or two tool functions, and returns a
structured AgentEvidence record.

Agents NEVER:
  - invent hazard scores, distances, populations, or capacities
  - issue official evacuation orders
  - loop indefinitely
  - call external APIs directly (that is the LLM provider layer's concern)

Agents ALWAYS:
  - ground their output in the structured PRAVAAH metrics passed to them
  - label outputs with data provenance
  - degrade gracefully when the LLM is unavailable
  - return a valid AgentEvidence even in fallback mode

LLM INTEGRATION:
  The LLM (if available) provides natural-language interpretation.
  It is given ONLY the relevant structured metrics for a single habitation —
  not the entire grid — keeping context windows small and costs low.

  If the LLM call fails, each agent falls back to a deterministic
  rule-based AgentEvidence with ai_assisted=False.

  Provider is configured via environment variables:
    PRAVAAH_LLM_PROVIDER  = "openai" | "anthropic" | "none"
    OPENAI_API_KEY         (for OpenAI)
    ANTHROPIC_API_KEY      (for Anthropic)

  A provider value of "none" (or missing env vars) disables the LLM entirely.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from flood_risk_zonation.models import AgentEvidence

logger = logging.getLogger(__name__)

# ── LLM provider configuration ────────────────────────────────────────────────
_PROVIDER = os.environ.get("PRAVAAH_LLM_PROVIDER", "none").lower().strip()
_MAX_LLM_RETRIES = 1
_LLM_TIMEOUT_S   = 15   # seconds per call
_MAX_TOKENS      = 300  # keep outputs concise


def _llm_available() -> bool:
    """Return True if a LLM provider is configured and the required key is present."""
    if _PROVIDER == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if _PROVIDER == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return False


def _call_llm(system_prompt: str, user_message: str) -> str | None:
    """
    Make one LLM call with the configured provider.

    Returns the response text, or None on any failure.
    The caller is responsible for fallback logic.
    """
    if not _llm_available():
        return None

    try:
        if _PROVIDER == "openai":
            import openai
            client = openai.OpenAI(timeout=_LLM_TIMEOUT_S)
            resp = client.chat.completions.create(
                model=os.environ.get("PRAVAAH_OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=_MAX_TOKENS,
                temperature=0.1,   # near-deterministic for decision support
            )
            return resp.choices[0].message.content.strip()

        if _PROVIDER == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=os.environ.get("PRAVAAH_ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
                max_tokens=_MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text.strip()

    except Exception as exc:
        logger.warning("LLM call failed (%s): %s", _PROVIDER, exc)
    return None


def _severity_from_score(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    if score >= 0.50:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# HAZARD ANALYST AGENT
# ─────────────────────────────────────────────────────────────────────────────

_HAZARD_SYSTEM = (
    "You are the Hazard Analyst for PRAVAAH, a geospatial decision-support system. "
    "You interpret structured hazard metrics produced by a geospatial ML model. "
    "You explain WHY a location is hazardous using the provided metrics only. "
    "You never invent, assume, or extrapolate beyond the data given. "
    "Respond in 2-3 sentences. Focus on the dominant risk factors. "
    "Never declare official emergency orders."
)


def run_hazard_agent(hazard_data: dict) -> AgentEvidence:
    """
    Interpret hazard metrics for one habitation.

    Parameters
    ----------
    hazard_data : dict
        Output of tools.get_hazard_details().

    Returns
    -------
    AgentEvidence
    """
    score = hazard_data["hazard_score"] / 100.0
    severity = _severity_from_score(score)
    dominant = hazard_data.get("dominant_features", [])

    # Deterministic fallback summary
    zone = hazard_data.get("spatial_zone", "UNKNOWN")
    fallback_summary = (
        f"Hazard score: {hazard_data['hazard_score']:.1f}/100 "
        f"(class: {hazard_data['hazard_class']}, zone: {zone}). "
        + (f"Dominant risk factors: {', '.join(dominant[:3])}." if dominant
           else "No specific dominant features identified from available data.")
    )

    ai_summary = None
    if _llm_available() and severity in ("HIGH", "CRITICAL"):
        # Only invoke LLM for high-stakes habitations to control cost
        user_msg = (
            f"Hazard metrics for a habitation:\n"
            f"  Hazard score: {hazard_data['hazard_score']:.1f}/100\n"
            f"  Hazard class: {hazard_data['hazard_class']}\n"
            f"  Spatial zone: {zone}\n"
            f"  % high-risk cells nearby: {hazard_data['pct_high_risk']:.0%}\n"
            f"  Dominant factors: {dominant}\n"
            f"Explain in 2-3 sentences why this location is at risk."
        )
        ai_summary = _call_llm(_HAZARD_SYSTEM, user_msg)

    return AgentEvidence(
        agent_name="HazardAnalyst",
        summary=ai_summary if ai_summary else fallback_summary,
        severity=severity,
        key_factors=dominant,
        metrics={
            "hazard_score":   hazard_data["hazard_score"],
            "hazard_class":   hazard_data["hazard_class"],
            "spatial_zone":   zone,
            "pct_high_risk":  hazard_data["pct_high_risk"],
        },
        ai_assisted=bool(ai_summary),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXPOSURE ANALYST AGENT
# ─────────────────────────────────────────────────────────────────────────────

_EXPOSURE_SYSTEM = (
    "You are the Exposure Analyst for PRAVAAH. "
    "You interpret structured exposure data about habitations at risk. "
    "If population is UNKNOWN, you say so explicitly — you never fabricate numbers. "
    "Respond in 2-3 sentences. State clearly what is known and what is not."
)


def run_exposure_agent(exposure_data: dict) -> AgentEvidence:
    """Interpret exposure metrics for one habitation."""
    is_red = exposure_data.get("is_in_red_zone", False)
    pop_label = exposure_data.get("population_label", "UNKNOWN")
    pop_src = exposure_data.get("population_source", "UNKNOWN")
    hclass = exposure_data.get("hazard_class", "Unknown")

    if is_red:
        severity = "HIGH" if pop_src == "UNKNOWN" else "CRITICAL"
    elif hclass == "Medium":
        severity = "MEDIUM"
    else:
        severity = "LOW"

    fallback_summary = (
        f"Habitation '{exposure_data.get('name', 'Unknown')}' "
        f"({'in' if is_red else 'not in'} red zone). "
        f"Population: {pop_label}. "
        f"Hazard class: {hclass}."
    )
    factors = []
    if is_red:
        factors.append("Located in primary hazard (RED) zone")
    if pop_src == "UNKNOWN":
        factors.append("Population exposure unknown — precautionary principle applies")
    if exposure_data.get("pct_high_risk", 0) > 0.5:
        factors.append(f"{exposure_data['pct_high_risk']:.0%} of surrounding cells are high-risk")

    ai_summary = None
    if _llm_available() and is_red:
        user_msg = (
            f"Exposure data for habitation '{exposure_data.get('name', 'Unknown')}':\n"
            f"  In red zone: {is_red}\n"
            f"  Population: {pop_label}\n"
            f"  Hazard class: {hclass}\n"
            f"  % high-risk cells: {exposure_data.get('pct_high_risk', 0):.0%}\n"
            f"Summarise the exposure situation in 2-3 sentences."
        )
        ai_summary = _call_llm(_EXPOSURE_SYSTEM, user_msg)

    return AgentEvidence(
        agent_name="ExposureAnalyst",
        summary=ai_summary if ai_summary else fallback_summary,
        severity=severity,
        key_factors=factors,
        metrics={
            "is_in_red_zone":    is_red,
            "population_label":  pop_label,
            "population_source": pop_src,
            "hazard_class":      hclass,
        },
        ai_assisted=bool(ai_summary),
    )


# ─────────────────────────────────────────────────────────────────────────────
# VULNERABILITY ANALYST AGENT
# ─────────────────────────────────────────────────────────────────────────────

_VULN_SYSTEM = (
    "You are the Vulnerability Analyst for PRAVAAH. "
    "You interpret structured vulnerability indicator scores. "
    "All scores come from a transparent weighted geospatial model — you do not alter them. "
    "Respond in 2-3 sentences. Identify the dominant vulnerability drivers."
)


def run_vulnerability_agent(vuln_data: dict) -> AgentEvidence:
    """Interpret vulnerability metrics for one habitation."""
    score = vuln_data["vulnerability_score"]
    vclass = vuln_data["vulnerability_class"]
    factors = vuln_data.get("dominant_factors", [])

    fallback_summary = (
        f"Vulnerability: {vclass} (score {score:.3f}). "
        + (f"Key drivers: {', '.join(factors[:3])}." if factors
           else "Vulnerability driven by combined geospatial indicators.")
    )

    ai_summary = None
    if _llm_available() and vclass in ("HIGH", "CRITICAL"):
        comps = vuln_data.get("component_scores", {})
        user_msg = (
            f"Vulnerability assessment:\n"
            f"  Score: {score:.3f} → {vclass}\n"
            f"  Components: {json.dumps(comps, indent=2)}\n"
            f"  Dominant factors: {factors}\n"
            f"Explain the key vulnerability drivers in 2-3 sentences."
        )
        ai_summary = _call_llm(_VULN_SYSTEM, user_msg)

    return AgentEvidence(
        agent_name="VulnerabilityAnalyst",
        summary=ai_summary if ai_summary else fallback_summary,
        severity=vclass,
        key_factors=factors,
        metrics={
            "vulnerability_score": score,
            "vulnerability_class": vclass,
            "components":          vuln_data.get("component_scores", {}),
        },
        ai_assisted=bool(ai_summary),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CAPACITY ANALYST AGENT
# ─────────────────────────────────────────────────────────────────────────────

_CAPACITY_SYSTEM = (
    "You are the Capacity Analyst for PRAVAAH. "
    "You interpret structured carrying-capacity data. "
    "Shelter capacity is UNAVAILABLE unless explicitly provided — never fabricate it. "
    "Respond in 2-3 sentences. Identify capacity constraints."
)


def run_capacity_agent(capacity_data: dict) -> AgentEvidence:
    """Interpret carrying capacity for one habitation."""
    status = capacity_data["capacity_status"]
    safe_area = capacity_data["safe_area_km2"]
    road = capacity_data["nearest_road"]
    health = capacity_data["nearest_healthcare"]

    # Severity from capacity status
    severity_map = {"CRITICAL": "CRITICAL", "STRESSED": "HIGH", "ADEQUATE": "LOW"}
    severity = severity_map.get(status, "MEDIUM")

    factors = []
    if safe_area < 0.5:
        factors.append(f"Very limited safe area ({safe_area:.2f} km²)")
    elif safe_area < 2.0:
        factors.append(f"Moderate safe area ({safe_area:.2f} km²)")
    if "not found" in road.lower():
        factors.append("No major road found in search area")
    if "not found" in health.lower():
        factors.append("No healthcare facility found in search area")

    fallback_summary = (
        f"Carrying capacity: {status} (score {capacity_data['capacity_score']:.3f}). "
        f"Safe area: {safe_area:.2f} km². "
        f"Nearest road: {road}. Nearest healthcare: {health}."
    )

    ai_summary = None
    if _llm_available() and status in ("CRITICAL", "STRESSED"):
        user_msg = (
            f"Capacity assessment:\n"
            f"  Status: {status} (score {capacity_data['capacity_score']:.3f})\n"
            f"  Safe area within {capacity_data['search_radius_km']:.0f}km: {safe_area:.2f} km²\n"
            f"  Nearest road: {road}\n"
            f"  Nearest healthcare: {health}\n"
            f"  Shelter capacity: {capacity_data.get('shelter_capacity', 'unavailable')}\n"
            f"Describe the capacity constraints in 2-3 sentences."
        )
        ai_summary = _call_llm(_CAPACITY_SYSTEM, user_msg)

    return AgentEvidence(
        agent_name="CapacityAnalyst",
        summary=ai_summary if ai_summary else fallback_summary,
        severity=severity,
        key_factors=factors,
        metrics={
            "capacity_status": status,
            "capacity_score":  capacity_data["capacity_score"],
            "safe_area_km2":   safe_area,
            "nearest_road":    road,
            "nearest_healthcare": health,
        },
        ai_assisted=bool(ai_summary),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RELOCATION PLANNER AGENT
# ─────────────────────────────────────────────────────────────────────────────

_RELOCATION_SYSTEM = (
    "You are the Relocation Planner for PRAVAAH, a decision-support system. "
    "You synthesise hazard, exposure, vulnerability, and capacity evidence to "
    "recommend relocation planning actions. "
    "IMPORTANT: You provide decision-support recommendations, NOT official evacuation orders. "
    "Always use language like 'recommended for review', 'priority for consideration', "
    "'potential relocation candidate', NOT 'mandatory evacuation' or 'official order'. "
    "Respond in 3-4 sentences. Cite the key evidence. "
    "If candidates are available, identify the best one with its key advantage."
)


def run_relocation_agent(
    relocation_data: dict,
    all_evidence: list[AgentEvidence],
    candidates: list[dict],
    comparison: dict,
) -> AgentEvidence:
    """
    Synthesise all agent evidence and recommend relocation planning actions.

    Parameters
    ----------
    relocation_data : dict
        Output of tools.get_relocation_details().
    all_evidence : list[AgentEvidence]
        Evidence from Hazard, Exposure, Vulnerability, Capacity agents.
    candidates : list[dict]
        Candidate summaries from tools.find_relocation_candidates_tool().
    comparison : dict
        Output of tools.compare_relocation_candidates_tool().
    """
    priority = relocation_data["priority_class"]
    score    = relocation_data["relocation_score"]
    factors  = relocation_data.get("contributing_factors", [])
    action   = relocation_data["recommended_action"]

    # Build deterministic summary from evidence
    evidence_bullets = [f"• {e.summary}" for e in all_evidence]
    candidate_text = comparison.get("comparison_narrative", "No candidates found.")
    top_cand = comparison.get("best_candidate_id")

    fallback_summary = (
        f"Priority: {priority} (relocation score {score:.3f}). "
        f"{action} "
        + (f"Recommended candidate area: {top_cand}. {candidate_text}"
           if top_cand else "No suitable relocation candidate areas found in search radius.")
    )

    top_cand_reason = ""
    if candidates:
        best = candidates[0]
        parts = []
        if best.get("candidate_score", 0) > 0.6:
            parts.append(f"high candidate quality score ({best['candidate_score']:.3f})")
        if best.get("area_km2", 0) > 1.0:
            parts.append(f"adequate safe area ({best['area_km2']:.2f} km²)")
        if best.get("distance_km", 99) < 5.0:
            parts.append(f"close proximity ({best['distance_km']:.1f} km)")
        top_cand_reason = (
            f"Recommended because: {', '.join(parts)}." if parts
            else "Selected as the highest-scoring candidate in the search area."
        )

    ai_summary = None
    if _llm_available() and priority in ("HIGH", "CRITICAL"):
        user_msg = (
            f"Relocation priority assessment for '{relocation_data.get('name', 'Unknown')}':\n"
            f"  Priority: {priority} (score {score:.3f})\n"
            f"  Key factors: {factors[:4]}\n"
            f"  Candidate areas found: {len(candidates)}\n"
            f"  Best candidate: {top_cand}\n"
            f"  Comparison: {candidate_text}\n\n"
            f"Evidence from specialist agents:\n"
            + "\n".join(evidence_bullets[:4]) +
            f"\n\nRecommend action in 3-4 sentences. "
            f"Use language appropriate for a decision-support system."
        )
        ai_summary = _call_llm(_RELOCATION_SYSTEM, user_msg)

    return AgentEvidence(
        agent_name="RelocationPlanner",
        summary=ai_summary if ai_summary else fallback_summary,
        severity=priority,
        key_factors=factors[:4],
        metrics={
            "priority_class":     priority,
            "relocation_score":   score,
            "candidates_found":   len(candidates),
            "best_candidate":     top_cand,
        },
        ai_assisted=bool(ai_summary),
    )
