"""
PRAVAAH-AI — Authority Alert Generation

Converts relocation priority results into structured government authority alerts.

Authority Categories:
  LOCAL       — Ward/Municipal Commissioner (HIGH/CRITICAL with pop < 500)
  REGIONAL    — District/State authorities (CRITICAL with pop > 500)
  NATIONAL    — NDMA (Multiple CRITICAL or >5000 affected total)
  SPECIALIZED — Water Authority, Ministry of Housing (specific conditions)

Alert Severity:
  LOW      → Routine monitoring (priority LOW)
  MEDIUM   → Preparedness action (priority MEDIUM)
  HIGH     → Priority intervention (priority HIGH)
  CRITICAL → Immediate relocation (priority CRITICAL)
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime

from flood_risk_zonation.models import AuthorityAlert, RelocationPriorityResult

logger = logging.getLogger(__name__)


def _generate_alert_id(
    settlement_name: str,
    priority_class: str,
    timestamp: str,
) -> str:
    """Generate deterministic alert ID."""
    key = f"{settlement_name}-{priority_class}-{timestamp}"
    digest = hashlib.md5(key.encode()).hexdigest()[:6]
    return f"PRAVAAH-{timestamp[:10].replace('-', '')}-{digest}"


def _classify_authority_category(
    priority_class: str,
    affected_population: int,
    is_coastal: bool,
    num_critical_total: int = 1,
) -> str:
    """Determine authority category based on severity and scale."""
    # NATIONAL: Multiple CRITICAL or >5000 affected
    if num_critical_total > 2 or affected_population > 5000:
        return "NATIONAL"
    # SPECIALIZED: Coastal requires water authority involvement
    if is_coastal and priority_class in ("HIGH", "CRITICAL"):
        return "SPECIALIZED"
    # REGIONAL: CRITICAL with significant population
    if priority_class == "CRITICAL" and affected_population > 500:
        return "REGIONAL"
    # LOCAL: Everything else
    return "LOCAL"


def generate_authority_alerts(
    relocation_results: list[RelocationPriorityResult],
) -> list[AuthorityAlert]:
    """
    Generate structured authority alerts from relocation results.

    Parameters
    ----------
    relocation_results : list[RelocationPriorityResult]
        Relocation priority assessments for all habitations.

    Returns
    -------
    list[AuthorityAlert]
        Structured alerts for authorities.
    """
    alerts: list[AuthorityAlert] = []
    timestamp = datetime.utcnow().isoformat()
    
    # Count total CRITICAL for NATIONAL escalation
    critical_count = sum(
        1 for r in relocation_results if r.priority_class == "CRITICAL"
    )
    total_critical_population = sum(
        r.population_exposed or 0
        for r in relocation_results
        if r.priority_class == "CRITICAL"
    )

    for result in relocation_results:
        # Generate alert only for HIGH and CRITICAL priority
        if result.priority_class not in ("HIGH", "CRITICAL"):
            continue

        # Map priority class to alert severity
        severity_map = {
            "HIGH": "HIGH",
            "CRITICAL": "CRITICAL",
        }
        severity = severity_map.get(result.priority_class, "MEDIUM")

        # Determine authority category
        authority_category = _classify_authority_category(
            result.priority_class,
            result.population_exposed or 0,
            result.is_coastal,
            num_critical_total=critical_count,
        )

        # Generate triggering condition narrative
        triggers = []
        if result.priority_class == "CRITICAL":
            triggers.append("CRITICAL relocation priority")
        elif result.priority_class == "HIGH":
            triggers.append("HIGH relocation priority")

        if result.hazard_score >= 80:
            triggers.append(f"High hazard score ({result.hazard_score:.1f}/100)")
        if result.population_exposed:
            triggers.append(f"{result.population_exposed:,} people potentially affected")
        if result.is_coastal:
            triggers.append("Coastal/tsunami inundation risk")
        if result.capacity_score < 0.3:
            triggers.append("Limited carrying capacity (<0.3 score)")

        triggering_condition = "; ".join(triggers)

        # Generate recommended action
        action_map = {
            "CRITICAL": (
                "IMMEDIATE ACTION REQUIRED: Initiate emergency evacuation procedures. "
                "Pre-position relief materials and coordinate with local emergency services."
            ),
            "HIGH": (
                "Priority Action: Develop evacuation plan, identify shelters, and brief community. "
                "Activate early warning system."
            ),
        }
        recommended_action = action_map.get(result.priority_class, "Monitor situation")

        # Build evidence dictionary
        evidence = {
            "hazard_score": result.hazard_score,
            "priority_class": result.priority_class,
            "relocation_score": result.relocation_score,
            "vulnerability_score": result.vulnerability_score,
            "capacity_score": result.capacity_score,
            "population_exposed": result.population_exposed,
            "population_source": result.population_source,
            "is_coastal": result.is_coastal,
            "capacity_status": (
                "CRITICAL" if result.capacity_score < 0.2 
                else "STRESSED" if result.capacity_score < 0.5 
                else "ADEQUATE"
            ),
            "component_scores": result.component_scores,
        }

        # Generate alert ID
        alert_id = _generate_alert_id(result.name, result.priority_class, timestamp)

        alert = AuthorityAlert(
            alert_id=alert_id,
            severity=severity,
            affected_area=result.name,
            affected_population=result.population_exposed or 0,
            triggering_condition=triggering_condition,
            evidence=evidence,
            recommended_action=recommended_action,
            relocation_horizon=result.time_horizon,
            authority_category=authority_category,
            generated_at=timestamp,
        )

        alerts.append(alert)

        logger.info(
            "Generated authority alert [%s] for %s: severity=%s, authority=%s",
            alert_id,
            result.name,
            severity,
            authority_category,
        )

    # Log summary
    if alerts:
        critical_alerts = sum(1 for a in alerts if a.severity == "CRITICAL")
        high_alerts = sum(1 for a in alerts if a.severity == "HIGH")
        logger.info(
            "Authority alert generation complete: %d total | %d CRITICAL | %d HIGH",
            len(alerts),
            critical_alerts,
            high_alerts,
        )

    return alerts
