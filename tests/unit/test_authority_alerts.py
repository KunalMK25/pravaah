"""Tests for Phase 3 authority alert generation."""
import pytest

from flood_risk_zonation.alerts.generation import (
    generate_authority_alerts,
    _classify_authority_category,
)
from flood_risk_zonation.models import RelocationPriorityResult


class TestAuthorityAlertGeneration:
    def test_no_alerts_for_low_priority(self):
        """LOW priority → no alert generated"""
        result = RelocationPriorityResult(
            hab_id="h1",
            name="Settlement A",
            relocation_score=0.1,
            priority_class="LOW",
            recommended_action="Monitor",
            time_horizon="LONG-TERM",
        )
        alerts = generate_authority_alerts([result])
        assert len(alerts) == 0

    def test_alert_generated_for_high_priority(self):
        """HIGH priority → alert generated"""
        result = RelocationPriorityResult(
            hab_id="h1",
            name="Settlement A",
            relocation_score=0.7,
            priority_class="HIGH",
            recommended_action="Action required",
            time_horizon="MEDIUM-TERM",
            population_exposed=300,
        )
        alerts = generate_authority_alerts([result])
        assert len(alerts) == 1
        assert alerts[0].severity == "HIGH"
        assert alerts[0].affected_area == "Settlement A"

    def test_alert_generated_for_critical_priority(self):
        """CRITICAL priority → alert generated with CRITICAL severity"""
        result = RelocationPriorityResult(
            hab_id="h1",
            name="Settlement A",
            relocation_score=0.9,
            priority_class="CRITICAL",
            recommended_action="Immediate action",
            time_horizon="SHORT-TERM",
            population_exposed=1000,
            hazard_score=95,
        )
        alerts = generate_authority_alerts([result])
        assert len(alerts) == 1
        assert alerts[0].severity == "CRITICAL"
        assert "IMMEDIATE" in alerts[0].recommended_action

    def test_alert_contains_evidence(self):
        """Alert contains supporting evidence data"""
        result = RelocationPriorityResult(
            hab_id="h1",
            name="Settlement A",
            relocation_score=0.8,
            priority_class="HIGH",
            recommended_action="Action",
            time_horizon="MEDIUM-TERM",
            hazard_score=80,
            vulnerability_score=0.7,
            capacity_score=0.3,
            population_exposed=500,
        )
        alerts = generate_authority_alerts([result])
        assert len(alerts) == 1
        assert alerts[0].evidence["hazard_score"] == 80
        assert alerts[0].evidence["vulnerability_score"] == 0.7
        assert alerts[0].evidence["capacity_score"] == 0.3

    def test_alert_contains_relocation_horizon(self):
        """Alert includes time horizon from relocation result"""
        result = RelocationPriorityResult(
            hab_id="h1",
            name="Settlement A",
            relocation_score=0.9,
            priority_class="CRITICAL",
            recommended_action="Immediate",
            time_horizon="SHORT-TERM",
        )
        alerts = generate_authority_alerts([result])
        assert alerts[0].relocation_horizon == "SHORT-TERM"

    def test_coastal_flag_in_evidence(self):
        """Coastal flag is captured in alert evidence"""
        result = RelocationPriorityResult(
            hab_id="h1",
            name="Coastal Settlement",
            relocation_score=0.7,
            priority_class="HIGH",
            recommended_action="Action",
            time_horizon="SHORT-TERM",
            is_coastal=True,
        )
        alerts = generate_authority_alerts([result])
        assert alerts[0].evidence["is_coastal"] is True
        assert "Coastal" in alerts[0].triggering_condition

    def test_alert_id_is_unique(self):
        """Different settlements get different alert IDs"""
        result1 = RelocationPriorityResult(
            hab_id="h1",
            name="Settlement A",
            relocation_score=0.7,
            priority_class="HIGH",
            recommended_action="Action",
            time_horizon="MEDIUM-TERM",
        )
        result2 = RelocationPriorityResult(
            hab_id="h2",
            name="Settlement B",
            relocation_score=0.7,
            priority_class="HIGH",
            recommended_action="Action",
            time_horizon="MEDIUM-TERM",
        )
        alerts = generate_authority_alerts([result1, result2])
        assert alerts[0].alert_id != alerts[1].alert_id

    def test_generated_at_timestamp(self):
        """Alert contains ISO timestamp"""
        result = RelocationPriorityResult(
            hab_id="h1",
            name="Settlement A",
            relocation_score=0.7,
            priority_class="HIGH",
            recommended_action="Action",
            time_horizon="MEDIUM-TERM",
        )
        alerts = generate_authority_alerts([result])
        assert len(alerts[0].generated_at) > 10
        assert "T" in alerts[0].generated_at  # ISO format includes T


class TestAuthorityCategoryClassification:
    def test_national_multiple_critical(self):
        """Multiple CRITICAL → NATIONAL authority"""
        category = _classify_authority_category(
            "CRITICAL", affected_population=200, is_coastal=False, num_critical_total=3
        )
        assert category == "NATIONAL"

    def test_national_large_population(self):
        """>5000 affected → NATIONAL authority"""
        category = _classify_authority_category(
            "CRITICAL", affected_population=6000, is_coastal=False, num_critical_total=1
        )
        assert category == "NATIONAL"

    def test_specialized_coastal_critical(self):
        """Coastal + CRITICAL → SPECIALIZED"""
        category = _classify_authority_category(
            "CRITICAL", affected_population=300, is_coastal=True, num_critical_total=1
        )
        assert category == "SPECIALIZED"

    def test_regional_critical_large_pop(self):
        """CRITICAL + >500 population → REGIONAL"""
        category = _classify_authority_category(
            "CRITICAL", affected_population=600, is_coastal=False, num_critical_total=1
        )
        assert category == "REGIONAL"

    def test_local_default(self):
        """Default → LOCAL authority"""
        category = _classify_authority_category(
            "HIGH", affected_population=100, is_coastal=False, num_critical_total=0
        )
        assert category == "LOCAL"
