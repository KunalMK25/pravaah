"""Unit tests for relocation priority scorer."""
import pytest
from flood_risk_zonation.models import (
    ExposureResult,
    VulnerabilityResult,
    CarryingCapacityResult,
)
from flood_risk_zonation.relocation.priority import (
    score_relocation_priority,
    RELOCATION_WEIGHTS,
)


def _make_inputs(
    hazard_score=70.0,
    hazard_class="High",
    vuln_score=0.7,
    vuln_class="HIGH",
    cap_score=0.3,
    cap_status="CRITICAL",
    pop_source="UNKNOWN",
    pop_exposed=None,
    is_coastal=False,
):
    exp = ExposureResult(
        hab_id="h1",
        name="Test Hab",
        hab_type="village",
        lat=12.9,
        lon=77.6,
        hazard_score=hazard_score,
        hazard_class=hazard_class,
        pct_high_risk=0.6,
        population_source=pop_source,
        population_exposed=pop_exposed,
        is_in_red_zone=(hazard_class == "High"),
    )
    vuln = VulnerabilityResult(
        hab_id="h1",
        vulnerability_score=vuln_score,
        vulnerability_class=vuln_class,
    )
    cap = CarryingCapacityResult(
        hab_id="h1",
        capacity_score=cap_score,
        capacity_status=cap_status,
        safe_area_km2=0.2,
        search_radius_km=5.0,
        nearest_healthcare_km=15.0,
        nearest_road_km=4.0,
    )
    return exp, vuln, cap, is_coastal


class TestWeights:
    def test_sum_to_one(self):
        assert abs(sum(RELOCATION_WEIGHTS.values()) - 1.0) < 1e-9


class TestRelocationPriority:
    def test_high_hazard_high_vuln_critical_cap_is_critical(self):
        exp, vuln, cap, coastal = _make_inputs(
            hazard_score=90.0, hazard_class="High",
            vuln_score=0.8, cap_score=0.1, cap_status="CRITICAL",
        )
        result = score_relocation_priority(exp, vuln, cap, is_coastal=coastal)
        assert result.priority_class == "CRITICAL"

    def test_low_hazard_good_cap_is_low(self):
        exp, vuln, cap, coastal = _make_inputs(
            hazard_score=10.0, hazard_class="Low",
            vuln_score=0.1, vuln_class="LOW",
            cap_score=0.9, cap_status="ADEQUATE",
        )
        result = score_relocation_priority(exp, vuln, cap)
        assert result.priority_class in ("LOW", "MEDIUM")

    def test_score_in_range(self):
        exp, vuln, cap, coastal = _make_inputs()
        result = score_relocation_priority(exp, vuln, cap)
        assert 0.0 <= result.relocation_score <= 1.0

    def test_coastal_escalation(self):
        exp, vuln, cap, _ = _make_inputs(
            hazard_score=70.0, hazard_class="High",
            vuln_score=0.6, cap_score=0.4, cap_status="STRESSED",
        )
        result_normal = score_relocation_priority(exp, vuln, cap, is_coastal=False)
        result_coastal = score_relocation_priority(exp, vuln, cap, is_coastal=True)
        # Coastal should be >= normal
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        assert order[result_coastal.priority_class] >= order[result_normal.priority_class]

    def test_capacity_critical_escalation(self):
        # Same inputs but cap_status CRITICAL should boost score
        exp, vuln, cap_ok, _ = _make_inputs(cap_score=0.7, cap_status="ADEQUATE")
        exp, vuln, cap_crit, _ = _make_inputs(cap_score=0.1, cap_status="CRITICAL")
        r_ok = score_relocation_priority(exp, vuln, cap_ok)
        r_crit = score_relocation_priority(exp, vuln, cap_crit)
        assert r_crit.relocation_score > r_ok.relocation_score

    def test_known_pop_high_hazard_increases_exposure_component(self):
        exp_pop, vuln, cap, _ = _make_inputs(
            hazard_class="High", pop_source="osm_tag", pop_exposed=3000
        )
        exp_nopop, _, _, _ = _make_inputs(
            hazard_class="High", pop_source="UNKNOWN"
        )
        r_pop = score_relocation_priority(exp_pop, vuln, cap)
        r_nopop = score_relocation_priority(exp_nopop, vuln, cap)
        assert r_pop.component_scores["exposure"] >= r_nopop.component_scores["exposure"]

    def test_contributing_factors_not_empty(self):
        exp, vuln, cap, coastal = _make_inputs(
            hazard_score=80.0, hazard_class="High",
            vuln_score=0.7, cap_score=0.2, cap_status="CRITICAL",
        )
        result = score_relocation_priority(exp, vuln, cap)
        assert len(result.contributing_factors) > 0

    def test_explanation_contains_key_info(self):
        exp, vuln, cap, _ = _make_inputs()
        result = score_relocation_priority(exp, vuln, cap)
        assert "WHY" in result.explanation
        assert "Hazard" in result.explanation
        assert "Vulnerability" in result.explanation

    def test_water_hab_not_critical(self):
        exp, vuln, cap, _ = _make_inputs(
            hazard_score=0.0, hazard_class="Water",
            vuln_score=0.9, cap_score=0.0, cap_status="CRITICAL",
        )
        result = score_relocation_priority(exp, vuln, cap)
        assert result.priority_class != "CRITICAL"

    def test_recommended_action_not_empty(self):
        exp, vuln, cap, _ = _make_inputs()
        result = score_relocation_priority(exp, vuln, cap)
        assert len(result.recommended_action) > 0

    def test_returns_correct_hab_id(self):
        exp, vuln, cap, _ = _make_inputs()
        result = score_relocation_priority(exp, vuln, cap)
        assert result.hab_id == "h1"

    def test_unknown_pop_non_high_hazard_capped(self):
        """Unknown population + medium hazard should not be CRITICAL."""
        exp, vuln, cap, _ = _make_inputs(
            hazard_score=40.0, hazard_class="Medium",
            pop_source="UNKNOWN",
            vuln_score=0.8, cap_score=0.0, cap_status="CRITICAL",
        )
        result = score_relocation_priority(exp, vuln, cap)
        assert result.priority_class != "CRITICAL"
