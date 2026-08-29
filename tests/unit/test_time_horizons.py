"""Tests for Phase 2 time horizon classification."""
import pytest

from flood_risk_zonation.relocation.priority import _classify_time_horizon


class TestTimeHorizonClassification:
    def test_short_term_critical_priority(self):
        """CRITICAL priority → SHORT-TERM"""
        horizon, explanation = _classify_time_horizon(
            "CRITICAL", is_coastal=False, hazard_score=90, pct_high_risk=0.8
        )
        assert horizon == "SHORT-TERM"
        assert "immediate" in explanation.lower()

    def test_short_term_coastal_high(self):
        """Coastal + HIGH → SHORT-TERM"""
        horizon, explanation = _classify_time_horizon(
            "HIGH", is_coastal=True, hazard_score=75, pct_high_risk=0.6
        )
        assert horizon == "SHORT-TERM"
        assert "coastal" in explanation.lower()

    def test_medium_term_high_priority(self):
        """HIGH priority (non-coastal) → MEDIUM-TERM"""
        horizon, explanation = _classify_time_horizon(
            "HIGH", is_coastal=False, hazard_score=75, pct_high_risk=0.5
        )
        assert horizon == "MEDIUM-TERM"
        assert "3–12" in explanation or "3 to 12" in explanation

    def test_medium_term_medium_high_hazard(self):
        """MEDIUM priority + high hazard → MEDIUM-TERM"""
        horizon, explanation = _classify_time_horizon(
            "MEDIUM", is_coastal=False, hazard_score=80, pct_high_risk=0.4
        )
        assert horizon == "MEDIUM-TERM"

    def test_long_term_medium_priority(self):
        """MEDIUM priority + moderate hazard → LONG-TERM"""
        horizon, explanation = _classify_time_horizon(
            "MEDIUM", is_coastal=False, hazard_score=60, pct_high_risk=0.3
        )
        assert horizon == "LONG-TERM"
        assert "1–5" in explanation or "1 to 5" in explanation

    def test_long_term_low_priority(self):
        """LOW priority → LONG-TERM"""
        horizon, explanation = _classify_time_horizon(
            "LOW", is_coastal=False, hazard_score=30, pct_high_risk=0.1
        )
        assert horizon == "LONG-TERM"
        assert "routine" in explanation.lower()

    def test_long_term_low_priority_moderate_hazard(self):
        """LOW priority + moderate hazard (>50) → LONG-TERM"""
        horizon, explanation = _classify_time_horizon(
            "LOW", is_coastal=False, hazard_score=55, pct_high_risk=0.2
        )
        assert horizon == "LONG-TERM"

    def test_explanation_is_non_empty(self):
        """All classifications produce explanations"""
        for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            for is_coastal in [True, False]:
                horizon, explanation = _classify_time_horizon(
                    priority, is_coastal=is_coastal, hazard_score=50, pct_high_risk=0.3
                )
                assert horizon in ("SHORT-TERM", "MEDIUM-TERM", "LONG-TERM")
                assert len(explanation) > 10
                assert "term" in explanation.lower()
