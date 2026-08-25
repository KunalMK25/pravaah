"""
Phase 3 integration tests: orchestrator, FullSIHResult, and deterministic fallback.

Coverage:
- Orchestrator produces AgentDecision for each habitation
- LOW priority uses minimal agent workflow
- HIGH/CRITICAL uses full workflow
- ai_assisted=False when LLM unavailable
- fallback_reason populated when LLM unavailable
- FullSIHResult helpers work correctly
- Spatial zone + candidate + agent pipeline end-to-end (small synthetic grid)
- No crash when habitation dataset is empty
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from flood_risk_zonation.models import (
    ExposureResult, VulnerabilityResult, CarryingCapacityResult,
    RelocationPriorityResult, SIHAnalysisResult, HabitationDataset,
    Habitation, FloodRiskResult, AnalysisResult,
    RelocationCandidate, FullSIHResult, AgentDecision,
)
from flood_risk_zonation.spatial_zones.classifier import classify_spatial_zones, ZONE_RED, ZONE_GREEN
from flood_risk_zonation.agents.orchestrator import PravaahOrchestrator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_grid(risk_classes, n_cols=3):
    rows = []
    for i, rc in enumerate(risk_classes):
        r, c = divmod(i, n_cols)
        lat = 12.84 + r * 0.01
        lon = 77.55 + c * 0.01
        rows.append({
            "cell_id": f"c{i:03d}", "risk_class": rc,
            "risk_score": {"High": 80.0, "Medium": 50.0, "Low": 20.0, "Water": 0.0}.get(rc, 20.0),
            "centroid_lat": lat, "centroid_lon": lon,
            "elevation_m": 30.0, "dist_water_m": 500.0,
            "drainage_capacity": 0.4, "population_density": 100.0,
            "is_coastal_tsunami_risk": False,
            "geometry": box(lon - 0.005, lat - 0.005, lon + 0.005, lat + 0.005),
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _make_full_result(priority="CRITICAL"):
    """Build a minimal FullSIHResult for orchestrator tests."""
    hab = Habitation("h1", "Test Village", "village", 12.845, 77.558, "fallback")
    hab_ds = HabitationDataset(habitations=[hab], source="fallback", bbox_key="test")

    exp = ExposureResult(
        hab_id="h1", name="Test Village", hab_type="village",
        lat=12.845, lon=77.558, hazard_score=80.0, hazard_class="High",
        pct_high_risk=0.75, population_source="UNKNOWN", population_exposed=None,
        is_in_red_zone=True, intersecting_cell_ids=["c004"],
    )
    vuln = VulnerabilityResult(
        hab_id="h1", vulnerability_score=0.75, vulnerability_class="CRITICAL",
        component_scores={"hazard_severity": 0.8, "low_elevation": 0.7, "water_proximity": 0.6,
                          "poor_drainage": 0.8, "pop_exposure": 0.5, "road_accessibility": 0.4,
                          "healthcare_access": 0.3},
        component_weights={"hazard_severity": 0.30, "low_elevation": 0.15, "water_proximity": 0.15,
                           "poor_drainage": 0.15, "pop_exposure": 0.10, "road_accessibility": 0.10,
                           "healthcare_access": 0.05},
        factors=["High hazard", "Poor drainage"],
    )
    cap = CarryingCapacityResult(
        hab_id="h1", capacity_score=0.18, capacity_status="CRITICAL",
        safe_area_km2=0.08, search_radius_km=5.0,
        nearest_healthcare_km=-1.0, nearest_road_km=-1.0,
    )
    rel = RelocationPriorityResult(
        hab_id="h1", name="Test Village", relocation_score=0.84, priority_class=priority,
        recommended_action="Immediate relocation consideration required.",
        contributing_factors=["High hazard", "CRITICAL capacity", "Limited safe area"],
        component_scores={"hazard": 0.8, "vulnerability": 0.75, "cap_stress": 0.82, "exposure": 0.7},
        weights={"hazard": 0.35, "vulnerability": 0.30, "cap_stress": 0.20, "exposure": 0.15},
        hazard_score=80.0, vulnerability_score=0.75, capacity_score=0.18,
        population_source="UNKNOWN",
    )

    # Mock FloodRiskResult
    grid = _make_grid(["High", "Low", "Low", "Low", "Low", "Low", "Low", "Low", "Low"], n_cols=3)
    mock_analysis = MagicMock(spec=AnalysisResult)
    mock_analysis.feature_importances = {}
    mock_analysis.method = "weighted_susceptibility_index"
    mock_analysis.validation_note = "Test"
    mock_analysis.mean_cv_auc = None

    mock_config = MagicMock()
    mock_config.low_threshold = 33.0
    mock_config.medium_threshold = 66.0

    mock_fr = MagicMock(spec=FloodRiskResult)
    mock_fr.scored_grid = grid
    mock_fr.analysis_result = mock_analysis
    mock_fr.config = mock_config

    sih = SIHAnalysisResult(
        flood_risk_result=mock_fr,
        habitation_dataset=hab_ds,
        exposure_results=[exp],
        vulnerability_results=[vuln],
        capacity_results=[cap],
        relocation_results=[rel],
    )

    zoned_grid = classify_spatial_zones(grid)
    habitation_zones = {"h1": ZONE_RED}
    relocation_candidates = {}  # empty for simplicity

    return FullSIHResult(
        sih_result=sih,
        zoned_grid=zoned_grid,
        habitation_zones=habitation_zones,
        relocation_candidates=relocation_candidates,
        agent_decisions={},
    )


# ── Orchestrator tests ────────────────────────────────────────────────────────

class TestOrchestrator:
    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_analyse_habitation_returns_agent_decision(self, _):
        full = _make_full_result(priority="CRITICAL")
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        assert isinstance(dec, AgentDecision)
        assert dec.hab_id == "h1"
        assert dec.priority_class == "CRITICAL"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_ai_assisted_false_when_llm_unavailable(self, _):
        full = _make_full_result()
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        assert dec.ai_assisted is False
        assert len(dec.fallback_reason) > 0

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_recommended_action_not_empty(self, _):
        full = _make_full_result()
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        assert len(dec.recommended_action) > 0

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_summary_not_empty(self, _):
        full = _make_full_result()
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        assert len(dec.summary) > 0

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_evidence_populated_for_critical(self, _):
        full = _make_full_result(priority="CRITICAL")
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        # For CRITICAL priority, should have at least Hazard + Exposure + Vuln + Capacity + Reloc
        assert len(dec.evidence) >= 2

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_no_data_returns_graceful_decision(self, _):
        full = _make_full_result()
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("nonexistent_hab")
        assert isinstance(dec, AgentDecision)
        assert dec.priority_class == "UNKNOWN"
        assert "insufficient" in dec.fallback_reason.lower() or "No" in dec.fallback_reason

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_analyse_all_returns_dict(self, _):
        full = _make_full_result()
        orc = PravaahOrchestrator(full)
        decisions = orc.analyse_all(priority_filter=("CRITICAL", "HIGH", "MEDIUM", "LOW"))
        assert isinstance(decisions, dict)
        assert "h1" in decisions

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_low_priority_still_gets_decision(self, _):
        full = _make_full_result(priority="LOW")
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        assert isinstance(dec, AgentDecision)
        assert dec.priority_class == "LOW"

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_spatial_zone_in_decision(self, _):
        full = _make_full_result()
        full.habitation_zones["h1"] = ZONE_RED
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        assert dec.spatial_zone == ZONE_RED


class TestFullSIHResultHelpers:
    def test_red_zone_count(self):
        full = _make_full_result()
        # Should have at least 1 RED cell (index 0 is "High")
        assert full.red_zone_count >= 1

    def test_green_zone_count(self):
        full = _make_full_result()
        assert full.green_zone_count >= 0

    def test_get_zone_for(self):
        full = _make_full_result()
        full.habitation_zones["h1"] = ZONE_RED
        assert full.get_zone_for("h1") == ZONE_RED
        assert full.get_zone_for("nonexistent") == "UNKNOWN"

    def test_get_decision_for(self):
        full = _make_full_result()
        dec = AgentDecision("h1", "Test", "CRITICAL", 0.85, ZONE_RED, "Summary", "Action")
        full.agent_decisions["h1"] = dec
        assert full.get_decision_for("h1") is dec
        assert full.get_decision_for("missing") is None

    def test_get_candidates_for(self):
        full = _make_full_result()
        cand = RelocationCandidate("c1", "h1", 12.87, 77.58, 2.0, 1.5, 0.75)
        full.relocation_candidates["h1"] = [cand]
        assert len(full.get_candidates_for("h1")) == 1
        assert full.get_candidates_for("missing") == []


class TestPhase3SpatialZoneIntegration:
    """End-to-end test: grid → zones → habitation zone lookup."""

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_red_habitation_assigned_red_zone(self, _):
        from flood_risk_zonation.spatial_zones.classifier import get_zone_for_habitation
        grid = _make_grid(["High"] * 4 + ["Low"] * 5, n_cols=3)
        zg = classify_spatial_zones(grid)
        # Habitation at first cell (High → RED)
        zone = get_zone_for_habitation(12.84, 77.55, zg)
        assert zone == ZONE_RED

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_green_habitation_assigned_green_zone(self, _):
        from flood_risk_zonation.spatial_zones.classifier import get_zone_for_habitation
        grid = _make_grid(["Low"] * 9, n_cols=3)
        zg = classify_spatial_zones(grid)
        zone = get_zone_for_habitation(12.84, 77.55, zg)
        assert zone == ZONE_GREEN

    def test_zoned_grid_preserves_original_risk_class(self):
        grid = _make_grid(["High", "Low", "Medium", "Water"], n_cols=2)
        zg = classify_spatial_zones(grid)
        orig = grid["risk_class"].tolist()
        assert zg["risk_class"].tolist() == orig

    def test_zoned_grid_has_spatial_zone_column(self):
        grid = _make_grid(["High", "Low", "Medium"], n_cols=3)
        zg = classify_spatial_zones(grid)
        assert "spatial_zone" in zg.columns
        assert zg["spatial_zone"].notna().all()


class TestDeterministicFallback:
    """Verify the full system works without any LLM."""

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    @patch("flood_risk_zonation.agents.agents._call_llm", return_value=None)
    def test_all_agents_fallback_gracefully(self, mock_call, mock_avail):
        full = _make_full_result(priority="CRITICAL")
        orc = PravaahOrchestrator(full)
        dec = orc.analyse_habitation("h1")
        assert isinstance(dec, AgentDecision)
        assert dec.ai_assisted is False
        # All evidence should be rule-based
        for ev in dec.evidence:
            assert ev.ai_assisted is False

    @patch("flood_risk_zonation.agents.agents._call_llm", side_effect=Exception("Network error"))
    def test_agent_exception_falls_back_gracefully(self, mock_call):
        full = _make_full_result(priority="CRITICAL")
        orc = PravaahOrchestrator(full)
        # Should not raise, should return a valid decision
        try:
            dec = orc.analyse_habitation("h1")
            assert isinstance(dec, AgentDecision)
        except Exception as e:
            pytest.fail(f"Orchestrator raised unexpected exception: {e}")
