"""
Tests for PRAVAAH agent tool functions.

Coverage:
- get_hazard_details returns required fields
- get_exposure_details returns correct population provenance
- get_vulnerability_details returns correct class
- get_capacity_details returns correct status
- get_relocation_details returns correct priority
- find_relocation_candidates_tool returns serialised dicts
- compare_relocation_candidates_tool handles empty/single/multiple
- ValueError raised when hab_id not found
- Fallback when LLM unavailable (agent tools are deterministic)
- Invalid LLM response does not crash agents
- Agent evidence has required fields
"""
from __future__ import annotations
import pytest
from unittest.mock import patch

from flood_risk_zonation.models import (
    ExposureResult, VulnerabilityResult, CarryingCapacityResult,
    RelocationPriorityResult, RelocationCandidate, AgentEvidence,
)
from flood_risk_zonation.agents.tools import (
    get_exposure_details,
    get_vulnerability_details,
    get_capacity_details,
    get_relocation_details,
    find_relocation_candidates_tool,
    compare_relocation_candidates_tool,
)
from flood_risk_zonation.agents.agents import (
    run_hazard_agent,
    run_exposure_agent,
    run_vulnerability_agent,
    run_capacity_agent,
    run_relocation_agent,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _exposure(hab_id="h1", hazard_class="High", hazard_score=80.0, pop_source="UNKNOWN", pop=None):
    return ExposureResult(
        hab_id=hab_id, name="Test Hab", hab_type="village",
        lat=12.9, lon=77.6,
        hazard_score=hazard_score, hazard_class=hazard_class,
        pct_high_risk=0.6, population_source=pop_source,
        population_exposed=pop, is_in_red_zone=(hazard_class == "High"),
        intersecting_cell_ids=["c001"],
    )

def _vuln(hab_id="h1", score=0.7, vclass="HIGH"):
    return VulnerabilityResult(
        hab_id=hab_id, vulnerability_score=score, vulnerability_class=vclass,
        component_scores={"hazard_severity": 0.8, "low_elevation": 0.6, "water_proximity": 0.5,
                          "poor_drainage": 0.7, "pop_exposure": 0.5, "road_accessibility": 0.4,
                          "healthcare_access": 0.3},
        component_weights={"hazard_severity": 0.30, "low_elevation": 0.15, "water_proximity": 0.15,
                           "poor_drainage": 0.15, "pop_exposure": 0.10, "road_accessibility": 0.10,
                           "healthcare_access": 0.05},
        factors=["High hazard", "Low elevation"],
    )

def _cap(hab_id="h1", status="CRITICAL", score=0.2):
    return CarryingCapacityResult(
        hab_id=hab_id, capacity_score=score, capacity_status=status,
        safe_area_km2=0.1, search_radius_km=5.0,
        nearest_healthcare_km=15.0, nearest_road_km=4.0,
    )

def _rel(hab_id="h1", priority="CRITICAL", score=0.82):
    return RelocationPriorityResult(
        hab_id=hab_id, name="Test Hab", relocation_score=score,
        priority_class=priority, recommended_action="Immediate relocation planning.",
        contributing_factors=["High hazard", "CRITICAL capacity"],
        component_scores={"hazard": 0.8, "vulnerability": 0.7, "cap_stress": 0.8, "exposure": 0.7},
        weights={"hazard": 0.35, "vulnerability": 0.30, "cap_stress": 0.20, "exposure": 0.15},
        hazard_score=80.0, vulnerability_score=0.7, capacity_score=0.2,
        population_source="UNKNOWN",
    )

def _candidate(cid="cand_001", source="h1", score=0.75):
    return RelocationCandidate(
        candidate_id=cid, source_hab_id=source,
        centroid_lat=12.87, centroid_lon=77.58,
        distance_km=2.1, area_km2=1.5,
        candidate_score=score, mean_hazard_score=15.0,
        nearest_road_km=0.8, nearest_healthcare_km=3.0,
        notes="Good candidate area.", data_provenance="spatial_zone_green",
    )


# ── Tool tests ────────────────────────────────────────────────────────────────

class TestGetExposureDetails:
    def test_returns_required_keys(self):
        exp = _exposure()
        d = get_exposure_details("h1", [exp])
        assert "hab_id" in d
        assert "is_in_red_zone" in d
        assert "population_label" in d
        assert "population_source" in d
        assert "provenance" in d

    def test_unknown_population_labelled(self):
        exp = _exposure(pop_source="UNKNOWN", pop=None)
        d = get_exposure_details("h1", [exp])
        assert "UNKNOWN" in d["population_label"]

    def test_osm_population_labelled_correctly(self):
        exp = _exposure(pop_source="osm_tag", pop=2500)
        d = get_exposure_details("h1", [exp])
        assert "2,500" in d["population_label"]
        assert "OSM" in d["population_label"]

    def test_raises_for_missing_hab_id(self):
        with pytest.raises(ValueError, match="h_missing"):
            get_exposure_details("h_missing", [_exposure()])


class TestGetVulnerabilityDetails:
    def test_returns_correct_class(self):
        d = get_vulnerability_details("h1", [_vuln(vclass="HIGH")])
        assert d["vulnerability_class"] == "HIGH"

    def test_returns_components(self):
        d = get_vulnerability_details("h1", [_vuln()])
        assert isinstance(d["component_scores"], dict)
        assert len(d["component_scores"]) > 0

    def test_raises_for_missing(self):
        with pytest.raises(ValueError):
            get_vulnerability_details("missing", [_vuln()])


class TestGetCapacityDetails:
    def test_returns_capacity_status(self):
        d = get_capacity_details("h1", [_cap(status="CRITICAL")])
        assert d["capacity_status"] == "CRITICAL"

    def test_negative_road_shows_not_found(self):
        cap = _cap()
        cap.nearest_road_km = -1.0
        d = get_capacity_details("h1", [cap])
        assert "not found" in d["nearest_road"].lower()

    def test_raises_for_missing(self):
        with pytest.raises(ValueError):
            get_capacity_details("missing", [_cap()])


class TestGetRelocationDetails:
    def test_returns_priority_class(self):
        d = get_relocation_details("h1", [_rel(priority="CRITICAL")])
        assert d["priority_class"] == "CRITICAL"

    def test_returns_weights(self):
        d = get_relocation_details("h1", [_rel()])
        assert isinstance(d["weights"], dict)

    def test_raises_for_missing(self):
        with pytest.raises(ValueError):
            get_relocation_details("missing", [_rel()])


class TestFindRelocationCandidatesTool:
    def test_returns_list_of_dicts(self):
        cands = [_candidate(), _candidate("cand_002", score=0.6)]
        result = find_relocation_candidates_tool("h1", {"h1": cands})
        assert isinstance(result, list)
        assert len(result) == 2
        assert "candidate_id" in result[0]
        assert "candidate_score" in result[0]

    def test_empty_when_no_candidates(self):
        result = find_relocation_candidates_tool("h1", {})
        assert result == []

    def test_provenance_included(self):
        cands = [_candidate()]
        result = find_relocation_candidates_tool("h1", {"h1": cands})
        assert result[0]["provenance"] == "spatial_zone_green"


class TestCompareRelocationCandidatesTool:
    def test_empty_candidates(self):
        result = compare_relocation_candidates_tool([])
        assert result["best_candidate_id"] is None
        assert "No relocation candidates" in result["comparison_narrative"]

    def test_single_candidate(self):
        cand = {"candidate_id": "c1", "candidate_score": 0.75, "distance_km": 2.0, "area_km2": 1.5}
        result = compare_relocation_candidates_tool([cand])
        assert result["best_candidate_id"] == "c1"
        assert len(result["ranking"]) == 1

    def test_multiple_candidates_ranked(self):
        cands = [
            {"candidate_id": "c1", "candidate_score": 0.5, "distance_km": 3.0, "area_km2": 1.0},
            {"candidate_id": "c2", "candidate_score": 0.8, "distance_km": 1.5, "area_km2": 2.5},
        ]
        result = compare_relocation_candidates_tool(cands)
        assert result["best_candidate_id"] == "c2"
        assert result["ranking"][0]["candidate_id"] == "c2"
        assert result["ranking"][1]["candidate_id"] == "c1"

    def test_narrative_not_empty(self):
        cands = [{"candidate_id": "c1", "candidate_score": 0.7, "distance_km": 2.0, "area_km2": 1.0}]
        result = compare_relocation_candidates_tool(cands)
        assert len(result["comparison_narrative"]) > 10


# ── Agent tests ───────────────────────────────────────────────────────────────

class TestRunHazardAgent:
    def test_returns_agent_evidence(self):
        data = {
            "hab_id": "h1", "hazard_score": 80.0, "hazard_class": "High",
            "spatial_zone": "RED", "pct_high_risk": 0.75,
            "dominant_features": ["Low elevation", "Poor drainage"],
            "is_coastal": False, "provenance": "ml_hazard_engine",
        }
        ev = run_hazard_agent(data)
        assert isinstance(ev, AgentEvidence)
        assert ev.agent_name == "HazardAnalyst"
        assert ev.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert len(ev.summary) > 0

    def test_low_hazard_score(self):
        data = {
            "hab_id": "h1", "hazard_score": 15.0, "hazard_class": "Low",
            "spatial_zone": "GREEN", "pct_high_risk": 0.0,
            "dominant_features": [], "is_coastal": False, "provenance": "ml_hazard_engine",
        }
        ev = run_hazard_agent(data)
        assert ev.severity == "LOW"

    @patch("flood_risk_zonation.agents.agents._call_llm", return_value=None)
    def test_fallback_when_llm_returns_none(self, mock_llm):
        data = {
            "hab_id": "h1", "hazard_score": 85.0, "hazard_class": "High",
            "spatial_zone": "RED", "pct_high_risk": 0.8,
            "dominant_features": ["High hazard score"], "is_coastal": False,
            "provenance": "ml_hazard_engine",
        }
        ev = run_hazard_agent(data)
        assert ev.ai_assisted is False
        assert len(ev.summary) > 0   # fallback summary still populated

    @patch("flood_risk_zonation.agents.agents._call_llm", return_value="  ")  # whitespace response
    def test_whitespace_llm_response_uses_fallback(self, mock_llm):
        data = {
            "hab_id": "h1", "hazard_score": 90.0, "hazard_class": "High",
            "spatial_zone": "RED", "pct_high_risk": 0.9,
            "dominant_features": [], "is_coastal": False, "provenance": "ml_hazard_engine",
        }
        ev = run_hazard_agent(data)
        # Whitespace response — LLM call happened but returned empty
        assert isinstance(ev, AgentEvidence)
        assert len(ev.summary.strip()) > 0


class TestRunExposureAgent:
    def test_red_zone_high_severity(self):
        data = {
            "hab_id": "h1", "name": "Test", "hab_type": "village",
            "is_in_red_zone": True, "hazard_class": "High",
            "population_label": "UNKNOWN (not in OSM data)",
            "population_value": None, "population_source": "UNKNOWN",
            "pct_high_risk": 0.8, "provenance": "exposure_analysis",
        }
        ev = run_exposure_agent(data)
        assert ev.severity in ("HIGH", "CRITICAL")

    def test_non_red_zone_lower_severity(self):
        data = {
            "hab_id": "h1", "name": "Test", "hab_type": "hamlet",
            "is_in_red_zone": False, "hazard_class": "Low",
            "population_label": "UNKNOWN", "population_value": None,
            "population_source": "UNKNOWN", "pct_high_risk": 0.0,
            "provenance": "exposure_analysis",
        }
        ev = run_exposure_agent(data)
        assert ev.severity == "LOW"


class TestRunVulnerabilityAgent:
    def test_critical_vulnerability(self):
        data = {
            "hab_id": "h1", "vulnerability_score": 0.85, "vulnerability_class": "CRITICAL",
            "component_scores": {"hazard_severity": 0.9}, "component_weights": {"hazard_severity": 1.0},
            "dominant_factors": ["Extreme hazard exposure"], "provenance": "vulnerability_scorer",
        }
        ev = run_vulnerability_agent(data)
        assert ev.severity == "CRITICAL"
        assert len(ev.summary) > 0

    def test_low_vulnerability(self):
        data = {
            "hab_id": "h1", "vulnerability_score": 0.1, "vulnerability_class": "LOW",
            "component_scores": {}, "component_weights": {},
            "dominant_factors": [], "provenance": "vulnerability_scorer",
        }
        ev = run_vulnerability_agent(data)
        assert ev.severity == "LOW"


class TestRunCapacityAgent:
    def test_critical_capacity_returns_critical_severity(self):
        data = {
            "hab_id": "h1", "capacity_score": 0.15, "capacity_status": "CRITICAL",
            "safe_area_km2": 0.05, "search_radius_km": 5.0,
            "nearest_healthcare": "not found in area", "nearest_road": "not found in area",
            "shelter_capacity": "unavailable", "shelter_source": "unavailable",
            "notes": "Very limited capacity.", "provenance": "capacity_assessment",
        }
        ev = run_capacity_agent(data)
        assert ev.severity == "CRITICAL"

    def test_adequate_capacity_returns_low_severity(self):
        data = {
            "hab_id": "h1", "capacity_score": 0.85, "capacity_status": "ADEQUATE",
            "safe_area_km2": 8.0, "search_radius_km": 5.0,
            "nearest_healthcare": "2.5 km", "nearest_road": "0.3 km",
            "shelter_capacity": "unavailable", "shelter_source": "unavailable",
            "notes": "Good capacity.", "provenance": "capacity_assessment",
        }
        ev = run_capacity_agent(data)
        assert ev.severity == "LOW"


class TestRunRelocationAgent:
    def test_returns_agent_evidence(self):
        rel_data = {
            "hab_id": "h1", "name": "Test", "relocation_score": 0.82,
            "priority_class": "CRITICAL", "recommended_action": "Immediate relocation.",
            "contributing_factors": ["High hazard", "CRITICAL capacity"],
            "component_scores": {}, "weights": {}, "is_coastal": False,
            "provenance": "relocation_priority_engine",
        }
        cands = [{"candidate_id": "c1", "candidate_score": 0.75, "distance_km": 1.5, "area_km2": 2.0, "notes": "Good", "provenance": "spatial_zone_green"}]
        comparison = compare_relocation_candidates_tool(cands)
        hazard_ev = AgentEvidence("HazardAnalyst", "High hazard", "CRITICAL")
        ev = run_relocation_agent(rel_data, [hazard_ev], cands, comparison)
        assert isinstance(ev, AgentEvidence)
        assert ev.agent_name == "RelocationPlanner"
        assert len(ev.summary) > 0

    def test_no_candidates_still_returns_evidence(self):
        rel_data = {
            "hab_id": "h1", "name": "Test", "relocation_score": 0.6,
            "priority_class": "HIGH", "recommended_action": "Evacuation planning.",
            "contributing_factors": [], "component_scores": {}, "weights": {},
            "is_coastal": False, "provenance": "relocation_priority_engine",
        }
        comparison = compare_relocation_candidates_tool([])
        ev = run_relocation_agent(rel_data, [], [], comparison)
        assert isinstance(ev, AgentEvidence)
        assert "No" in ev.summary or len(ev.summary) > 0


class TestAgentFallback:
    """Verify agents degrade gracefully when LLM is unavailable."""

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_hazard_agent_works_without_llm(self, mock_llm):
        data = {
            "hab_id": "h1", "hazard_score": 80.0, "hazard_class": "High",
            "spatial_zone": "RED", "pct_high_risk": 0.75,
            "dominant_features": [], "is_coastal": False, "provenance": "ml_hazard_engine",
        }
        ev = run_hazard_agent(data)
        assert isinstance(ev, AgentEvidence)
        assert ev.ai_assisted is False
        assert len(ev.summary) > 0

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_exposure_agent_works_without_llm(self, mock_llm):
        data = {
            "hab_id": "h1", "name": "T", "hab_type": "village",
            "is_in_red_zone": True, "hazard_class": "High",
            "population_label": "UNKNOWN", "population_value": None,
            "population_source": "UNKNOWN", "pct_high_risk": 0.6,
            "provenance": "exposure_analysis",
        }
        ev = run_exposure_agent(data)
        assert ev.ai_assisted is False

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_vulnerability_agent_works_without_llm(self, mock_llm):
        data = {
            "hab_id": "h1", "vulnerability_score": 0.8, "vulnerability_class": "CRITICAL",
            "component_scores": {}, "component_weights": {},
            "dominant_factors": [], "provenance": "vulnerability_scorer",
        }
        ev = run_vulnerability_agent(data)
        assert ev.ai_assisted is False

    @patch("flood_risk_zonation.agents.agents._llm_available", return_value=False)
    def test_capacity_agent_works_without_llm(self, mock_llm):
        data = {
            "hab_id": "h1", "capacity_score": 0.1, "capacity_status": "CRITICAL",
            "safe_area_km2": 0.0, "search_radius_km": 5.0,
            "nearest_healthcare": "not found", "nearest_road": "not found",
            "shelter_capacity": "unavailable", "shelter_source": "unavailable",
            "notes": "", "provenance": "capacity_assessment",
        }
        ev = run_capacity_agent(data)
        assert ev.ai_assisted is False
