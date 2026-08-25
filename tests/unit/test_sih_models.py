"""Unit tests for SIH26191 data model dataclasses."""
import pytest
from flood_risk_zonation.models import (
    Habitation,
    HabitationDataset,
    ExposureResult,
    VulnerabilityResult,
    CarryingCapacityResult,
    RelocationPriorityResult,
    SIHAnalysisResult,
)


class TestHabitation:
    def test_basic_construction(self):
        h = Habitation(
            hab_id="osm_123",
            name="Test Village",
            hab_type="village",
            lat=12.90,
            lon=77.60,
            source="osm_overpass",
        )
        assert h.hab_id == "osm_123"
        assert h.name == "Test Village"
        assert h.population is None

    def test_with_population(self):
        h = Habitation(
            hab_id="osm_456",
            name="Town A",
            hab_type="town",
            lat=12.85,
            lon=77.58,
            source="osm_cache",
            population=5000,
        )
        assert h.population == 5000

    def test_fallback_source(self):
        h = Habitation(
            hab_id="fallback_001",
            name="Settlement (fallback)",
            hab_type="village",
            lat=12.87,
            lon=77.57,
            source="fallback",
        )
        assert h.source == "fallback"


class TestHabitationDataset:
    def test_empty_dataset(self):
        ds = HabitationDataset(habitations=[], source="fallback", bbox_key="test_key")
        assert len(ds.habitations) == 0
        assert ds.source == "fallback"

    def test_dataset_with_habitations(self):
        habs = [
            Habitation("osm_1", "A", "village", 12.9, 77.6, "osm_overpass"),
            Habitation("osm_2", "B", "hamlet", 12.88, 77.58, "osm_overpass"),
        ]
        ds = HabitationDataset(habitations=habs, source="osm_overpass", bbox_key="k1")
        assert len(ds.habitations) == 2


class TestExposureResult:
    def test_red_zone_flag(self):
        exp = ExposureResult(
            hab_id="osm_1",
            name="High Risk Village",
            hab_type="village",
            lat=12.9, lon=77.6,
            hazard_score=80.0,
            hazard_class="High",
            pct_high_risk=0.75,
            population_source="osm_tag",
            population_exposed=1200,
            is_in_red_zone=True,
        )
        assert exp.is_in_red_zone is True
        assert exp.population_source == "osm_tag"

    def test_unknown_population(self):
        exp = ExposureResult(
            hab_id="osm_2",
            name="Unknown Pop Village",
            hab_type="hamlet",
            lat=12.87, lon=77.57,
            hazard_score=45.0,
            hazard_class="Medium",
            pct_high_risk=0.2,
            population_source="UNKNOWN",
            population_exposed=None,
            is_in_red_zone=False,
        )
        assert exp.population_exposed is None
        assert exp.population_source == "UNKNOWN"


class TestVulnerabilityResult:
    def test_classes(self):
        for score, expected_class in [
            (0.1, "LOW"), (0.3, "MEDIUM"), (0.6, "HIGH"), (0.8, "CRITICAL")
        ]:
            v = VulnerabilityResult(
                hab_id="test",
                vulnerability_score=score,
                vulnerability_class=expected_class,
            )
            assert v.vulnerability_class == expected_class
            assert 0.0 <= v.vulnerability_score <= 1.0


class TestCarryingCapacityResult:
    def test_statuses(self):
        for score, status in [(0.8, "ADEQUATE"), (0.45, "STRESSED"), (0.2, "CRITICAL")]:
            c = CarryingCapacityResult(
                hab_id="test",
                capacity_score=score,
                capacity_status=status,
                safe_area_km2=2.0,
                search_radius_km=5.0,
                nearest_healthcare_km=3.0,
                nearest_road_km=0.5,
            )
            assert c.capacity_status == status

    def test_unknown_distances(self):
        c = CarryingCapacityResult(
            hab_id="test",
            capacity_score=0.3,
            capacity_status="CRITICAL",
            safe_area_km2=0.0,
            search_radius_km=5.0,
            nearest_healthcare_km=-1.0,
            nearest_road_km=-1.0,
        )
        assert c.nearest_healthcare_km == -1.0
        assert c.nearest_road_km == -1.0


class TestRelocationPriorityResult:
    def test_action_classes(self):
        for score, expected in [
            (0.1, "LOW"), (0.35, "MEDIUM"), (0.6, "HIGH"), (0.85, "CRITICAL")
        ]:
            r = RelocationPriorityResult(
                hab_id="test",
                name="Test Hab",
                relocation_score=score,
                priority_class=expected,
                recommended_action="Action",
            )
            assert r.priority_class == expected

    def test_population_defaults(self):
        r = RelocationPriorityResult(
            hab_id="test",
            name="Test",
            relocation_score=0.5,
            priority_class="HIGH",
            recommended_action="Action",
        )
        assert r.population_exposed is None
        assert r.population_source == "UNKNOWN"


class TestSIHAnalysisResult:
    def _make_result(self):
        from unittest.mock import MagicMock
        mock_hazard = MagicMock()
        mock_dataset = HabitationDataset(
            habitations=[
                Habitation("h1", "A", "village", 12.9, 77.6, "osm_overpass"),
                Habitation("h2", "B", "hamlet", 12.87, 77.57, "osm_overpass"),
            ],
            source="osm_overpass",
            bbox_key="test",
        )
        exp = [
            ExposureResult("h1", "A", "village", 12.9, 77.6, 80.0, "High", 0.8, "UNKNOWN", None, True),
            ExposureResult("h2", "B", "hamlet", 12.87, 77.57, 30.0, "Low", 0.1, "osm_tag", 500, False),
        ]
        rel = [
            RelocationPriorityResult("h1", "A", 0.85, "CRITICAL", "Immediate relocation"),
            RelocationPriorityResult("h2", "B", 0.2, "LOW", "Routine monitoring"),
        ]
        return SIHAnalysisResult(
            flood_risk_result=mock_hazard,
            habitation_dataset=mock_dataset,
            exposure_results=exp,
            relocation_results=rel,
        )

    def test_critical_habitations(self):
        sih = self._make_result()
        assert len(sih.critical_habitations) == 1
        assert sih.critical_habitations[0].hab_id == "h1"

    def test_red_zone_habitations(self):
        sih = self._make_result()
        assert len(sih.red_zone_habitations) == 1

    def test_get_relocation_by_id(self):
        sih = self._make_result()
        r = sih.get_relocation_by_id("h1")
        assert r is not None
        assert r.priority_class == "CRITICAL"

    def test_get_relocation_missing(self):
        sih = self._make_result()
        assert sih.get_relocation_by_id("nonexistent") is None
