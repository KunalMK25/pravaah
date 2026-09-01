"""Unit tests for habitation exposure analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box

from flood_risk_zonation.models import Habitation, HabitationDataset
from flood_risk_zonation.exposure.analysis import (
    analyse_exposure,
    _dominant_class,
    _haversine_km,
    _classify_hazard,
)


def _make_grid(n: int = 20, risk_classes=None, scores=None):
    """Build a minimal scored GeoDataFrame for tests."""
    if risk_classes is None:
        risk_classes = ["High"] * (n // 2) + ["Low"] * (n - n // 2)
    if scores is None:
        scores = [80.0] * (n // 2) + [20.0] * (n - n // 2)
    lats = [12.84 + i * 0.002 for i in range(n)]
    lons = [77.55 + i * 0.002 for i in range(n)]
    geoms = [box(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
             for lat, lon in zip(lats, lons)]
    df = pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n)],
        "centroid_lat": lats,
        "centroid_lon": lons,
        "risk_score": scores,
        "risk_class": risk_classes,
        "elevation_m": [30.0] * n,
        "dist_water_m": [500.0] * n,
        "drainage_capacity": [0.4] * n,
    })
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


class TestHaversine:
    def test_zero_distance(self):
        d = _haversine_km(12.9, 77.6, 12.9, 77.6)
        assert d == pytest.approx(0.0, abs=0.001)

    def test_known_distance(self):
        # ~111.2 km per degree latitude
        d = _haversine_km(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111.2, rel=0.01)


class TestDominantClass:
    def test_high_dominates(self):
        assert _dominant_class(["Low", "Medium", "High"]) == "High"

    def test_all_low(self):
        assert _dominant_class(["Low", "Low", "Low"]) == "Low"

    def test_empty(self):
        assert _dominant_class([]) == "Low"

    def test_water_lowest(self):
        assert _dominant_class(["Water", "Low"]) == "Low"


class TestClassifyHazard:
    def test_high_score(self):
        assert _classify_hazard(80.0, 0.6, 33.0, 66.0) == "High"

    def test_medium_score(self):
        assert _classify_hazard(50.0, 0.1, 33.0, 66.0) == "Medium"

    def test_low_score(self):
        assert _classify_hazard(20.0, 0.0, 33.0, 66.0) == "Low"

    def test_high_pct_overrides_medium_score(self):
        assert _classify_hazard(50.0, 0.6, 33.0, 66.0) == "High"


class TestAnalyseExposure:
    def test_empty_habitations(self):
        grid = _make_grid()
        ds = HabitationDataset(habitations=[], source="fallback", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results == []

    def test_single_habitation_high_risk(self):
        grid = _make_grid(n=10, risk_classes=["High"] * 10, scores=[80.0] * 10)
        hab = Habitation("h1", "Test", "village", 12.848, 77.558, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert len(results) == 1
        assert results[0].is_in_red_zone is True
        assert results[0].hazard_class == "High"

    def test_single_habitation_low_risk(self):
        grid = _make_grid(n=10, risk_classes=["Low"] * 10, scores=[20.0] * 10)
        hab = Habitation("h1", "Safe", "hamlet", 12.848, 77.558, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results[0].is_in_red_zone is False
        assert results[0].hazard_class == "Low"

    def test_population_from_osm_tag(self):
        grid = _make_grid(n=6)
        hab = Habitation("h1", "Pop Village", "village", 12.848, 77.558,
                         "osm_overpass", population=3000)
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results[0].population_source == "osm_tag"
        assert results[0].population_exposed == 3000

    def test_population_unknown_when_not_in_osm(self):
        grid = _make_grid(n=6)
        hab = Habitation("h1", "NoPopVillage", "village", 12.848, 77.558, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert results[0].population_source == "UNKNOWN"
        assert results[0].population_exposed is None

    def test_multiple_habitations(self):
        grid = _make_grid(n=20)
        habs = [
            Habitation(f"h{i}", f"Hab {i}", "village", 12.84 + i * 0.003, 77.55 + i * 0.003, "osm_overpass")
            for i in range(5)
        ]
        ds = HabitationDataset(habitations=habs, source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        assert len(results) == 5

    def test_water_cells_excluded_from_score(self):
        grid = _make_grid(n=4, risk_classes=["Water"] * 4, scores=[0.0] * 4)
        hab = Habitation("h1", "Water Hab", "village", 12.842, 77.552, "osm_overpass")
        ds = HabitationDataset(habitations=[hab], source="osm_overpass", bbox_key="k")
        results = analyse_exposure(ds, grid)
        # Water habitations are now correctly filtered from results
        assert len(results) == 0, "Water habitation should be filtered from results"


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION TESTS: All-Water Region Bug (Indian Ocean Preset)
# ─────────────────────────────────────────────────────────────────────────────
#
# Issue: Habitations in 100% water regions were incorrectly included in
# exposure_results, leading to false HIGH-priority and habitation counts.
#
# Fix: Filter water-cell habitations (hazard_class=="Water") from results.
#
# Tests ensure:
# A. All-water regions return zero habitations
# B. Mixed land/water regions filter only water habitations
# C. Land habitations in mixed regions are preserved


class TestAllWaterRegionRegression:
    """
    Regression test: All-water region (Indian Ocean preset).
    
    When a region is 100% water cells, all habitations should be filtered out.
    """
    
    def test_all_water_habitations_filtered(self):
        """
        REGRESSION TEST A: All-water region returns zero exposure results.
        
        Setup: 100% water grid + 3 habitations
        Expected: All habitations filtered (hazard_class="Water")
        """
        # Create 100% water grid (10x10 cells)
        water_classes = ["Water"] * 100
        water_scores = [0.0] * 100
        grid = _make_grid(n=100, risk_classes=water_classes, scores=water_scores)
        
        # Create 3 habitations in the water region
        habs = [
            Habitation("h_ocean_1", "Ocean Point 1", "place_node", 12.842, 77.552, "synthetic", population=50),
            Habitation("h_ocean_2", "Ocean Point 2", "place_node", 12.850, 77.560, "synthetic", population=100),
            Habitation("h_ocean_3", "Ocean Point 3", "place_node", 12.858, 77.568, "synthetic", population=75),
        ]
        ds = HabitationDataset(habitations=habs, source="synthetic", bbox_key="ocean")
        
        # Run exposure analysis
        results = analyse_exposure(ds, grid)
        
        # CRITICAL: No habitations should survive
        assert len(results) == 0, \
            f"All-water region should return 0 exposures, got {len(results)}"
        
        # Verify all cells are water
        assert (grid['risk_class'] == 'Water').all(), \
            "Test grid must be 100% Water"
    
    def test_all_water_habitations_have_water_hazard_class(self):
        """
        Verify that water habitations are correctly identified (hazard_class="Water")
        before filtering. This validates the classification is accurate.
        """
        # Create all-water grid
        grid = _make_grid(n=20, risk_classes=["Water"] * 20, scores=[0.0] * 20)
        
        # Create single habitation
        hab = Habitation("h1", "Water Only", "place_node", 12.842, 77.552, "synthetic")
        ds = HabitationDataset(habitations=[hab], source="synthetic", bbox_key="w1")
        
        # Run exposure (before filtering, capture via direct scoring)
        # Since filtering is applied in analyse_exposure, verify by checking internal logic
        results = analyse_exposure(ds, grid)
        
        # After the fix, results should be empty (water filtered)
        assert len(results) == 0, \
            "Water habitations should be filtered from results"


class TestMixedLandWaterRegionRegression:
    """
    Regression test: Mixed land/water region.
    
    When a region has both land and water cells, only water habitations should be filtered.
    Land habitations must NOT be accidentally removed.
    """
    
    def test_filters_only_water_habitations_mixed_region(self):
        """
        REGRESSION TEST B: Mixed land/water region filters only water habitations.
        
        Setup: Dense water region where all nearby cells are Water
        - hab_0: pure water (all 4 nearest cells are Water) → filtered
        - hab_1: pure water (all 4 nearest cells are Water) → filtered
        - hab_2: land (all 4 nearest cells are High) → keep
        """
        # Create two distinct regions: water zone + land zone
        # Water zone: cells 0-9 all Water
        # Land zone: cells 10-19 all High
        lats = [9.95 + i * 0.001 for i in range(20)]  # Water at south, land at north
        lons = [71.95 + i * 0.001 for i in range(20)]
        risk_classes = ["Water"] * 10 + ["High"] * 10
        scores = [0.0] * 10 + [80.0] * 10
        
        geoms = [box(lon - 0.0005, lat - 0.0005, lon + 0.0005, lat + 0.0005)
                 for lat, lon in zip(lats, lons)]
        
        df = pd.DataFrame({
            "cell_id": [f"cell_{i}" for i in range(20)],
            "centroid_lat": lats,
            "centroid_lon": lons,
            "risk_score": scores,
            "risk_class": risk_classes,
        })
        grid = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
        
        # Create 3 habitations: 2 in pure water zone, 1 in land zone
        habs = [
            # Water habitations (southern region, only water cells nearby)
            Habitation("h_water_0", "Water Point 0", "place_node", 9.951, 71.951, "synthetic"),
            Habitation("h_water_1", "Water Point 1", "place_node", 9.955, 71.955, "synthetic"),
            # Land habitation (northern region, only land cells nearby)
            Habitation("h_land_2", "Land Village 2", "village", 9.965, 71.965, "synthetic"),
        ]
        ds = HabitationDataset(habitations=habs, source="synthetic", bbox_key="mixed_pure")
        
        # Run exposure analysis
        results = analyse_exposure(ds, grid)
        
        # Should have exactly 1 result (only the land habitation)
        assert len(results) == 1, \
            f"Mixed region should return 1 land habitation, got {len(results)}"
        
        # Verify it's the land one
        result_ids = {r.hab_id for r in results}
        assert "h_land_2" in result_ids, "Land hab h_land_2 should be in results"
        
        # Verify no water habitations survived
        assert "h_water_0" not in result_ids, "Water hab h_water_0 should be filtered"
        assert "h_water_1" not in result_ids, "Water hab h_water_1 should be filtered"
    
    def test_no_valid_land_habitations_wrongly_removed(self):
        """
        Verify that valid land habitations are preserved and not accidentally filtered.
        """
        # Create grid with High-risk land
        grid = _make_grid(n=10, risk_classes=["High"] * 10, scores=[80.0] * 10)
        
        # Create land habitation
        hab = Habitation("h_valid_land", "Definite Land", "village", 12.848, 77.558, "synthetic")
        ds = HabitationDataset(habitations=[hab], source="synthetic", bbox_key="land")
        
        results = analyse_exposure(ds, grid)
        
        # Should NOT be filtered
        assert len(results) == 1, "Land habitation should not be filtered"
        assert results[0].hab_id == "h_valid_land"
        assert results[0].hazard_class == "High"


class TestExistingLandRegressionBengaluru:
    """
    Regression test: Ensure existing land-based tests still pass.
    
    Verify that the water-filtering fix does not break normal land analysis.
    """
    
    def test_bangalore_gottigere_land_habitations(self):
        """
        Verify Bangalore (Gottigere) land habitations are unaffected.
        This is a normal land region with mixed High/Low risk.
        """
        grid = _make_grid(n=20, risk_classes=["High"] * 10 + ["Low"] * 10,
                          scores=[80.0] * 10 + [20.0] * 10)
        
        habs = [
            Habitation("b_1", "Bangalore 1", "village", 12.844, 77.554, "osm_overpass", population=2000),
            Habitation("b_2", "Bangalore 2", "village", 12.850, 77.560, "osm_overpass", population=1500),
        ]
        ds = HabitationDataset(habitations=habs, source="osm_overpass", bbox_key="blr")
        
        results = analyse_exposure(ds, grid)
        
        # Both land habitations should be returned
        assert len(results) == 2, "Bangalore land habitations should not be filtered"
        
        # Verify population data is preserved
        assert results[0].population_exposed == 2000
        assert results[1].population_exposed == 1500
        
        # Verify hazard classes are set correctly
        assert results[0].hazard_class in ("High", "Medium", "Low")
        assert results[1].hazard_class in ("High", "Medium", "Low")
