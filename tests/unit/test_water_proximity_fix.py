"""
Tests for the water-proximity flood-risk fix (Phase 9 requirements).

Coverage:
  Water proximity:
    - Immediately adjacent land receives elevated risk (GREEN -> YELLOW/RED)
    - Strong proximity -> RED/HIGH; medium distance -> MEDIUM; far -> GREEN
    - Monotonic risk decrease as distance from water increases
    - All major water body types (ocean, river, lake, reservoir, bay) influence land
    - Far-away land remains GREEN when baseline risk is low

  Spatial continuity:
    - HIGH cell causes immediate neighbours to receive at least MEDIUM
    - Existing HIGH cells remain HIGH (never downgraded)
    - Spatial continuity is bounded/local (cells >1.5x cell_size away unaffected)
    - Water cells are not upgraded by continuity

  water_proximity_score column:
    - Always present in output
    - Non-zero for adjacent cells, zero for far cells
    - Always in [0, 100]

  Water cells excluded from land-risk scoring:
    - Water cells not boosted by proximity
    - Water cells retain risk_score = 0.0

  Relocation:
    - Candidate farther from water preferred when other factors equal
    - Missing water_proximity_score column handled gracefully

  Regression:
    - Far-inland cells unchanged without water bodies
    - Boost only raises, never lowers, risk_score
    - is_coastal_tsunami_risk column always present
    - water_proximity_score column always present
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from unittest.mock import patch

from flood_risk_zonation.config import PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.relocation.candidates import find_relocation_candidates
from flood_risk_zonation.spatial_zones.classifier import classify_spatial_zones


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(cell_size=500.0):
    return PipelineConfig(
        cell_size_meters=cell_size, use_cache=False, allow_network=False,
        low_threshold=33.0, medium_threshold=66.0,
    )


def _pipeline(cell_size=500.0):
    return FloodRiskPipeline(_config(cell_size))


def _make_grid(lons, lats, scores, risk_classes=None, csd=0.004):
    n = len(lons)
    if risk_classes is None:
        risk_classes = [
            "High" if s > 66 else ("Medium" if s > 33 else "Low")
            for s in scores
        ]
    half = csd / 2
    geoms = [box(lo - half, la - half, lo + half, la + half)
             for lo, la in zip(lons, lats)]
    return gpd.GeoDataFrame({
        "cell_id": [str(i) for i in range(n)],
        "centroid_lon": lons, "centroid_lat": lats,
        "elevation_m": np.full(n, 15.0, dtype=np.float32),
        "risk_score": np.array(scores, dtype=np.float32),
        "risk_class": risk_classes,
        "slope_deg": np.zeros(n, dtype=np.float32),
        "twi": np.zeros(n, dtype=np.float32),
        "rainfall_mean_mm": np.zeros(n, dtype=np.float32),
        "rainfall_max_24h_mm": np.zeros(n, dtype=np.float32),
        "dist_water_m": np.full(n, 5000.0, dtype=np.float32),
        "drainage_capacity": np.full(n, 0.5, dtype=np.float32),
        "population_density": np.zeros(n, dtype=np.float32),
        "aspect_deg": np.zeros(n, dtype=np.float32),
        "curvature": np.zeros(n, dtype=np.float32),
    }, geometry=geoms, crs="EPSG:4326")


def _wgdf(polys, wtypes):
    return gpd.GeoDataFrame(
        {"geometry": polys, "water_type": wtypes, "name": [""] * len(polys)},
        crs="EPSG:4326",
    )


def _boost(grid, water, cell_size=500.0):
    cfg = _config(cell_size)
    return _pipeline(cell_size)._apply_water_mask_and_proximity_boost(grid, water, cfg)


# Synthetic land polygon covering the test area
_LAND = box(77.0, 12.0, 78.0, 13.0)


# ---------------------------------------------------------------------------
# Water proximity boost — constants and gradient
# ---------------------------------------------------------------------------

class TestWaterProximityBoostConstants:
    """Verify boost_radius=5x cell_size, boost_max=100 produce intended gradient."""

    def test_adjacent_land_cell_gets_elevated_risk(self):
        """Land immediately adjacent to water must be >= MEDIUM."""
        grid = _make_grid([77.504], [12.504], [20.0])
        water = _wgdf([box(77.490, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        rc = result.iloc[0]["risk_class"]
        score = float(result.iloc[0]["risk_score"])
        assert rc in ("High", "Medium"), (
            "Cell adjacent to water must be >=MEDIUM, got " + rc
            + " score=" + str(round(score, 1))
        )
        assert score > 33.0, "Adjacent score must exceed low_threshold=33; got " + str(round(score, 2))

    def test_immediately_adjacent_land_gets_high(self):
        """Land cell essentially touching the water boundary must be HIGH."""
        grid = _make_grid([77.501], [12.504], [20.0])
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        rc = result.iloc[0]["risk_class"]
        score = float(result.iloc[0]["risk_score"])
        assert rc == "High", "Immediately adjacent land must be HIGH, got " + rc + " score=" + str(round(score, 1))

    def test_risk_score_decreases_with_distance(self):
        """Scores must be non-increasing as distance from water increases."""
        lons = [77.502, 77.506, 77.510, 77.514, 77.518]
        grid = _make_grid(lons, [12.504] * 5, [20.0] * 5)
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        s = result["risk_score"].values
        for j in range(len(s) - 1):
            assert s[j] >= s[j + 1] - 0.01, (
                "Cell " + str(j) + " (" + str(round(float(s[j]), 1)) + ")"
                + " should >= cell " + str(j + 1) + " (" + str(round(float(s[j + 1]), 1)) + ")"
            )

    def test_medium_distance_cell_at_least_medium(self):
        """Cell at ~2 cell widths (1000m) from water should exceed low_threshold."""
        grid = _make_grid([77.510], [12.504], [10.0])
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water, cell_size=500.0)
        score = float(result.iloc[0]["risk_score"])
        # At ~1000m distance, boost = 100*(1-1000/2500)=60 -> MEDIUM
        assert score > 33.0, "Cell at ~2 cell-widths must exceed low_threshold=33; got " + str(round(score, 2))

    def test_far_land_cell_remains_green(self):
        """Land >5x cell_size from water must remain at baseline LOW."""
        grid = _make_grid([77.590], [12.504], [20.0])
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water, cell_size=500.0)
        rc = result.iloc[0]["risk_class"]
        score = float(result.iloc[0]["risk_score"])
        assert rc == "Low", "Far inland cell must remain LOW; got " + rc
        assert score <= 33.0, "Far inland score must stay at baseline; got " + str(round(score, 2))


# ---------------------------------------------------------------------------
# All major water body types must influence land
# ---------------------------------------------------------------------------

class TestWaterTypes:
    """ocean, river, lake, reservoir, bay all influence surrounding land."""

    def _run(self, wtype):
        grid = _make_grid([77.502], [12.504], [15.0])
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], [wtype])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            return _boost(grid, water)

    def test_coastline_influences_land(self):
        assert self._run("coastline").iloc[0]["risk_score"] > 33.0, "Coastline must boost adjacent land"

    def test_river_influences_land(self):
        assert self._run("river").iloc[0]["risk_score"] > 15.0, "River must boost adjacent land"

    def test_lake_influences_land(self):
        assert self._run("water").iloc[0]["risk_score"] > 15.0, "Lake must boost adjacent land"

    def test_reservoir_influences_land(self):
        assert self._run("reservoir").iloc[0]["risk_score"] > 15.0, "Reservoir must boost adjacent land"

    def test_bay_influences_land(self):
        assert self._run("bay").iloc[0]["risk_score"] > 33.0, "Bay must boost adjacent land >=MEDIUM"


# ---------------------------------------------------------------------------
# Spatial continuity: HIGH cells influence their immediate neighbours
# ---------------------------------------------------------------------------

class TestSpatialContinuity:
    """HIGH cells propagate a bounded MEDIUM influence to immediate neighbours."""

    def test_high_cell_neighbour_gets_medium(self):
        """A LOW cell 1 grid-cell away from a HIGH cell must become MEDIUM."""
        lons = [77.502, 77.506, 77.510]
        grid = _make_grid(lons, [12.504] * 3, [80.0, 10.0, 10.0], ["High", "Low", "Low"])
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty, _config())
        rc1 = result.iloc[1]["risk_class"]
        assert rc1 in ("High", "Medium"), "Neighbour of HIGH must be >=MEDIUM; got " + rc1

    def test_existing_high_remains_high(self):
        """HIGH cells must never be downgraded by spatial continuity."""
        lons = [77.502, 77.506]
        grid = _make_grid(lons, [12.504] * 2, [80.0, 75.0], ["High", "High"])
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty, _config())
        assert result.iloc[0]["risk_class"] == "High", "HIGH cell 0 must remain HIGH"
        assert result.iloc[1]["risk_class"] == "High", "HIGH cell 1 must remain HIGH"

    def test_spatial_continuity_bounded_not_global(self):
        """Cells farther than 1.5x cell_size from HIGH are not boosted."""
        lons = [77.502, 77.506, 77.510, 77.514, 77.518]
        grid = _make_grid(lons, [12.504] * 5, [80.0, 10.0, 10.0, 10.0, 10.0],
                          ["High", "Low", "Low", "Low", "Low"])
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty, _config())
        # cell 4 is ~2000m from HIGH cell 0; radius=750m -> no continuity boost
        rc4 = result.iloc[4]["risk_class"]
        assert rc4 == "Low", "Cell far from HIGH must remain LOW; got " + rc4

    def test_water_cells_not_upgraded_by_continuity(self):
        """Water cells must not be upgraded to MEDIUM by spatial continuity."""
        lons = [77.502, 77.506]
        grid = _make_grid(lons, [12.504] * 2, [80.0, 0.0], ["High", "Water"])
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty, _config())
        assert result.iloc[1]["risk_class"] == "Water", "Water cell must not be upgraded by continuity"


# ---------------------------------------------------------------------------
# water_proximity_score column
# ---------------------------------------------------------------------------

class TestWaterProximityScoreColumn:
    """water_proximity_score must be present, non-negative, and informative."""

    def test_column_always_present(self):
        grid = _make_grid([77.55], [12.50], [20.0])
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, empty)
        assert "water_proximity_score" in result.columns

    def test_adjacent_cell_nonzero(self):
        grid = _make_grid([77.502], [12.504], [20.0])
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        assert result.iloc[0]["water_proximity_score"] > 0.0, "Adjacent cell must have non-zero proximity score"

    def test_far_cell_zero(self):
        grid = _make_grid([77.590], [12.504], [20.0])
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water, cell_size=500.0)
        assert result.iloc[0]["water_proximity_score"] == 0.0, "Far cell must have proximity score = 0"

    def test_scores_in_valid_range(self):
        lons = [77.502, 77.510, 77.550, 77.590]
        grid = _make_grid(lons, [12.504] * 4, [20.0] * 4)
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        s = result["water_proximity_score"].values
        assert (s >= 0.0).all() and (s <= 100.0).all(), "Proximity scores must be in [0, 100]"


# ---------------------------------------------------------------------------
# Water cells excluded from land-risk scoring
# ---------------------------------------------------------------------------

class TestWaterCellsExcluded:
    """Water cells must not be upgraded by proximity boost or spatial continuity."""

    def test_water_cell_not_boosted(self):
        grid = _make_grid([77.502], [12.504], [20.0], ["Water"])
        water = _wgdf([box(77.480, 12.50, 77.510, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        assert result.iloc[0]["risk_class"] == "Water", "Water cell must remain Water"

    def test_water_cell_score_zero(self):
        grid = _make_grid([77.502], [12.504], [0.0], ["Water"])
        water = _wgdf([box(77.480, 12.50, 77.510, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        assert result.iloc[0]["risk_score"] == 0.0, "Water cell risk_score must remain 0.0"


# ---------------------------------------------------------------------------
# Relocation: water-close candidates penalised
# ---------------------------------------------------------------------------

class TestRelocationWaterPenalty:
    """Relocation candidates close to water must score lower than farther ones."""

    def _zoned(self, rcs, wps, nc=3):
        n = len(rcs)
        rows = []
        score_map = {"High": 80.0, "Medium": 50.0, "Low": 20.0, "Water": 0.0}
        for i, (rc, wp) in enumerate(zip(rcs, wps)):
            r, c = i // nc, i % nc
            la, lo = 12.84 + r * 0.008, 77.55 + c * 0.008
            rows.append({
                "cell_id": "c" + str(i), "risk_class": rc,
                "risk_score": score_map.get(rc, 20.0),
                "centroid_lat": la, "centroid_lon": lo,
                "population_density": 0.0,
                "water_proximity_score": wp,
                "geometry": box(lo - 0.004, la - 0.004, lo + 0.004, la + 0.004),
            })
        return classify_spatial_zones(gpd.GeoDataFrame(rows, crs="EPSG:4326"))

    def test_water_proximity_penalises_candidate_score(self):
        """Candidate with high water_proximity_score scores lower than identical one with low proximity."""
        rc = ["Low"] * 4
        g_close = self._zoned(rc, [80.0] * 4, nc=2)
        g_far = self._zoned(rc, [0.0] * 4, nc=2)
        kw = dict(hab_lat=12.844, hab_lon=77.558, hab_id="h1", hab_name="T",
                  search_radius_km=20.0, max_candidates=5)
        c_close = find_relocation_candidates(zoned_grid=g_close, **kw)
        c_far = find_relocation_candidates(zoned_grid=g_far, **kw)
        if c_close and c_far:
            assert c_far[0].candidate_score >= c_close[0].candidate_score, (
                "Far-from-water candidate (" + str(round(c_far[0].candidate_score, 3)) + ")"
                + " must score >= water-close ("
                + str(round(c_close[0].candidate_score, 3)) + ")"
            )

    def test_candidates_sorted_descending(self):
        rc = ["Low"] * 9
        wps = [80.0] * 3 + [0.0] * 6
        grid = self._zoned(rc, wps)
        cands = find_relocation_candidates(
            hab_lat=12.844, hab_lon=77.558, hab_id="h1", hab_name="T",
            zoned_grid=grid, search_radius_km=20.0, max_candidates=10,
        )
        scores = [c.candidate_score for c in cands]
        assert scores == sorted(scores, reverse=True), "Candidates must be sorted by score descending"

    def test_missing_column_graceful(self):
        """If water_proximity_score column is absent, relocation still works."""
        rows = []
        for i in range(9):
            r, c = i // 3, i % 3
            la, lo = 12.84 + r * 0.008, 77.55 + c * 0.008
            rows.append({
                "cell_id": "c" + str(i), "risk_class": "Low", "risk_score": 20.0,
                "centroid_lat": la, "centroid_lon": lo, "population_density": 0.0,
                "geometry": box(lo - 0.004, la - 0.004, lo + 0.004, la + 0.004),
            })
        gdf = classify_spatial_zones(gpd.GeoDataFrame(rows, crs="EPSG:4326"))
        cands = find_relocation_candidates(
            hab_lat=12.844, hab_lon=77.558, hab_id="h1", hab_name="T",
            zoned_grid=gdf, search_radius_km=30.0, max_candidates=5,
        )
        assert isinstance(cands, list)


# ---------------------------------------------------------------------------
# Regression: unrelated calculations unchanged
# ---------------------------------------------------------------------------

class TestRegression:
    """Existing risk calculations must be unaffected."""

    def test_no_change_without_water_bodies(self):
        """Far-inland cells with no water bodies keep their baseline scores."""
        full = box(77.0, 12.0, 78.0, 13.0)
        grid = _make_grid(
            [77.55, 77.56, 77.57], [12.84] * 3,
            [80.0, 50.0, 20.0], ["High", "Medium", "Low"],
        )
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=full):
            result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty, _config())
        assert float(result.iloc[0]["risk_score"]) == 80.0
        assert float(result.iloc[1]["risk_score"]) == 50.0
        assert float(result.iloc[2]["risk_score"]) == 20.0
        assert result.iloc[0]["risk_class"] == "High"
        assert result.iloc[1]["risk_class"] == "Medium"
        assert result.iloc[2]["risk_class"] == "Low"

    def test_boost_only_raises_never_lowers(self):
        """A pre-existing HIGH cell next to water must not be lowered."""
        grid = _make_grid([77.502], [12.504], [90.0], ["High"])
        water = _wgdf([box(77.480, 12.50, 77.500, 12.510)], ["water"])
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, water)
        assert float(result.iloc[0]["risk_score"]) >= 90.0, "Boost must not lower existing score"
        assert result.iloc[0]["risk_class"] == "High", "Existing HIGH must remain HIGH"

    def test_coastal_tsunami_column_present(self):
        grid = _make_grid([77.55], [12.84], [20.0])
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, empty)
        assert "is_coastal_tsunami_risk" in result.columns

    def test_water_proximity_column_always_present(self):
        grid = _make_grid([77.55], [12.84], [20.0])
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=_LAND):
            result = _boost(grid, empty)
        assert "water_proximity_score" in result.columns
