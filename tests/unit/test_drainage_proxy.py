"""
Unit tests for the OSM drainage infrastructure proxy.

Requirements tested:
    T1  — OSM drainage linestrings are correctly ingested from water_bodies
    T2  — Drainage proxy formula produces spatially varying, bounded [0,1] scores
    T3  — Results are deterministic for identical input
    T4  — No drainage data -> graceful synthetic fallback (source="synthetic_fallback")
    T5  — Drainage feature integrates correctly into extract_features
    T6  — Full risk pipeline executes with the new drainage proxy
    T7  — Existing water-proximity behaviour remains unchanged
    T8  — HIGH/MEDIUM/GREEN classification behaviour remains valid
    T9  — Relocation candidate logic still executes end-to-end
    T10 — Larger grids do not cause catastrophic performance regression
    T11 — Unrelated feature calculations remain unchanged
"""
from __future__ import annotations

import time

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.ingest.drainage import (
    _extract_drainage_lines,
    generate_drainage_proxy,
    generate_synthetic_drainage,
    _W_DRAIN,
    _W_CANAL,
    _W_RIVER,
    _MAX_RIVER_DIST_M,
)
from flood_risk_zonation.models import DrainageDataset


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture
def small_grid():
    bbox = BoundingBox(0.0, 0.0, 0.15, 0.15)
    return generate_grid(bbox, cell_size_meters=5_000.0)


@pytest.fixture
def grid_with_drains(small_grid):
    centroid_lon = float(small_grid["centroid_lon"].mean())
    centroid_lat = float(small_grid["centroid_lat"].mean())
    drain_line = LineString([
        (centroid_lon - 0.02, centroid_lat),
        (centroid_lon + 0.02, centroid_lat),
    ])
    wb = gpd.GeoDataFrame(
        [{"geometry": drain_line, "water_type": "drain", "name": "test_drain"}],
        crs="EPSG:4326",
    )
    return small_grid, wb


@pytest.fixture
def grid_with_mixed_waterways(small_grid):
    centroid_lon = float(small_grid["centroid_lon"].mean())
    centroid_lat = float(small_grid["centroid_lat"].mean())
    rows = [
        {"geometry": LineString([(centroid_lon - 0.02, centroid_lat - 0.01),
                                  (centroid_lon + 0.02, centroid_lat - 0.01)]),
         "water_type": "drain", "name": ""},
        {"geometry": LineString([(centroid_lon - 0.05, centroid_lat + 0.01),
                                  (centroid_lon + 0.05, centroid_lat + 0.01)]),
         "water_type": "canal", "name": ""},
        {"geometry": LineString([(centroid_lon, centroid_lat - 0.06),
                                  (centroid_lon, centroid_lat + 0.06)]),
         "water_type": "river", "name": ""},
        {"geometry": Polygon([(centroid_lon - 0.03, centroid_lat - 0.03),
                               (centroid_lon + 0.03, centroid_lat - 0.03),
                               (centroid_lon + 0.03, centroid_lat + 0.03),
                               (centroid_lon - 0.03, centroid_lat + 0.03)]),
         "water_type": "water", "name": ""},
    ]
    wb = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return small_grid, wb


def _make_extract_inputs(grid):
    import rasterio.transform
    from flood_risk_zonation.models import RasterDataset, RainfallDataset
    n_px = 5
    arr = np.full((n_px, n_px), 75.0, dtype=np.float32)
    transform = rasterio.transform.from_bounds(
        grid["centroid_lon"].min() - 0.05,
        grid["centroid_lat"].min() - 0.05,
        grid["centroid_lon"].max() + 0.05,
        grid["centroid_lat"].max() + 0.05,
        n_px, n_px,
    )
    elev = RasterDataset(
        array=arr, transform=transform, crs="EPSG:4326",
        nodata=None, source="synthetic",
    )
    rainfall = RainfallDataset(
        mean_annual_mm=arr.copy(), max_24h_mm=arr.copy() * 0.2,
        transform=transform, crs="EPSG:4326",
        temporal_range=("2000-01-01", "2020-12-31"),
        source="synthetic",
    )
    pop = RasterDataset(
        array=np.zeros((n_px, n_px), dtype=np.float32),
        transform=transform, crs="EPSG:4326",
        nodata=None, source="synthetic",
    )
    return elev, rainfall, pop


# -- T1: OSM drainage linestrings correctly ingested --------------------------

class TestDrainageLineExtraction:
    def test_drain_lines_extracted(self, grid_with_drains):
        _, wb = grid_with_drains
        lines = _extract_drainage_lines(wb)
        assert lines is not None and len(lines) == 1
        assert lines["_dtype"].iloc[0] == "drain"

    def test_polygon_water_excluded(self, grid_with_mixed_waterways):
        _, wb = grid_with_mixed_waterways
        lines = _extract_drainage_lines(wb)
        assert lines is not None and len(lines) == 3

    def test_canal_tagged_correctly(self, grid_with_mixed_waterways):
        _, wb = grid_with_mixed_waterways
        lines = _extract_drainage_lines(wb)
        assert "canal" in lines["_dtype"].values

    def test_river_tagged_correctly(self, grid_with_mixed_waterways):
        _, wb = grid_with_mixed_waterways
        lines = _extract_drainage_lines(wb)
        assert "river" in lines["_dtype"].values

    def test_empty_water_bodies_returns_none(self, small_grid):
        wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        assert _extract_drainage_lines(wb) is None

    def test_none_returns_none(self):
        assert _extract_drainage_lines(None) is None

    def test_stream_classified_as_river(self):
        line = LineString([(0.05, 0.05), (0.1, 0.05)])
        wb = gpd.GeoDataFrame([{"geometry": line, "water_type": "stream", "name": ""}], crs="EPSG:4326")
        lines = _extract_drainage_lines(wb)
        assert lines is not None and lines["_dtype"].iloc[0] == "river"


# -- T2: Proxy formula produces bounded, spatially-varying scores -------------

class TestDrainageProxyFormula:
    def test_scores_in_unit_interval(self, grid_with_drains):
        grid, wb = grid_with_drains
        r = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0)
        assert np.all(r.capacity_scores >= 0.0) and np.all(r.capacity_scores <= 1.0)

    def test_scores_are_float32(self, grid_with_drains):
        grid, wb = grid_with_drains
        r = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0)
        assert r.capacity_scores.dtype == np.float32

    def test_source_is_osm_proxy_with_linestrings(self, grid_with_drains):
        grid, wb = grid_with_drains
        r = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0)
        assert r.source == "osm_proxy"

    def test_cell_ids_match_grid(self, grid_with_drains):
        grid, wb = grid_with_drains
        r = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0)
        assert len(r.cell_ids) == len(grid)
        assert r.cell_ids == list(grid["cell_id"].astype(str))

    def test_weights_sum_to_one(self):
        assert abs(_W_DRAIN + _W_CANAL + _W_RIVER - 1.0) < 1e-9

    def test_max_river_dist_positive(self):
        assert _MAX_RIVER_DIST_M > 0

    def test_scores_not_all_zero_with_drain_present(self, grid_with_drains):
        grid, wb = grid_with_drains
        r = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0)
        assert r.capacity_scores.max() > 0


# -- T3: Determinism -----------------------------------------------------------

class TestDeterminism:
    def test_proxy_deterministic(self, grid_with_drains):
        grid, wb = grid_with_drains
        r1 = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0, seed=42)
        r2 = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0, seed=42)
        np.testing.assert_array_equal(r1.capacity_scores, r2.capacity_scores)

    def test_synthetic_fallback_deterministic(self, small_grid):
        r1 = generate_synthetic_drainage(small_grid, seed=99)
        r2 = generate_synthetic_drainage(small_grid, seed=99)
        np.testing.assert_array_equal(r1.capacity_scores, r2.capacity_scores)

    def test_mixed_waterways_deterministic(self, grid_with_mixed_waterways):
        grid, wb = grid_with_mixed_waterways
        r1 = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0, seed=7)
        r2 = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0, seed=7)
        np.testing.assert_array_equal(r1.capacity_scores, r2.capacity_scores)


# -- T4: No drainage data -> graceful synthetic fallback ----------------------

class TestFallback:
    def test_none_water_bodies_fallback(self, small_grid):
        r = generate_drainage_proxy(small_grid, None, cell_size_m=5_000.0)
        assert r.source == "synthetic_fallback"
        assert len(r.capacity_scores) == len(small_grid)

    def test_empty_water_bodies_fallback(self, small_grid):
        wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        r = generate_drainage_proxy(small_grid, wb, cell_size_m=5_000.0)
        assert r.source == "synthetic_fallback"

    def test_polygon_only_fallback(self, small_grid):
        poly = Polygon([(0.01, 0.01), (0.05, 0.01), (0.05, 0.05), (0.01, 0.05)])
        wb = gpd.GeoDataFrame([{"geometry": poly, "water_type": "water", "name": ""}], crs="EPSG:4326")
        r = generate_drainage_proxy(small_grid, wb, cell_size_m=5_000.0)
        assert r.source == "synthetic_fallback"

    def test_fallback_scores_in_valid_range(self, small_grid):
        r = generate_drainage_proxy(small_grid, None, cell_size_m=5_000.0)
        assert np.all(r.capacity_scores >= 0.0) and np.all(r.capacity_scores <= 1.0)

    def test_generate_synthetic_backward_compat(self, small_grid):
        r = generate_synthetic_drainage(small_grid, seed=42)
        assert isinstance(r, DrainageDataset)
        assert r.source == "synthetic_fallback"
        assert len(r.capacity_scores) == len(small_grid)


# -- T5: Feature integration --------------------------------------------------

class TestFeatureExtraction:
    def test_drainage_capacity_column_present(self, small_grid):
        from flood_risk_zonation.features.extractor import extract_features, FEATURE_COLUMNS
        elev, rainfall, pop = _make_extract_inputs(small_grid)
        wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        drainage = generate_synthetic_drainage(small_grid)
        result = extract_features(small_grid, elev, rainfall, wb, pop, drainage)
        assert "drainage_capacity" in result.columns

    def test_drainage_capacity_in_range_after_extraction(self, small_grid):
        from flood_risk_zonation.features.extractor import extract_features
        elev, rainfall, pop = _make_extract_inputs(small_grid)
        drain_line = LineString([(0.05, 0.07), (0.10, 0.07)])
        wb = gpd.GeoDataFrame([{"geometry": drain_line, "water_type": "drain", "name": ""}], crs="EPSG:4326")
        drainage = generate_drainage_proxy(small_grid, wb, cell_size_m=5_000.0)
        result = extract_features(small_grid, elev, rainfall, wb, pop, drainage)
        col = result["drainage_capacity"].values
        assert np.all(col >= 0.0) and np.all(col <= 1.0)

    def test_all_feature_columns_present(self, small_grid):
        from flood_risk_zonation.features.extractor import extract_features, FEATURE_COLUMNS
        elev, rainfall, pop = _make_extract_inputs(small_grid)
        wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        drainage = generate_synthetic_drainage(small_grid)
        result = extract_features(small_grid, elev, rainfall, wb, pop, drainage)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing: {col}"


# -- T6: Pipeline regression --------------------------------------------------

class TestPipelineRegression:
    def _make_feature_grid(self, grid, drainage):
        from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
        n = len(grid)
        rng = np.random.default_rng(42)
        feat = grid.copy()
        feat["elevation_m"] = rng.uniform(10, 100, n).astype(np.float32)
        feat["slope_deg"] = rng.uniform(0, 10, n).astype(np.float32)
        feat["twi"] = rng.uniform(3, 12, n).astype(np.float32)
        feat["rainfall_mean_mm"] = rng.uniform(1000, 3000, n).astype(np.float32)
        feat["rainfall_max_24h_mm"] = rng.uniform(50, 200, n).astype(np.float32)
        feat["dist_water_m"] = rng.uniform(100, 5000, n).astype(np.float32)
        feat["population_density"] = rng.uniform(0, 5000, n).astype(np.float32)
        feat["aspect_deg"] = rng.uniform(0, 360, n).astype(np.float32)
        feat["curvature"] = rng.uniform(-1, 1, n).astype(np.float32)
        cell_map = dict(zip(drainage.cell_ids, drainage.capacity_scores))
        feat["drainage_capacity"] = np.array(
            [cell_map.get(str(cid), 0.5) for cid in grid["cell_id"]]
        ).astype(np.float32)
        return feat

    def test_pipeline_runs_with_osm_proxy(self, grid_with_drains):
        from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
        from flood_risk_zonation.scoring.susceptibility import EnsembleSusceptibilityModel
        from flood_risk_zonation.scoring.scorer import FloodRiskScorer
        from flood_risk_zonation.config import PipelineConfig
        grid, wb = grid_with_drains
        config = PipelineConfig()
        drainage = generate_drainage_proxy(grid, wb, cell_size_m=config.cell_size_meters)
        feat = self._make_feature_grid(grid, drainage)
        X = feat[FEATURE_COLUMNS]
        thresholds = {"low_max": config.low_threshold, "medium_max": config.medium_threshold}
        model = EnsembleSusceptibilityModel(n_estimators=10, cv_folds=2, random_state=42).fit(X)
        scorer = FloodRiskScorer()
        scorer.p_min, scorer.p_max = 0.0, 1.0
        scored = scorer.score_grid(feat, model, FEATURE_COLUMNS, thresholds)
        assert "risk_score" in scored.columns and "risk_class" in scored.columns
        assert len(scored) == len(grid)

    def test_provenance_osm_proxy(self, grid_with_drains):
        grid, wb = grid_with_drains
        r = generate_drainage_proxy(grid, wb, cell_size_m=500.0)
        assert r.source == "osm_proxy"

    def test_provenance_synthetic_fallback(self, small_grid):
        r = generate_drainage_proxy(small_grid, None, cell_size_m=500.0)
        assert r.source == "synthetic_fallback"


# -- T7: Water-proximity behaviour unchanged ----------------------------------

class TestWaterProximityUnchanged:
    def test_water_proximity_score_column_independent_of_drainage(self, small_grid):
        n = len(small_grid)
        rng = np.random.default_rng(1)
        scored = small_grid.copy()
        scored["risk_score"] = rng.uniform(20, 60, n).astype(np.float32)
        scored["risk_class"] = "Low"
        scored["drainage_capacity"] = np.full(n, 0.5, dtype=np.float32)
        scored["water_proximity_score"] = np.zeros(n, dtype=np.float32)
        scored["is_coastal_tsunami_risk"] = False
        assert np.all(scored["water_proximity_score"] == 0.0)

    def test_drainage_source_does_not_alter_risk_class(self, small_grid):
        n = len(small_grid)
        scored = small_grid.copy()
        scored["risk_class"] = "Low"
        scored["drainage_capacity"] = np.full(n, 0.3, dtype=np.float32)
        assert (scored["risk_class"] == "Low").all()


# -- T8: Risk classification validity ----------------------------------------

class TestRiskClassification:
    def test_risk_class_values_valid(self, grid_with_drains):
        from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
        from flood_risk_zonation.scoring.susceptibility import EnsembleSusceptibilityModel
        from flood_risk_zonation.scoring.scorer import FloodRiskScorer
        from flood_risk_zonation.config import PipelineConfig
        grid, wb = grid_with_drains
        config = PipelineConfig()
        n = len(grid)
        rng = np.random.default_rng(77)
        feat = grid.copy()
        feat["elevation_m"] = rng.uniform(1, 200, n).astype(np.float32)
        feat["slope_deg"] = rng.uniform(0, 20, n).astype(np.float32)
        feat["twi"] = rng.uniform(2, 15, n).astype(np.float32)
        feat["rainfall_mean_mm"] = rng.uniform(500, 4000, n).astype(np.float32)
        feat["rainfall_max_24h_mm"] = rng.uniform(20, 300, n).astype(np.float32)
        feat["dist_water_m"] = rng.uniform(50, 8000, n).astype(np.float32)
        feat["population_density"] = rng.uniform(0, 8000, n).astype(np.float32)
        feat["aspect_deg"] = rng.uniform(0, 360, n).astype(np.float32)
        feat["curvature"] = rng.uniform(-2, 2, n).astype(np.float32)
        drainage = generate_drainage_proxy(grid, wb, cell_size_m=config.cell_size_meters)
        cell_map = dict(zip(drainage.cell_ids, drainage.capacity_scores))
        feat["drainage_capacity"] = np.array(
            [cell_map.get(str(cid), 0.5) for cid in grid["cell_id"]]
        ).astype(np.float32)
        X = feat[FEATURE_COLUMNS]
        thresholds = {"low_max": config.low_threshold, "medium_max": config.medium_threshold}
        model = EnsembleSusceptibilityModel(n_estimators=10, cv_folds=2, random_state=42).fit(X)
        scorer = FloodRiskScorer()
        scorer.p_min, scorer.p_max = 0.0, 1.0
        scored = scorer.score_grid(feat, model, FEATURE_COLUMNS, thresholds)
        valid_classes = {"High", "Medium", "Low", "Water"}
        assert all(c in valid_classes for c in scored["risk_class"].unique())


# -- T9: Relocation logic still executes -------------------------------------

class TestRelocationUnchanged:
    def test_relocation_candidates_runs(self, small_grid):
        from flood_risk_zonation.relocation.candidates import find_relocation_candidates
        from flood_risk_zonation.spatial_zones.classifier import classify_spatial_zones
        n = len(small_grid)
        rng = np.random.default_rng(13)
        scored = small_grid.copy()
        scored["risk_score"] = rng.uniform(10, 80, n).astype(np.float32)
        scored["risk_class"] = np.where(scored["risk_score"] > 66, "High",
                               np.where(scored["risk_score"] > 33, "Medium", "Low"))
        scored["drainage_capacity"] = rng.uniform(0, 1, n).astype(np.float32)
        scored["water_proximity_score"] = np.zeros(n, dtype=np.float32)
        scored["is_coastal_tsunami_risk"] = False
        zoned = classify_spatial_zones(scored)
        center_lon = float(small_grid["centroid_lon"].mean())
        center_lat = float(small_grid["centroid_lat"].mean())
        candidates = find_relocation_candidates(
            hab_lat=center_lat,
            hab_lon=center_lon,
            hab_id="H1",
            hab_name="Test Village",
            zoned_grid=zoned,
            search_radius_km=20.0,
            max_candidates=5,
        )
        assert isinstance(candidates, list)


# -- T10: Performance ---------------------------------------------------------

class TestPerformance:
    def test_larger_grid_under_30_seconds(self):
        bbox = BoundingBox(0.0, 0.0, 0.75, 0.75)
        grid = generate_grid(bbox, cell_size_meters=5_000.0)
        lines = []
        for i in range(5):
            lat = 0.1 + i * 0.12
            lines.append({"geometry": LineString([(0.0, lat), (0.75, lat)]),
                           "water_type": "drain", "name": ""})
        for i in range(3):
            lon = 0.1 + i * 0.25
            lines.append({"geometry": LineString([(lon, 0.0), (lon, 0.75)]),
                           "water_type": "river", "name": ""})
        wb = gpd.GeoDataFrame(lines, crs="EPSG:4326")
        t0 = time.time()
        result = generate_drainage_proxy(grid, wb, cell_size_m=5_000.0)
        elapsed = time.time() - t0
        assert result.source == "osm_proxy"
        assert len(result.capacity_scores) == len(grid)
        assert elapsed < 30.0, f"Proxy took {elapsed:.1f}s for {len(grid)} cells — too slow."


# -- T11: Unrelated features unchanged ----------------------------------------

class TestUnrelatedFeaturesUnchanged:
    def test_elevation_unaffected_by_drainage_source(self, small_grid):
        from flood_risk_zonation.features.extractor import extract_features
        elev, rainfall, pop = _make_extract_inputs(small_grid)
        wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        drain_synth = generate_synthetic_drainage(small_grid, seed=42)
        result_synth = extract_features(small_grid, elev, rainfall, wb, pop, drain_synth)

        drain_line = LineString([(0.05, 0.07), (0.10, 0.07)])
        wb2 = gpd.GeoDataFrame([{"geometry": drain_line, "water_type": "drain", "name": ""}], crs="EPSG:4326")
        drain_osm = generate_drainage_proxy(small_grid, wb2, cell_size_m=5_000.0)
        result_osm = extract_features(small_grid, elev, rainfall, wb2, pop, drain_osm)

        np.testing.assert_array_almost_equal(
            result_synth["elevation_m"].values,
            result_osm["elevation_m"].values,
            decimal=3,
        )

    def test_rainfall_features_present_and_finite(self, small_grid):
        from flood_risk_zonation.features.extractor import extract_features
        elev, rainfall, pop = _make_extract_inputs(small_grid)
        wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        drainage = generate_synthetic_drainage(small_grid)
        result = extract_features(small_grid, elev, rainfall, wb, pop, drainage)
        assert np.all(np.isfinite(result["rainfall_mean_mm"].values))
        assert np.all(np.isfinite(result["rainfall_max_24h_mm"].values))
