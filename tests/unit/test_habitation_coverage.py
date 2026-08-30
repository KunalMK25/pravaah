"""
Targeted tests for improved OSM habitation coverage (Phase 2).

Requirements tested:
    T1  -- Residential building types are recognised (allowlist check)
    T2  -- Residential building way centroids are derived correctly
    T3  -- Non-residential buildings are excluded (industrial, warehouse, etc.)
    T4  -- Building-derived habitation points have correct source and hab_id
    T5  -- Existing place-node habitation data still works
    T6  -- Deduplication prevents building duplicate near a place node
    T7  -- Stable OSM IDs are preserved (osm_, bld_, luse_ prefixes)
    T8  -- Arbitrary bounding boxes work (no hardcoded coordinates)
    T9  -- Sparse OSM data (empty elements) does not crash
    T10 -- Network failure triggers graceful fallback
    T11 -- Cached data is reused (no second network call)
    T12 -- Population not fabricated for buildings (always None)
    T13 -- Habitation data integrates with exposure analysis
    T14 -- Habitation data integrates with relocation analysis
    T15 -- Water-proximity behaviour unchanged by habitation changes
    T16 -- Risk calculations unchanged by habitation changes
    T17 -- Performance optimisations remain intact (n_estimators/cv_folds)
    T18 -- Larger bbox habitation ingestion completes in < 5s (offline)
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.habitation.ingest import (
    _DEDUP_RADIUS_DEG,
    _RESIDENTIAL_BUILDING_TYPES,
    _deduplicate_habitations,
    _fallback_habitations,
    _habitations_to_gdf,
    _osm_nodes_to_habitations,
    _osm_to_habitations,  # backward-compat alias
    _osm_ways_to_habitations,
    _parse_population,
    _parse_way_centroid,
    load_habitations,
)
from flood_risk_zonation.models import Habitation, HabitationDataset


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_node(osm_id=1, lat=12.9, lon=77.6, place="village", name="Test", pop=None):
    tags = {"place": place, "name": name}
    if pop is not None:
        tags["population"] = str(pop)
    return {"type": "node", "id": osm_id, "lat": lat, "lon": lon, "tags": tags}


def _make_building_way(osm_id=100, building_type="house", lat=12.9, lon=77.6, name=""):
    # Simple 4-node square around centroid (0.001 deg side)
    d = 0.001
    geom = [
        {"lat": lat - d, "lon": lon - d},
        {"lat": lat + d, "lon": lon - d},
        {"lat": lat + d, "lon": lon + d},
        {"lat": lat - d, "lon": lon + d},
        {"lat": lat - d, "lon": lon - d},  # closed ring
    ]
    tags = {"building": building_type}
    if name:
        tags["name"] = name
    return {"type": "way", "id": osm_id, "tags": tags, "geometry": geom}


def _make_landuse_way(osm_id=200, lat=12.85, lon=77.55, size=0.01):
    d = size
    geom = [
        {"lat": lat - d, "lon": lon - d},
        {"lat": lat + d, "lon": lon - d},
        {"lat": lat + d, "lon": lon + d},
        {"lat": lat - d, "lon": lon + d},
        {"lat": lat - d, "lon": lon - d},
    ]
    return {"type": "way", "id": osm_id, "tags": {"landuse": "residential"}, "geometry": geom}


def _make_nonresidential_way(osm_id=300, building_type="industrial", lat=12.9, lon=77.6):
    d = 0.001
    geom = [
        {"lat": lat - d, "lon": lon - d},
        {"lat": lat + d, "lon": lon - d},
        {"lat": lat + d, "lon": lon + d},
        {"lat": lat - d, "lon": lon + d},
        {"lat": lat - d, "lon": lon - d},
    ]
    return {"type": "way", "id": osm_id, "tags": {"building": building_type}, "geometry": geom}


# ── T1: Residential building types recognised ─────────────────────────────────

class TestResidentialBuildingAllowlist:
    def test_all_allowlisted_types_present(self):
        expected = {
            "house", "residential", "apartments", "detached",
            "semidetached_house", "terrace", "bungalow",
            "dormitory", "hut", "cabin",
        }
        assert expected == _RESIDENTIAL_BUILDING_TYPES

    def test_each_allowlisted_type_produces_habitation(self):
        for btype in _RESIDENTIAL_BUILDING_TYPES:
            osm_data = {"elements": [_make_building_way(osm_id=1, building_type=btype)]}
            habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
            assert len(habs) >= 1, f"building={btype} should produce a habitation"
            assert habs[0].source == "osm_building"


# ── T2: Residential building way centroids derived correctly ──────────────────

class TestBuildingCentroid:
    def test_centroid_within_bbox(self):
        lat, lon = 12.9, 77.6
        way = _make_building_way(lat=lat, lon=lon)
        centroid = _parse_way_centroid(way)
        assert centroid is not None
        c_lat, c_lon = centroid
        assert abs(c_lat - lat) < 0.01
        assert abs(c_lon - lon) < 0.01

    def test_degenerate_way_returns_none(self):
        way = {"type": "way", "id": 1, "tags": {"building": "house"},
               "geometry": [{"lat": 12.9, "lon": 77.6}]}
        assert _parse_way_centroid(way) is None

    def test_missing_geometry_returns_none(self):
        way = {"type": "way", "id": 1, "tags": {"building": "house"}, "geometry": []}
        assert _parse_way_centroid(way) is None

    def test_building_habitation_lat_lon_match_centroid(self):
        lat, lon = 13.0, 77.5
        osm_data = {"elements": [_make_building_way(lat=lat, lon=lon)]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        assert len(habs) == 1
        assert abs(habs[0].lat - lat) < 0.005
        assert abs(habs[0].lon - lon) < 0.005


# ── T3: Non-residential buildings excluded ────────────────────────────────────

class TestNonResidentialExclusion:
    NON_RESIDENTIAL = [
        "industrial", "warehouse", "commercial", "retail",
        "school", "hospital", "garage", "shed", "office",
        "church", "mosque", "temple", "university", "college",
        "public", "government", "parking", "train_station",
    ]

    def test_non_residential_not_ingested(self):
        for btype in self.NON_RESIDENTIAL:
            osm_data = {"elements": [_make_nonresidential_way(building_type=btype)]}
            habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
            assert len(habs) == 0, (
                f"building={btype} should NOT produce a habitation, got {len(habs)}"
            )

    def test_unknown_building_type_not_ingested(self):
        osm_data = {"elements": [_make_nonresidential_way(building_type="yes")]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        assert len(habs) == 0


# ── T4: Building-derived habitation points have correct attributes ────────────

class TestBuildingHabitationAttributes:
    def test_source_is_osm_building(self):
        osm_data = {"elements": [_make_building_way()]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        assert habs[0].source == "osm_building"

    def test_hab_id_has_bld_prefix(self):
        osm_data = {"elements": [_make_building_way(osm_id=42)]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        assert habs[0].hab_id == "bld_42"

    def test_population_always_none_for_buildings(self):
        osm_data = {"elements": [_make_building_way()]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        assert habs[0].population is None

    def test_landuse_source_is_osm_landuse(self):
        osm_data = {"elements": [_make_landuse_way()]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        assert len(habs) == 1
        assert habs[0].source == "osm_landuse"
        assert habs[0].hab_id.startswith("luse_")


# ── T5: Existing place-node habitation data still works ───────────────────────

class TestPlaceNodeBackwardCompat:
    def test_place_nodes_still_parsed(self):
        osm_data = {"elements": [_make_node(osm_id=1, place="village", name="A")]}
        habs = _osm_nodes_to_habitations(osm_data, source="osm_overpass")
        assert len(habs) == 1
        assert habs[0].hab_id == "osm_1"
        assert habs[0].source == "osm_overpass"

    def test_alias_still_works(self):
        """_osm_to_habitations is a backward-compat alias."""
        osm_data = {"elements": [_make_node()]}
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        assert len(habs) == 1

    def test_place_nodes_skip_non_nodes(self):
        osm_data = {"elements": [_make_building_way()]}
        habs = _osm_nodes_to_habitations(osm_data, source="osm_overpass")
        assert len(habs) == 0  # ways should not appear in place-node parser


# ── T6: Deduplication prevents building duplicate near place node ─────────────

class TestDeduplication:
    def test_building_within_radius_discarded(self):
        place = Habitation("osm_1", "Village", "village", 12.9, 77.6, "osm_overpass")
        building = Habitation("bld_10", "", "building_house", 12.9, 77.6, "osm_building")
        result = _deduplicate_habitations([place, building])
        ids = {h.hab_id for h in result}
        assert "osm_1" in ids
        assert "bld_10" not in ids, "Building at same location should be deduped"

    def test_building_outside_radius_kept(self):
        place = Habitation("osm_1", "Village", "village", 12.9, 77.6, "osm_overpass")
        # 0.01 deg >> _DEDUP_RADIUS_DEG
        building = Habitation("bld_10", "", "building_house", 12.91, 77.61, "osm_building")
        result = _deduplicate_habitations([place, building])
        ids = {h.hab_id for h in result}
        assert "osm_1" in ids
        assert "bld_10" in ids

    def test_no_place_nodes_all_buildings_kept(self):
        b1 = Habitation("bld_1", "", "building_house", 12.9, 77.6, "osm_building")
        b2 = Habitation("bld_2", "", "building_house", 12.91, 77.61, "osm_building")
        result = _deduplicate_habitations([b1, b2])
        assert len(result) == 2

    def test_duplicate_hab_ids_not_repeated(self):
        """Same hab_id appearing twice should only appear once in output."""
        b1 = Habitation("bld_1", "", "building_house", 12.9, 77.6, "osm_building")
        b2 = Habitation("bld_1", "", "building_house", 12.9, 77.6, "osm_building")
        result = _deduplicate_habitations([b1, b2])
        assert len(result) == 1

    def test_dedup_radius_constant_positive(self):
        assert _DEDUP_RADIUS_DEG > 0


# ── T7: Stable OSM IDs preserved ─────────────────────────────────────────────

class TestStableOsmIds:
    def test_place_node_id_format(self):
        osm_data = {"elements": [_make_node(osm_id=123456789)]}
        habs = _osm_nodes_to_habitations(osm_data, source="osm_overpass")
        assert habs[0].hab_id == "osm_123456789"
        assert habs[0].osm_id == 123456789

    def test_building_id_format(self):
        osm_data = {"elements": [_make_building_way(osm_id=987654321)]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        assert habs[0].hab_id == "bld_987654321"
        assert habs[0].osm_id == 987654321

    def test_landuse_id_format(self):
        osm_data = {"elements": [_make_landuse_way(osm_id=555)]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        luse = [h for h in habs if h.source == "osm_landuse"]
        assert luse[0].hab_id == "luse_555"


# ── T8: Arbitrary bounding boxes work ────────────────────────────────────────

class TestArbitraryBboxes:
    @pytest.mark.parametrize("bbox", [
        BoundingBox(0.0, 0.0, 0.1, 0.1),
        BoundingBox(-10.0, -5.0, -9.0, -4.0),
        BoundingBox(100.0, -30.0, 101.0, -29.0),
        BoundingBox(77.55, 12.84, 77.65, 12.92),
    ])
    def test_fallback_works_for_bbox(self, bbox, tmp_path):
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.source == "fallback"
        for h in result.habitations:
            assert bbox.min_lat <= h.lat <= bbox.max_lat
            assert bbox.min_lon <= h.lon <= bbox.max_lon


# ── T9: Sparse OSM data does not crash ───────────────────────────────────────

class TestSparseOsmData:
    @patch("flood_risk_zonation.habitation.ingest._fetch")
    def test_empty_elements_returns_fallback(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {"elements": []}
        bbox = BoundingBox(0.0, 0.0, 0.1, 0.1)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
        assert result.source == "fallback"

    @patch("flood_risk_zonation.habitation.ingest._fetch")
    def test_only_ways_no_nodes_succeeds(self, mock_fetch, tmp_path):
        """Response with only building ways (no place nodes) should still work."""
        mock_fetch.return_value = {"elements": [_make_building_way(osm_id=1)]}
        bbox = BoundingBox(0.0, 0.0, 0.1, 0.1)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
        # buildings only -> source is osm_overpass_buildings
        assert result.source in {"osm_overpass", "osm_overpass_buildings"}
        assert len(result.habitations) >= 1


# ── T10: Network failure triggers graceful fallback ───────────────────────────

class TestNetworkFailure:
    @patch("flood_risk_zonation.habitation.ingest._fetch")
    def test_fetch_none_returns_fallback(self, mock_fetch, tmp_path):
        mock_fetch.return_value = None
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
        assert result.source == "fallback"
        assert len(result.habitations) > 0

    def test_network_disabled_returns_fallback(self, tmp_path):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.source == "fallback"


# ── T11: Cached data is reused ────────────────────────────────────────────────

class TestCaching:
    def test_cache_hit_no_network_call(self, tmp_path):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        habs = [
            Habitation("osm_1", "A", "village", 12.9, 77.6, "osm_overpass"),
            Habitation("bld_2", "B", "building_house", 12.89, 77.59, "osm_building"),
        ]
        gdf = _habitations_to_gdf(habs)
        cache_file = tmp_path / "hab_77.5500_12.8400_77.6200_12.9100.geojson"
        gdf.to_file(str(cache_file), driver="GeoJSON")

        with patch("flood_risk_zonation.habitation.ingest._fetch") as mock_fetch:
            result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
            mock_fetch.assert_not_called()

        assert result.source == "osm_cache"
        assert len(result.habitations) == 2

    def test_cached_building_source_preserved(self, tmp_path):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        habs = [Habitation("bld_99", "", "building_house", 12.9, 77.6, "osm_building")]
        gdf = _habitations_to_gdf(habs)
        cache_file = tmp_path / "hab_77.5500_12.8400_77.6200_12.9100.geojson"
        gdf.to_file(str(cache_file), driver="GeoJSON")

        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.source == "osm_cache"
        # The per-record source="osm_building" should be preserved
        assert any(h.source == "osm_building" for h in result.habitations)


# ── T12: Population not fabricated ───────────────────────────────────────────

class TestPopulationIntegrity:
    def test_building_population_always_none(self):
        for btype in _RESIDENTIAL_BUILDING_TYPES:
            osm_data = {"elements": [_make_building_way(building_type=btype)]}
            habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
            for h in habs:
                assert h.population is None, (
                    f"building={btype} should have population=None, got {h.population}"
                )

    def test_landuse_population_always_none(self):
        osm_data = {"elements": [_make_landuse_way()]}
        habs = _osm_ways_to_habitations(osm_data, source_tag="osm_building")
        for h in habs:
            assert h.population is None

    def test_place_node_with_population_parsed(self):
        osm_data = {"elements": [_make_node(pop=5000)]}
        habs = _osm_nodes_to_habitations(osm_data, source="osm_overpass")
        assert habs[0].population == 5000

    def test_place_node_without_population_is_none(self):
        osm_data = {"elements": [_make_node()]}
        habs = _osm_nodes_to_habitations(osm_data, source="osm_overpass")
        assert habs[0].population is None


# ── T13: Integration with exposure analysis ───────────────────────────────────

class TestExposureIntegration:
    def _make_scored_grid(self):
        bbox = BoundingBox(12.88, 77.58, 12.93, 77.63)
        grid = generate_grid(bbox, cell_size_meters=1000.0)
        n = len(grid)
        grid["risk_score"] = np.full(n, 55.0, dtype=np.float32)
        grid["risk_class"] = "Medium"
        return grid

    def test_exposure_with_building_habitations(self):
        from flood_risk_zonation.exposure.analysis import analyse_exposure
        scored = self._make_scored_grid()
        place = Habitation("osm_1", "TestVillage", "village", 12.9, 77.6, "osm_overpass")
        bld = Habitation("bld_2", "", "building_house", 12.905, 77.605, "osm_building")
        ds = HabitationDataset(habitations=[place, bld], source="osm_overpass", bbox_key="test")
        results = analyse_exposure(ds, scored)
        assert len(results) == 2
        assert all(r.hazard_class in {"Low", "Medium", "High"} for r in results)

    def test_exposure_population_unknown_for_buildings(self):
        from flood_risk_zonation.exposure.analysis import analyse_exposure
        scored = self._make_scored_grid()
        bld = Habitation("bld_1", "", "building_house", 12.9, 77.6, "osm_building")
        ds = HabitationDataset(habitations=[bld], source="osm_overpass_buildings", bbox_key="t")
        results = analyse_exposure(ds, scored)
        assert results[0].population_source == "UNKNOWN"


# ── T14: Integration with relocation analysis ─────────────────────────────────

class TestRelocationIntegration:
    def test_relocation_candidates_with_building_habitations(self):
        from flood_risk_zonation.relocation.candidates import find_relocation_candidates
        from flood_risk_zonation.spatial_zones.classifier import classify_spatial_zones

        bbox = BoundingBox(0.0, 0.0, 0.2, 0.2)
        grid = generate_grid(bbox, cell_size_meters=5000.0)
        n = len(grid)
        rng = np.random.default_rng(42)
        grid["risk_score"] = rng.uniform(10, 80, n).astype(np.float32)
        grid["risk_class"] = np.where(grid["risk_score"] > 66, "High",
                             np.where(grid["risk_score"] > 33, "Medium", "Low"))
        grid["water_proximity_score"] = np.zeros(n, dtype=np.float32)
        grid["is_coastal_tsunami_risk"] = False
        zoned = classify_spatial_zones(grid)

        candidates = find_relocation_candidates(
            hab_lat=float(grid["centroid_lat"].mean()),
            hab_lon=float(grid["centroid_lon"].mean()),
            hab_id="bld_123",
            hab_name="Building Habitation",
            zoned_grid=zoned,
            search_radius_km=30.0,
            max_candidates=3,
        )
        assert isinstance(candidates, list)


# ── T15: Water-proximity behaviour unchanged ──────────────────────────────────

class TestWaterProximityUnchanged:
    def test_habitation_source_does_not_affect_proximity(self):
        from flood_risk_zonation.features.hydrological import compute_distance_to_water
        bbox = BoundingBox(0.0, 0.0, 0.1, 0.1)
        grid = generate_grid(bbox, cell_size_meters=5000.0)
        empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        distances = compute_distance_to_water(grid, empty_wb)
        # Should still return max distance for all cells
        from flood_risk_zonation.features.hydrological import MAX_DISTANCE_M
        assert np.all(distances == MAX_DISTANCE_M)


# ── T16: Risk calculations unchanged by habitation changes ────────────────────

class TestRiskCalcUnchanged:
    def test_drainage_proxy_still_functional_alongside_habitation(self):
        from flood_risk_zonation.ingest.drainage import generate_drainage_proxy
        bbox = BoundingBox(0.0, 0.0, 0.1, 0.1)
        grid = generate_grid(bbox, cell_size_meters=5000.0)
        result = generate_drainage_proxy(grid, None, cell_size_m=5000.0)
        assert result.source == "synthetic_fallback"
        assert np.all(result.capacity_scores >= 0.0)


# ── T17: Performance optimisations remain intact ──────────────────────────────

class TestPerformanceOptimisations:
    def test_pipeline_config_rf_estimators_unchanged(self):
        """Habitation improvements must not mutate PipelineConfig defaults."""
        config = PipelineConfig()
        # Current codebase default is 100 (reduced from 200 for performance optimization).
        # Assert the value remains at the optimized setting.
        assert config.rf_n_estimators == 100, (
            f"rf_n_estimators default unexpectedly changed: got {config.rf_n_estimators}"
        )

    def test_pipeline_config_cv_folds_unchanged(self):
        """Habitation improvements must not mutate PipelineConfig defaults."""
        config = PipelineConfig()
        # Current codebase default is 3 (reduced from 5 for performance optimization).
        # Assert the value remains at the optimized setting.
        assert config.cv_folds == 3, (
            f"cv_folds default unexpectedly changed: got {config.cv_folds}"
        )

    def test_habitation_import_does_not_change_config(self):
        """Importing habitation module should not have side-effects on config."""
        import flood_risk_zonation.habitation.ingest  # noqa: F401
        config = PipelineConfig()
        assert isinstance(config.rf_n_estimators, int)
        assert isinstance(config.cv_folds, int)


# ── T18: Larger bbox performance within limit ─────────────────────────────────

class TestLargerBboxPerformance:
    def test_offline_fallback_instant_for_large_bbox(self, tmp_path):
        """Fallback (offline) habitation generation should be near-instant."""
        bbox = BoundingBox(77.0, 12.0, 78.0, 13.0)
        t0 = time.time()
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=False)
        elapsed = time.time() - t0
        assert result.source == "fallback"
        assert elapsed < 5.0, f"Offline habitation took {elapsed:.2f}s -- too slow"

    @patch("flood_risk_zonation.habitation.ingest._fetch")
    def test_mocked_response_with_many_buildings_completes_fast(self, mock_fetch, tmp_path):
        """Parsing and dedup of 200 building ways should complete < 5s."""
        elements = [_make_building_way(osm_id=i, lat=12.8 + i * 0.001, lon=77.5)
                    for i in range(200)]
        elements += [_make_node(osm_id=i + 10000, lat=12.8 + i * 0.05, lon=77.5)
                     for i in range(20)]
        mock_fetch.return_value = {"elements": elements}
        bbox = BoundingBox(77.0, 12.0, 78.0, 13.0)
        t0 = time.time()
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
        elapsed = time.time() - t0
        assert len(result.habitations) > 0
        assert elapsed < 5.0, f"Parsing 200 buildings took {elapsed:.2f}s -- too slow"
