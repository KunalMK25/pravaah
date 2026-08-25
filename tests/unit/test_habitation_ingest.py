"""Unit tests for habitation ingestion module."""
from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.habitation.ingest import (
    _parse_population,
    _osm_to_habitations,
    _fallback_habitations,
    _habitations_to_gdf,
    _gdf_to_habitations,
    load_habitations,
)
from flood_risk_zonation.models import Habitation


class TestParsePopulation:
    def test_valid_integer(self):
        assert _parse_population({"population": "5000"}) == 5000

    def test_comma_separated(self):
        assert _parse_population({"population": "1,200"}) == 1200

    def test_missing_tag(self):
        assert _parse_population({}) is None

    def test_zero(self):
        assert _parse_population({"population": "0"}) is None

    def test_negative(self):
        assert _parse_population({"population": "-100"}) is None

    def test_non_numeric(self):
        assert _parse_population({"population": "unknown"}) is None

    def test_float_string(self):
        # OSM sometimes has decimals — should fail gracefully
        result = _parse_population({"population": "1200.5"})
        assert result is None


class TestOsmToHabitations:
    def _make_osm_node(self, osm_id=1, lat=12.9, lon=77.6, place="village", name="Test", pop=None):
        tags = {"place": place, "name": name}
        if pop is not None:
            tags["population"] = str(pop)
        return {
            "type": "node",
            "id": osm_id,
            "lat": lat,
            "lon": lon,
            "tags": tags,
        }

    def test_basic_node_parsing(self):
        osm_data = {"elements": [self._make_osm_node()]}
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        assert len(habs) == 1
        assert habs[0].hab_type == "village"
        assert habs[0].name == "Test"
        assert habs[0].hab_id == "osm_1"

    def test_population_extracted(self):
        osm_data = {"elements": [self._make_osm_node(pop=2500)]}
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        assert habs[0].population == 2500

    def test_no_population_is_none(self):
        osm_data = {"elements": [self._make_osm_node()]}
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        assert habs[0].population is None

    def test_skips_non_nodes(self):
        osm_data = {"elements": [
            {"type": "way", "id": 99, "lat": 12.9, "lon": 77.6, "tags": {"place": "village"}},
        ]}
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        assert len(habs) == 0

    def test_skips_no_place_tag(self):
        osm_data = {"elements": [
            {"type": "node", "id": 1, "lat": 12.9, "lon": 77.6, "tags": {"name": "NoPlace"}},
        ]}
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        assert len(habs) == 0

    def test_multiple_nodes(self):
        osm_data = {"elements": [
            self._make_osm_node(osm_id=i, place="village", name=f"V{i}")
            for i in range(5)
        ]}
        habs = _osm_to_habitations(osm_data, source="osm_overpass")
        assert len(habs) == 5


class TestFallbackHabitations:
    def test_returns_3_habitations(self):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        habs = _fallback_habitations(bbox)
        assert len(habs) == 3

    def test_all_within_bbox(self):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        habs = _fallback_habitations(bbox)
        for h in habs:
            assert bbox.min_lat <= h.lat <= bbox.max_lat
            assert bbox.min_lon <= h.lon <= bbox.max_lon

    def test_source_is_fallback(self):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        habs = _fallback_habitations(bbox)
        for h in habs:
            assert h.source == "fallback"

    def test_population_is_none(self):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        habs = _fallback_habitations(bbox)
        for h in habs:
            assert h.population is None


class TestHabitationGdfRoundtrip:
    def test_roundtrip(self):
        habs = [
            Habitation("osm_1", "A", "village", 12.9, 77.6, "osm_overpass", population=1000),
            Habitation("osm_2", "B", "hamlet", 12.87, 77.57, "osm_overpass"),
        ]
        gdf = _habitations_to_gdf(habs)
        assert len(gdf) == 2
        assert gdf.crs.to_epsg() == 4326

        recovered = _gdf_to_habitations(gdf, source="osm_cache")
        assert len(recovered) == 2
        assert recovered[0].population == 1000
        assert recovered[1].population is None

    def test_empty_roundtrip(self):
        gdf = _habitations_to_gdf([])
        assert len(gdf) == 0


class TestLoadHabitations:
    def test_fallback_when_network_disabled(self, tmp_path):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.source == "fallback"
        assert len(result.habitations) > 0

    def test_cache_hit(self, tmp_path):
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        habs = [Habitation("osm_1", "A", "village", 12.9, 77.6, "osm_overpass")]
        import geopandas as gpd
        gdf = _habitations_to_gdf(habs)
        cache_file = tmp_path / "hab_77.5500_12.8400_77.6200_12.9100.geojson"
        gdf.to_file(str(cache_file), driver="GeoJSON")

        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=False)
        assert result.source == "osm_cache"
        assert len(result.habitations) == 1

    @patch("flood_risk_zonation.habitation.ingest._fetch")
    def test_live_fetch_success(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {
            "elements": [
                {
                    "type": "node", "id": 999, "lat": 12.88, "lon": 77.58,
                    "tags": {"place": "village", "name": "Mocked Village"},
                }
            ]
        }
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
        assert result.source == "osm_overpass"
        assert len(result.habitations) == 1
        assert result.habitations[0].name == "Mocked Village"

    @patch("flood_risk_zonation.habitation.ingest._fetch")
    def test_live_fetch_failure_returns_fallback(self, mock_fetch, tmp_path):
        mock_fetch.return_value = None
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
        assert result.source == "fallback"
        assert len(result.habitations) > 0

    @patch("flood_risk_zonation.habitation.ingest._fetch")
    def test_empty_osm_returns_fallback(self, mock_fetch, tmp_path):
        mock_fetch.return_value = {"elements": []}
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
        result = load_habitations(bbox, cache_dir=str(tmp_path), allow_network=True)
        assert result.source == "fallback"
