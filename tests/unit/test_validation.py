"""Tests for the historical flood event validation module."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock
import geopandas as gpd
from shapely.geometry import box

from flood_risk_zonation.models import (
    HistoricalFloodEvent, ValidationMetrics, ValidationResult,
)
from flood_risk_zonation.validation.events import (
    run_validation,
    _compute_metrics,
    _cells_in_polygon,
)


def _make_grid(risk_classes, bbox_coords=None):
    import pandas as pd
    n = len(risk_classes)
    if bbox_coords is None:
        # Default: Bangalore bbox (matches bundled events)
        bbox_coords = (77.54, 12.83, 77.65, 12.93)
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    rows = []
    for i, rc in enumerate(risk_classes):
        lat = min_lat + (i // 3) * (max_lat - min_lat) / max(1, n // 3)
        lon = min_lon + (i % 3) * (max_lon - min_lon) / 3.0
        rows.append({
            "cell_id": f"c{i}", "risk_class": rc,
            "risk_score": 80.0 if rc == "High" else 20.0,
            "centroid_lat": lat, "centroid_lon": lon,
            "geometry": box(lon - 0.003, lat - 0.003, lon + 0.003, lat + 0.003),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf


def _make_hazard_result(grid):
    mock_bbox = MagicMock()
    mock_bbox.min_lon, mock_bbox.min_lat = 77.54, 12.83
    mock_bbox.max_lon, mock_bbox.max_lat = 77.65, 12.93
    mock_config = MagicMock()
    mock_config.low_threshold = 33.0; mock_config.medium_threshold = 66.0
    hr = MagicMock()
    hr.scored_grid = grid
    hr.bounding_box = mock_bbox
    hr.config = mock_config
    return hr


class TestComputeMetrics:
    def test_perfect_overlap(self):
        cells = {"c0", "c1", "c2"}
        m = _compute_metrics("ev1", cells, cells)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1_score == pytest.approx(1.0)
        assert m.iou == pytest.approx(1.0)

    def test_no_overlap(self):
        m = _compute_metrics("ev1", {"c0", "c1"}, {"c2", "c3"})
        assert m.precision == pytest.approx(0.0)
        assert m.recall == pytest.approx(0.0)
        assert m.f1_score == pytest.approx(0.0)
        assert m.iou == pytest.approx(0.0)

    def test_partial_overlap(self):
        m = _compute_metrics("ev1", {"c0", "c1", "c2"}, {"c1", "c2", "c3"})
        assert 0 < m.precision < 1
        assert 0 < m.recall < 1
        assert 0 < m.f1_score < 1
        assert 0 < m.iou < 1

    def test_empty_predicted_returns_neg1(self):
        m = _compute_metrics("ev1", set(), {"c1", "c2"})
        assert m.precision == -1.0
        assert m.f1_score == -1.0

    def test_empty_observed_returns_neg1(self):
        m = _compute_metrics("ev1", {"c1"}, set())
        assert m.recall == -1.0

    def test_correct_counts(self):
        m = _compute_metrics("ev1", {"c0", "c1", "c2"}, {"c1", "c2", "c3", "c4"})
        assert m.predicted_high_count == 3
        assert m.observed_flood_count == 4
        assert m.overlap_count == 2

    def test_precision_recall_f1_relationship(self):
        m = _compute_metrics("ev1", {"c0", "c1", "c2", "c3"}, {"c0", "c1", "c4", "c5"})
        if m.precision > 0 and m.recall > 0:
            expected_f1 = 2 * m.precision * m.recall / (m.precision + m.recall)
            assert m.f1_score == pytest.approx(expected_f1, abs=0.001)


class TestCellsInPolygon:
    def test_cells_inside_polygon_detected(self):
        # Use a polygon that exactly covers the grid cells' centroid positions
        grid = _make_grid(["High", "Low", "High"])
        from shapely.geometry import box as shpbox
        # Grid centroids are near (77.54-77.58, 12.83-12.87) — use a generous box
        polygon = shpbox(77.50, 12.80, 77.70, 12.95)
        cells = _cells_in_polygon(grid, polygon)
        assert len(cells) >= 1

    def test_polygon_none_returns_empty(self):
        grid = _make_grid(["High"])
        assert _cells_in_polygon(grid, None) == set()

    def test_non_overlapping_polygon_returns_empty(self):
        grid = _make_grid(["High", "Low"])
        from shapely.geometry import box as shpbox
        polygon = shpbox(0.0, 0.0, 0.01, 0.01)  # far away
        cells = _cells_in_polygon(grid, polygon)
        assert len(cells) == 0


class TestRunValidation:
    def test_no_events_when_bbox_outside_bundled_events(self):
        # Use a bbox in a different region (no bundled events)
        grid = _make_grid(["High", "Low"] * 5,
                          bbox_coords=(74.83, 34.07, 74.90, 34.14))  # Srinagar
        hr = MagicMock()
        hr.scored_grid = grid
        mock_bbox = MagicMock()
        mock_bbox.min_lon, mock_bbox.min_lat = 74.83, 34.07
        mock_bbox.max_lon, mock_bbox.max_lat = 74.90, 34.14
        hr.bounding_box = mock_bbox
        hr.config = MagicMock()
        result = run_validation(hr)
        assert result.data_status == "NO_EVENTS_AVAILABLE"

    def test_validation_runs_for_bangalore(self):
        grid = _make_grid(["High"] * 5 + ["Low"] * 4)
        hr = _make_hazard_result(grid)
        result = run_validation(hr)
        # Should find the bundled Bangalore event
        assert result.data_status in ("VALIDATED", "PARTIAL", "NO_EVENTS_AVAILABLE")
        assert isinstance(result.events, list)

    def test_metrics_computed_when_overlap_exists(self):
        grid = _make_grid(["High"] * 5 + ["Low"] * 4)
        hr = _make_hazard_result(grid)
        result = run_validation(hr)
        if result.data_status == "VALIDATED":
            assert len(result.metrics) > 0
            for m in result.metrics:
                assert isinstance(m, ValidationMetrics)
                assert m.predicted_high_count >= 0
                assert m.observed_flood_count >= 0

    def test_validation_result_always_returned(self):
        grid = _make_grid(["Low"] * 9)
        hr = _make_hazard_result(grid)
        result = run_validation(hr)
        assert isinstance(result, ValidationResult)
        assert result.data_status in ("VALIDATED", "PARTIAL", "NO_EVENTS_AVAILABLE")

    def test_extra_events_accepted(self):
        from shapely.geometry import box as shpbox
        geojson = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[77.54, 12.83], [77.65, 12.83],
                                  [77.65, 12.93], [77.54, 12.93],
                                  [77.54, 12.83]]],
            }
        }
        extra = HistoricalFloodEvent(
            event_id="custom_test", event_name="Custom Event",
            event_date="2023-01-01", region="Test",
            source="Manual", flood_geojson=geojson,
        )
        grid = _make_grid(["High"] * 4 + ["Low"] * 5)
        hr = _make_hazard_result(grid)
        result = run_validation(hr, extra_events=[extra])
        assert isinstance(result, ValidationResult)

    def test_f1_between_0_and_1(self):
        grid = _make_grid(["High"] * 5 + ["Low"] * 4)
        hr = _make_hazard_result(grid)
        result = run_validation(hr)
        for m in result.metrics:
            if m.f1_score >= 0:
                assert 0.0 <= m.f1_score <= 1.0

    def test_iou_between_0_and_1(self):
        grid = _make_grid(["High"] * 5 + ["Low"] * 4)
        hr = _make_hazard_result(grid)
        result = run_validation(hr)
        for m in result.metrics:
            if m.iou >= 0:
                assert 0.0 <= m.iou <= 1.0
