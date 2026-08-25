"""
Integration test: export pipeline.
Task 23 — Validates: Requirements 6.3
"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.visualization.export import export_csv, export_geojson, export_html
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder


@pytest.fixture(scope="module")
def pipeline_result():
    bbox = BoundingBox(0.0, 0.0, 0.5, 0.5)
    config = PipelineConfig(cell_size_meters=5000, rf_n_estimators=10, cv_folds=3, use_cache=False)
    return FloodRiskPipeline(config).run(bbox)


def test_html_export_creates_valid_file(pipeline_result):
    """export_html must create a file containing <html> tag."""
    builder = FloodRiskMapBuilder()
    center = pipeline_result.bounding_box.center
    m = builder.build_choropleth_map(pipeline_result.scored_grid, center=center)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "map.html"
        export_html(m, path)
        assert path.exists()
        assert "<html" in path.read_text(encoding="utf-8").lower()


def test_geojson_export_is_valid(pipeline_result):
    """export_geojson must produce a valid GeoJSON FeatureCollection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.geojson"
        export_geojson(pipeline_result.scored_grid, path)
        data = json.loads(path.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0


def test_csv_export_has_correct_columns(pipeline_result):
    """export_csv must include all FEATURE_COLUMNS plus risk_score and risk_class."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.csv"
        export_csv(pipeline_result.scored_grid, path)
        df = pd.read_csv(path)
        for col in FEATURE_COLUMNS + ["risk_score", "risk_class"]:
            assert col in df.columns, f"Column '{col}' missing from CSV"
