"""Unit tests for popup content and export functions."""
import json
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
from flood_risk_zonation.visualization.export import export_csv, export_geojson, export_html
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder


def _make_scored_grid(n=3):
    rng = np.random.default_rng(0)
    geoms = [box(i * 0.01, 0, (i + 1) * 0.01, 0.01) for i in range(n)]
    data = {col: rng.random(n).astype("float32") for col in FEATURE_COLUMNS}
    data["cell_id"] = [f"0_{i}" for i in range(n)]
    data["centroid_lat"] = [0.005] * n
    data["centroid_lon"] = [i * 0.01 + 0.005 for i in range(n)]
    data["risk_score"] = [25.0, 55.0, 80.0]
    data["risk_class"] = ["Low", "Medium", "High"]
    return gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")


def test_popup_contains_required_fields():
    """add_popup_layer popup HTML must contain all 8 required fields."""
    grid = _make_scored_grid()
    builder = FloodRiskMapBuilder()
    import folium
    m = folium.Map(location=[0.005, 0.015], zoom_start=12)
    builder.add_popup_layer(m, grid)
    html = m._repr_html_()
    for field in ["Elevation", "Slope", "TWI", "Rainfall", "Dist", "Drainage"]:
        assert field in html, f"Field '{field}' missing from popup HTML"


def test_export_html_creates_file():
    """export_html must create a file at the specified path."""
    grid = _make_scored_grid()
    builder = FloodRiskMapBuilder()
    m = builder.build_choropleth_map(grid, center=(0.005, 0.015))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "map.html"
        export_html(m, path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "<html" in content.lower()


def test_export_geojson_creates_valid_file():
    """export_geojson must create a valid GeoJSON FeatureCollection."""
    grid = _make_scored_grid()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.geojson"
        export_geojson(grid, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0


def test_export_csv_has_correct_columns():
    """export_csv must create a CSV with all FEATURE_COLUMNS plus risk_score and risk_class."""
    grid = _make_scored_grid()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "output.csv"
        export_csv(grid, path)
        assert path.exists()
        df = pd.read_csv(path)
        for col in FEATURE_COLUMNS + ["risk_score", "risk_class"]:
            assert col in df.columns, f"Column '{col}' missing from CSV"
