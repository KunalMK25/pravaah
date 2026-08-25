"""Unit tests for FloodRiskMapBuilder."""
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import pytest
from shapely.geometry import box

from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder, RISK_COLOR_MAP
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS


def _make_scored_grid(n=5):
    rng = np.random.default_rng(0)
    geoms = [box(i * 0.01, 0, (i + 1) * 0.01, 0.01) for i in range(n)]
    data = {col: rng.random(n).astype("float32") for col in FEATURE_COLUMNS}
    data["cell_id"] = [f"0_{i}" for i in range(n)]
    data["centroid_lat"] = [0.005] * n
    data["centroid_lon"] = [i * 0.01 + 0.005 for i in range(n)]
    data["risk_score"] = rng.uniform(0, 100, n).astype("float32")
    data["risk_class"] = ["Low", "Medium", "High", "Low", "High"][:n]
    return gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")


def test_build_choropleth_map_returns_folium_map():
    grid = _make_scored_grid()
    builder = FloodRiskMapBuilder()
    m = builder.build_choropleth_map(grid, center=(0.005, 0.025), zoom_start=10)
    assert isinstance(m, folium.Map)


def test_map_contains_layer_control():
    import tempfile, pathlib
    grid = _make_scored_grid()
    builder = FloodRiskMapBuilder()
    m = builder.build_choropleth_map(grid, center=(0.005, 0.025))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "map.html"
        m.save(str(path))
        html = path.read_text(encoding="utf-8")
    # Folium 0.20 uses "layer_control_" prefix in the saved HTML
    assert "layer_control" in html or "LayerControl" in html or "L.control.layers" in html


def test_risk_color_map_has_all_classes():
    assert "Low" in RISK_COLOR_MAP
    assert "Medium" in RISK_COLOR_MAP
    assert "High" in RISK_COLOR_MAP
    assert RISK_COLOR_MAP["Low"] == "#2ecc71"
    assert RISK_COLOR_MAP["Medium"] == "#f39c12"
    assert RISK_COLOR_MAP["High"] == "#e74c3c"
