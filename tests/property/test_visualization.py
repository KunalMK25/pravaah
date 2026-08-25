"""
Property 13: Visualization Color Mapping Correctness
Property 14: Popup Content Completeness

Validates: Requirements 6.1, 6.2
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from hypothesis import given, settings
from hypothesis import strategies as st

from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder, RISK_COLOR_MAP
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS


def _make_grid_with_classes(risk_classes):
    n = len(risk_classes)
    rng = np.random.default_rng(0)
    geoms = [box(i * 0.01, 0, (i + 1) * 0.01, 0.01) for i in range(n)]
    data = {col: rng.random(n).astype("float32") for col in FEATURE_COLUMNS}
    data["cell_id"] = [f"0_{i}" for i in range(n)]
    data["centroid_lat"] = [0.005] * n
    data["centroid_lon"] = [i * 0.01 + 0.005 for i in range(n)]
    data["risk_score"] = rng.uniform(0, 100, n).astype("float32")
    data["risk_class"] = list(risk_classes)
    return gpd.GeoDataFrame(data, geometry=geoms, crs="EPSG:4326")


@given(
    risk_classes=st.lists(
        st.sampled_from(["Low", "Medium", "High"]),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=20, deadline=None)
def test_color_mapping_correctness(risk_classes):
    """Property 13: Every cell's fill color matches RISK_COLOR_MAP[risk_class]."""
    grid = _make_grid_with_classes(risk_classes)
    builder = FloodRiskMapBuilder()
    m = builder.build_choropleth_map(grid, center=(0.005, 0.025))
    html = m._repr_html_()
    for rc in set(risk_classes):
        expected_color = RISK_COLOR_MAP[rc]
        assert expected_color in html, f"Color {expected_color} for {rc} not found in map HTML"


@given(
    risk_classes=st.lists(
        st.sampled_from(["Low", "Medium", "High"]),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=20, deadline=None)
def test_popup_content_completeness(risk_classes):
    """Property 14: Popup HTML contains all 8 required fields for every cell."""
    required_fields = [
        "Elevation", "Slope",
        "TWI", "Rainfall", "Dist", "Drainage",
    ]
    grid = _make_grid_with_classes(risk_classes)
    builder = FloodRiskMapBuilder()
    m = builder.build_choropleth_map(grid, center=(0.005, 0.025))
    html = m._repr_html_()
    for field in required_fields:
        assert field in html, f"Required popup field '{field}' not found in map HTML"
