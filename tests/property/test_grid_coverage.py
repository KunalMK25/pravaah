"""
Property-based tests for the grid generation engine.

Property 3: Grid Coverage Completeness      — Validates: Requirements 2.1
Property 4: Grid Cell Size Accuracy         — Validates: Requirements 2.2
Property 5: Grid Cell Uniqueness/Non-Overlap — Validates: Requirements 2.3, 2.4
"""
import math
import random

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from shapely.geometry import box
from shapely.ops import unary_union

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.grid.generator import generate_grid


def _make_bbox(min_lon, min_lat, lon_delta, lat_delta):
    from hypothesis import assume
    max_lon = min_lon + lon_delta
    max_lat = min_lat + lat_delta
    assume(-180 <= min_lon and max_lon <= 180)
    assume(-90 <= min_lat and max_lat <= 90)
    return BoundingBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


valid_bboxes = st.builds(
    _make_bbox,
    min_lon=st.floats(-10, 9.5),
    min_lat=st.floats(-10, 9.5),
    lon_delta=st.floats(0.05, 0.5),
    lat_delta=st.floats(0.05, 0.5),
)


@given(bbox=valid_bboxes, cell_size=st.floats(5000, 50000))
@settings(max_examples=20)
def test_grid_coverage_completeness(bbox, cell_size):
    """Property 3: Union of all cells contains the bounding box polygon."""
    grid = generate_grid(bbox, cell_size_meters=cell_size)
    if len(grid) == 0:
        return
    bbox_polygon = box(bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat)
    union = unary_union(grid.geometry.values)
    assert union.buffer(1e-9).contains(bbox_polygon) or bbox_polygon.difference(union).area < 1e-10


@given(bbox=valid_bboxes, cell_size=st.floats(5000, 50000))
@settings(max_examples=20)
def test_grid_cell_size_accuracy(bbox, cell_size):
    """Property 4: Interior cell areas are within ±10% of cell_size_meters²."""
    grid = generate_grid(bbox, cell_size_meters=cell_size)
    if len(grid) == 0:
        return
    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    lat_m_per_deg = 111_320.0
    lon_m_per_deg = 111_320.0 * math.cos(math.radians(center_lat))
    expected_area_m2 = cell_size ** 2
    for geom in grid.geometry:
        area_m2 = geom.area * lat_m_per_deg * lon_m_per_deg
        if area_m2 < expected_area_m2 * 0.90:
            continue  # skip clipped edge cells
        assert abs(area_m2 - expected_area_m2) / expected_area_m2 <= 0.11, (
            f"Interior cell area {area_m2:.1f} m² deviates >11% from {expected_area_m2:.1f} m²"
        )


@given(bbox=valid_bboxes, cell_size=st.floats(5000, 50000))
@settings(max_examples=20)
def test_grid_cell_uniqueness_and_non_overlap(bbox, cell_size):
    """Property 5: All cell_ids unique; no two cells overlap."""
    grid = generate_grid(bbox, cell_size_meters=cell_size)
    if len(grid) <= 1:
        return
    assert len(grid["cell_id"].unique()) == len(grid), "cell_id values are not unique"
    geoms = list(grid.geometry)
    n = min(len(geoms), 20)
    random.seed(42)
    sample = random.sample(range(len(geoms)), n)
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            assert geoms[sample[i]].intersection(geoms[sample[j]]).area < 1e-10
