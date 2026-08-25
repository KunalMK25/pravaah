"""
Property 8: Distance to Water Non-Negativity

Validates: Requirements 3.3
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.features.hydrological import compute_distance_to_water


def make_small_grid(min_lon, min_lat, delta=0.1):
    bbox = BoundingBox(min_lon, min_lat, min_lon + delta, min_lat + delta)
    return generate_grid(bbox, cell_size_meters=5000.0)


@given(min_lon=st.floats(-10, 9.9), min_lat=st.floats(-10, 9.9))
@settings(max_examples=20, deadline=None)
def test_distance_to_water_non_negative_empty_water_bodies(min_lon, min_lat):
    """Property 8: Distances are non-negative with empty water bodies."""
    grid = make_small_grid(min_lon, min_lat)
    empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    distances = compute_distance_to_water(grid, empty_wb)
    assert np.all(distances >= 0)


@given(
    min_lon=st.floats(-10, 9.9),
    min_lat=st.floats(-10, 9.9),
    wb_offset=st.floats(0.01, 0.05),
)
@settings(max_examples=20, deadline=None)
def test_distance_to_water_non_negative_with_water_bodies(min_lon, min_lat, wb_offset):
    """Property 8: Distances are non-negative with arbitrary water body geometries."""
    grid = make_small_grid(min_lon, min_lat)
    water_polygon = box(
        min_lon + wb_offset, min_lat + wb_offset,
        min_lon + wb_offset + 0.02, min_lat + wb_offset + 0.02,
    )
    water_bodies = gpd.GeoDataFrame(geometry=[water_polygon], crs="EPSG:4326")
    distances = compute_distance_to_water(grid, water_bodies)
    assert np.all(distances >= 0)
