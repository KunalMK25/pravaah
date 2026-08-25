"""
Unit tests for hydrological feature computation.
"""
import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import Point, Polygon

from flood_risk_zonation.features.hydrological import (
    compute_distance_to_water,
    compute_drainage_density,
    MAX_DISTANCE_M,
)
from flood_risk_zonation.models import DrainageDataset
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.config import BoundingBox


@pytest.fixture
def small_grid():
    """A small 3x3 grid at the equator."""
    bbox = BoundingBox(0.0, 0.0, 0.1, 0.1)
    return generate_grid(bbox, cell_size_meters=5000.0)


def test_distance_to_water_returns_max_when_no_water_bodies(small_grid):
    """compute_distance_to_water returns MAX_DISTANCE_M when no water bodies present."""
    empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    distances = compute_distance_to_water(small_grid, empty_wb)
    assert np.all(distances == MAX_DISTANCE_M), \
        f"Expected all distances = {MAX_DISTANCE_M}, got {distances}"


def test_distance_to_water_returns_zero_for_centroid_on_water_body(small_grid):
    """compute_distance_to_water returns 0 for a centroid that lies on a water body."""
    # Get the first cell's centroid
    first_row = small_grid.iloc[0]
    centroid_lon = first_row["centroid_lon"]
    centroid_lat = first_row["centroid_lat"]

    # Create a water body polygon that contains the centroid
    delta = 0.01
    water_polygon = Polygon([
        (centroid_lon - delta, centroid_lat - delta),
        (centroid_lon + delta, centroid_lat - delta),
        (centroid_lon + delta, centroid_lat + delta),
        (centroid_lon - delta, centroid_lat + delta),
        (centroid_lon - delta, centroid_lat - delta),
    ])
    water_bodies = gpd.GeoDataFrame(geometry=[water_polygon], crs="EPSG:4326")

    distances = compute_distance_to_water(small_grid, water_bodies)
    # The first cell's centroid is inside the water body → distance should be 0
    assert distances[0] == pytest.approx(0.0, abs=1.0), \
        f"Expected distance ~0 for centroid inside water body, got {distances[0]}"


def test_distance_to_water_all_non_negative(small_grid):
    """All computed distances must be non-negative."""
    water_polygon = Polygon([(0.02, 0.02), (0.04, 0.02), (0.04, 0.04), (0.02, 0.04)])
    water_bodies = gpd.GeoDataFrame(geometry=[water_polygon], crs="EPSG:4326")
    distances = compute_distance_to_water(small_grid, water_bodies)
    assert np.all(distances >= 0), f"Negative distances found: {distances[distances < 0]}"


def test_compute_drainage_density_matches_cell_ids(small_grid):
    """compute_drainage_density correctly maps scores to cells by cell_id."""
    cell_ids = list(small_grid["cell_id"].astype(str))
    scores = np.linspace(0.1, 0.9, len(cell_ids)).astype(np.float32)
    drainage = DrainageDataset(capacity_scores=scores, cell_ids=cell_ids)

    result = compute_drainage_density(small_grid, drainage)
    assert len(result) == len(small_grid)
    assert np.allclose(result, scores, atol=1e-5)
