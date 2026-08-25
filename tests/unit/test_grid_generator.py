"""
Unit tests for the grid generation engine.

Tests cover:
- Approximate cell count for a known bounding box
- Uniqueness of cell_id values
- Validity of cell geometries
- Return type
- Required columns
"""
import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.grid.generator import generate_grid


@pytest.fixture
def equator_bbox():
    """1×1 degree bounding box at the equator."""
    return BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=1.0, max_lat=1.0)


def test_grid_cell_count_approximate(equator_bbox):
    """
    For a 1×1 degree bbox at the equator with 500m cells, the cell count
    should be between 30,000 and 60,000.

    At the equator, 1 degree ≈ 111,320m, so ~222 cells per side → ~49,284 total.
    """
    gdf = generate_grid(equator_bbox, cell_size_meters=500.0)
    assert 30_000 <= len(gdf) <= 60_000, (
        f"Expected cell count between 30,000 and 60,000, got {len(gdf)}"
    )


def test_grid_cell_ids_unique(equator_bbox):
    """All cell_id values must be unique strings."""
    gdf = generate_grid(equator_bbox, cell_size_meters=5000.0)
    # Check values are string-like (works with both object and pandas StringDtype)
    assert all(isinstance(v, str) for v in gdf["cell_id"]), \
        "cell_id values should be strings"
    assert len(gdf["cell_id"].unique()) == len(gdf), (
        "cell_id values are not all unique"
    )


def test_grid_cell_geometries_valid(equator_bbox):
    """All geometries must be valid Shapely polygons."""
    gdf = generate_grid(equator_bbox, cell_size_meters=5000.0)
    assert len(gdf) > 0, "Grid should contain at least one cell"
    for geom in gdf.geometry:
        assert isinstance(geom, Polygon), f"Expected Polygon, got {type(geom)}"
        assert geom.is_valid, f"Geometry is not valid: {geom}"


def test_grid_returns_geodataframe(equator_bbox):
    """Return type must be gpd.GeoDataFrame."""
    result = generate_grid(equator_bbox, cell_size_meters=5000.0)
    assert isinstance(result, gpd.GeoDataFrame), (
        f"Expected GeoDataFrame, got {type(result)}"
    )


def test_grid_has_required_columns(equator_bbox):
    """Output must include cell_id, geometry, centroid_lat, centroid_lon columns."""
    gdf = generate_grid(equator_bbox, cell_size_meters=5000.0)
    required_columns = {"cell_id", "geometry", "centroid_lat", "centroid_lon"}
    missing = required_columns - set(gdf.columns)
    assert not missing, f"Missing required columns: {missing}"
