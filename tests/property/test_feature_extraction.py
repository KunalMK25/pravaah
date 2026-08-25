"""
Property 6: Feature Extraction Produces Valid, Complete Feature Matrices

Validates: Requirements 3.1, 3.4
"""
import numpy as np
import geopandas as gpd
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.ingest.elevation import generate_synthetic_elevation
from flood_risk_zonation.ingest.rainfall import generate_synthetic_rainfall
from flood_risk_zonation.ingest.drainage import generate_synthetic_drainage
from flood_risk_zonation.ingest.population import load_population
from flood_risk_zonation.features.extractor import extract_features, FEATURE_COLUMNS


def make_test_inputs(seed: int = 42):
    """Create a minimal set of synthetic inputs for feature extraction."""
    bbox = BoundingBox(0.0, 0.0, 0.2, 0.2)
    grid = generate_grid(bbox, cell_size_meters=10000)
    elevation = generate_synthetic_elevation(bbox, resolution_m=1000, seed=seed)
    rainfall = generate_synthetic_rainfall(bbox, resolution_m=1000, seed=seed)
    water_bodies = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    population = load_population(bbox, data_dir="data/cache")  # uses synthetic fallback
    drainage = generate_synthetic_drainage(grid, seed=seed)
    return grid, elevation, rainfall, water_bodies, population, drainage


@given(seed=st.integers(0, 100))
@settings(max_examples=20, deadline=None)
def test_feature_extraction_no_nan_no_inf(seed):
    """
    Property 6: Feature matrix contains no NaN and no infinite values.
    """
    grid, elevation, rainfall, water_bodies, population, drainage = make_test_inputs(seed)
    result = extract_features(grid, elevation, rainfall, water_bodies, population, drainage)

    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature column: {col}"
        values = result[col].values
        assert not np.any(np.isnan(values)), f"NaN in column {col}"
        assert not np.any(np.isinf(values)), f"Inf in column {col}"


@given(seed=st.integers(0, 100))
@settings(max_examples=20, deadline=None)
def test_feature_extraction_physical_ranges(seed):
    """
    Property 6: All features are within physically valid ranges.
    """
    from flood_risk_zonation.features.extractor import FEATURE_RANGES
    grid, elevation, rainfall, water_bodies, population, drainage = make_test_inputs(seed)
    result = extract_features(grid, elevation, rainfall, water_bodies, population, drainage)

    for col, (lo, hi) in FEATURE_RANGES.items():
        if col in result.columns:
            values = result[col].values
            assert np.all(values >= lo - 1e-3), f"{col} has values below {lo}: min={values.min()}"
            assert np.all(values <= hi + 1e-3), f"{col} has values above {hi}: max={values.max()}"
