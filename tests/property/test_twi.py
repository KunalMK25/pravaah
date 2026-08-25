"""
Property 7: TWI Formula Correctness

Validates: Requirements 3.2
"""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from flood_risk_zonation.features.terrain import compute_twi


@given(size=st.integers(3, 8), cell_size=st.floats(10.0, 1000.0))
@settings(max_examples=20, deadline=None)
def test_twi_flat_terrain_is_finite(size, cell_size):
    """Property 7: Flat terrain must produce finite TWI."""
    dem = np.zeros((size, size), dtype=np.float32)
    twi = compute_twi(dem, cell_size_m=cell_size)
    assert np.all(np.isfinite(twi))


@given(size=st.integers(3, 8), cell_size=st.floats(10.0, 500.0), elevation_scale=st.floats(1.0, 100.0))
@settings(max_examples=20, deadline=None)
def test_twi_output_is_always_finite(size, cell_size, elevation_scale):
    """Property 7: For any valid DEM, TWI output must be finite."""
    rng = np.random.default_rng(42)
    dem = (rng.random((size, size)) * elevation_scale).astype(np.float32)
    twi = compute_twi(dem, cell_size_m=cell_size)
    assert np.all(np.isfinite(twi))
