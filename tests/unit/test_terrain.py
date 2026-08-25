"""
Unit tests for terrain feature computation.
"""
import math
import numpy as np
import pytest
from flood_risk_zonation.features.terrain import (
    compute_slope, compute_twi, compute_aspect, compute_curvature
)


def test_compute_twi_flat_dem_produces_finite_values():
    """compute_twi on a flat DEM (all zeros) must produce all-finite values."""
    dem = np.zeros((10, 10), dtype=np.float32)
    twi = compute_twi(dem, cell_size_m=30.0)
    assert np.all(np.isfinite(twi)), f"TWI contains non-finite values: {twi[~np.isfinite(twi)]}"


def test_compute_slope_inclined_plane():
    """compute_slope on a known inclined plane should produce the expected angle."""
    # Create a DEM that rises 1m per 1m horizontally → 45° slope
    cell_size = 1.0
    size = 10
    dem = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        dem[:, i] = float(i)  # rises 1m per cell in x direction

    slope = compute_slope(dem, cell_size_m=cell_size)
    # Interior cells should be close to 45°
    interior = slope[1:-1, 1:-1]
    assert np.allclose(interior, 45.0, atol=1.0), \
        f"Expected ~45° slope, got mean={interior.mean():.2f}°"


def test_compute_aspect_north_slope():
    """compute_aspect on a DEM sloping due north (decreasing lat = south) returns ~0°."""
    # DEM that decreases from top to bottom (north to south) → aspect = 0° (north)
    size = 10
    dem = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        dem[i, :] = float(size - i)  # decreases going south (increasing row index)

    aspect = compute_aspect(dem)
    # Interior cells should be close to 0° (north-facing)
    interior = aspect[1:-1, 1:-1]
    # Allow for 0° or 360° (both represent north)
    north_facing = np.logical_or(interior < 10.0, interior > 350.0)
    assert np.all(north_facing), \
        f"Expected north-facing aspect (~0°), got values: {interior}"


def test_compute_slope_flat_dem_returns_zero():
    """compute_slope on a flat DEM should return all zeros."""
    dem = np.zeros((5, 5), dtype=np.float32)
    slope = compute_slope(dem, cell_size_m=30.0)
    assert np.allclose(slope, 0.0), f"Expected all-zero slope, got {slope}"


def test_compute_curvature_flat_dem_returns_zero():
    """compute_curvature on a flat DEM should return all zeros."""
    dem = np.zeros((5, 5), dtype=np.float32)
    curv = compute_curvature(dem, cell_size_m=30.0)
    assert np.allclose(curv, 0.0, atol=1e-6), f"Expected all-zero curvature, got {curv}"
