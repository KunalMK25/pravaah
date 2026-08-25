"""
Terrain feature computation for the Flood Risk Zonation System.

Implements slope, TWI, aspect, and curvature from a DEM array.
All functions handle flat terrain (all-zero DEM) without producing NaN or infinity.
"""
from __future__ import annotations

import numpy as np


def compute_slope(dem_array: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Compute slope in degrees from a DEM array using the Horn (1981) method.

    Uses numpy gradient for dz/dx and dz/dy, then arctan of the magnitude.
    Flat terrain returns 0 degrees.

    Parameters
    ----------
    dem_array : np.ndarray
        2D float array of elevation values in metres.
    cell_size_m : float
        Cell size in metres (used to scale gradients).

    Returns
    -------
    np.ndarray
        2D array of slope values in degrees [0, 90].
    """
    if dem_array.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    dz_dy, dz_dx = np.gradient(dem_array.astype(np.float64), cell_size_m)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)
    return np.clip(slope_deg, 0.0, 90.0).astype(np.float32)


def compute_twi(dem_array: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Compute Topographic Wetness Index: TWI = ln(A / (tan(β) + ε))

    Where:
    - A is the upslope contributing area per unit contour length (m²/m),
      approximated via D8 flow accumulation on the DEM.
    - β is the local slope in radians.
    - ε = 1e-6 prevents division by zero on flat terrain.

    Parameters
    ----------
    dem_array : np.ndarray
        2D float array of elevation values in metres.
    cell_size_m : float
        Cell size in metres.

    Returns
    -------
    np.ndarray
        2D array of TWI values (finite, no NaN or infinity).
    """
    if dem_array.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    dem = dem_array.astype(np.float64)
    nrows, ncols = dem.shape

    # --- Slope in radians ---
    dz_dy, dz_dx = np.gradient(dem, cell_size_m)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    # --- D8 flow accumulation (simplified) ---
    # Each cell contributes 1 unit of area; accumulate downslope
    # For a simple approximation, use a uniform contributing area
    # proportional to cell_size_m² (1 cell = 1 unit)
    flow_acc = np.ones((nrows, ncols), dtype=np.float64)

    # Simple D8: for each cell, find the steepest downslope neighbour
    # and accumulate flow. We use a simplified version that iterates
    # from high to low elevation.
    flat_indices = np.argsort(dem.ravel())[::-1]  # high to low

    for flat_idx in flat_indices:
        row = flat_idx // ncols
        col = flat_idx % ncols
        current_elev = dem[row, col]

        # Find steepest downslope neighbour
        best_drop = 0.0
        best_r, best_c = -1, -1

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < nrows and 0 <= nc < ncols:
                    drop = current_elev - dem[nr, nc]
                    dist = cell_size_m * (1.414 if dr != 0 and dc != 0 else 1.0)
                    gradient = drop / dist
                    if gradient > best_drop:
                        best_drop = gradient
                        best_r, best_c = nr, nc

        if best_r >= 0:
            flow_acc[best_r, best_c] += flow_acc[row, col]

    # Upslope area per unit contour length (m²/m)
    upslope_area = flow_acc * cell_size_m

    # TWI = ln(A / (tan(β) + ε))
    epsilon = 1e-6
    tan_slope = np.tan(slope_rad) + epsilon
    twi = np.log(upslope_area / tan_slope)

    # Ensure finite values
    twi = np.where(np.isfinite(twi), twi, 0.0)
    return twi.astype(np.float32)


def compute_aspect(dem_array: np.ndarray) -> np.ndarray:
    """
    Compute terrain aspect in degrees (0–360, clockwise from north).

    Flat terrain returns 0 degrees.

    Parameters
    ----------
    dem_array : np.ndarray
        2D float array of elevation values.

    Returns
    -------
    np.ndarray
        2D array of aspect values in degrees [0, 360).
    """
    if dem_array.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    dem = dem_array.astype(np.float64)
    dz_dy, dz_dx = np.gradient(dem)

    # Aspect: angle from north, clockwise
    # North is -dz_dy direction (increasing row = south)
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    # Convert to 0–360
    aspect_deg = aspect_deg % 360.0
    return aspect_deg.astype(np.float32)


def compute_curvature(dem_array: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Compute plan curvature from a DEM array.

    Plan curvature is the curvature of the terrain surface in the horizontal
    plane. Positive values indicate convex terrain; negative values indicate
    concave terrain.

    Parameters
    ----------
    dem_array : np.ndarray
        2D float array of elevation values in metres.
    cell_size_m : float
        Cell size in metres.

    Returns
    -------
    np.ndarray
        2D array of plan curvature values.
    """
    if dem_array.size == 0:
        return np.zeros_like(dem_array, dtype=np.float32)

    dem = dem_array.astype(np.float64)
    # Second derivatives
    dz_dy, dz_dx = np.gradient(dem, cell_size_m)
    d2z_dx2 = np.gradient(dz_dx, cell_size_m, axis=1)
    d2z_dy2 = np.gradient(dz_dy, cell_size_m, axis=0)

    # Plan curvature = -(d²z/dx² + d²z/dy²)
    curvature = -(d2z_dx2 + d2z_dy2)
    curvature = np.where(np.isfinite(curvature), curvature, 0.0)
    return curvature.astype(np.float32)
