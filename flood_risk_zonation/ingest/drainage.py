"""
Synthetic drainage capacity generation for the Flood Risk Zonation System.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np

from flood_risk_zonation.models import DrainageDataset

logger = logging.getLogger(__name__)


def generate_synthetic_drainage(
    grid: gpd.GeoDataFrame,
    seed: int = 42,
) -> DrainageDataset:
    """
    Assign synthetic drainage capacity scores [0, 1] to grid cells.

    Scores are inversely correlated with population density (if available
    in the grid) to simulate urban impervious surface effects. Higher
    population density → lower drainage capacity.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid GeoDataFrame. If it contains a 'population_density' column,
        drainage is inversely correlated with it.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DrainageDataset
        Per-cell drainage capacity scores in [0, 1].
    """
    n = len(grid)
    rng = np.random.default_rng(seed)

    if "population_density" in grid.columns:
        pop = grid["population_density"].fillna(0).values.astype(np.float64)
        # Normalise population to [0, 1]
        pop_max = pop.max()
        if pop_max > 0:
            pop_norm = pop / pop_max
        else:
            pop_norm = np.zeros(n)
        # Drainage inversely correlated with population + small random noise
        noise = rng.uniform(0, 0.1, n)
        scores = np.clip(1.0 - pop_norm * 0.8 + noise, 0.0, 1.0).astype(np.float32)
    else:
        # No population data: uniform random scores
        scores = rng.uniform(0.2, 1.0, n).astype(np.float32)

    cell_ids = list(grid["cell_id"].astype(str))

    return DrainageDataset(
        capacity_scores=scores,
        cell_ids=cell_ids,
    )
