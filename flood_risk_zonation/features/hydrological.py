"""
Hydrological feature computation for the Flood Risk Zonation System.

Computes distance to nearest water body and drainage density per grid cell.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from flood_risk_zonation.models import DrainageDataset

# Maximum distance cap in metres
MAX_DISTANCE_M = 10_000.0


def compute_distance_to_water(
    grid: gpd.GeoDataFrame,
    water_bodies: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Compute the minimum distance from each grid cell centroid to the nearest
    water body polygon boundary, using GeoPandas STRtree spatial index.

    Distances are capped at MAX_DISTANCE_M (10,000m) and are always non-negative.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid GeoDataFrame with centroid_lat and centroid_lon columns.
    water_bodies : gpd.GeoDataFrame
        Water body polygons. If empty, returns MAX_DISTANCE_M for all cells.

    Returns
    -------
    np.ndarray
        1D float array of distances in metres, one per grid cell.
    """
    n = len(grid)

    if water_bodies is None or len(water_bodies) == 0:
        return np.full(n, MAX_DISTANCE_M, dtype=np.float64)

    # Build centroid points
    centroids = gpd.GeoDataFrame(
        geometry=[
            Point(row["centroid_lon"], row["centroid_lat"])
            for _, row in grid.iterrows()
        ],
        crs="EPSG:4326",
    )

    # Reproject to a metric CRS for accurate distance calculation
    # Use a simple equirectangular approximation via EPSG:3857 (Web Mercator)
    try:
        centroids_m = centroids.to_crs("EPSG:3857")
        if water_bodies.crs is None:
            wb_m = water_bodies.set_crs("EPSG:4326").to_crs("EPSG:3857")
        else:
            wb_m = water_bodies.to_crs("EPSG:3857")
    except Exception:
        # Fallback: use degree-based distances scaled to metres
        centroids_m = centroids
        wb_m = water_bodies

    # Build STRtree for efficient nearest-neighbour queries
    tree = wb_m.sindex

    distances = np.zeros(n, dtype=np.float64)

    for i, centroid_geom in enumerate(centroids_m.geometry):
        # Find nearest water body using STRtree
        nearest_idx = tree.nearest(centroid_geom)
        # nearest() may return None, a scalar, or a 0-d/1-d numpy array
        if nearest_idx is None:
            distances[i] = MAX_DISTANCE_M
            continue
        nearest_array = np.asarray(nearest_idx)
        # GeoPandas returns a 2 x N array: query indexes in row 0 and tree
        # indexes in row 1. A scalar/1-D result is retained for compatibility
        # with older spatial-index implementations.
        if nearest_array.ndim == 2 and nearest_array.shape[0] == 2:
            idx = int(nearest_array[1, 0])
        else:
            idx = int(nearest_array.flat[-1])
        nearest_geom = wb_m.geometry.iloc[idx]
        dist = centroid_geom.distance(nearest_geom)
        distances[i] = min(dist, MAX_DISTANCE_M)

    return np.clip(distances, 0.0, MAX_DISTANCE_M)


def compute_drainage_density(
    grid: gpd.GeoDataFrame,
    drainage_data: DrainageDataset,
) -> np.ndarray:
    """
    Assign drainage capacity scores from a DrainageDataset to each grid cell.

    Matches cells by cell_id. Cells not found in drainage_data receive a
    default score of 0.5.

    Parameters
    ----------
    grid : gpd.GeoDataFrame
        Grid GeoDataFrame with cell_id column.
    drainage_data : DrainageDataset
        Per-cell drainage capacity scores.

    Returns
    -------
    np.ndarray
        1D float array of drainage capacity scores in [0, 1].
    """
    # Build lookup dict from cell_id to score
    score_map = dict(zip(drainage_data.cell_ids, drainage_data.capacity_scores))

    scores = np.array(
        [score_map.get(str(cid), 0.5) for cid in grid["cell_id"]],
        dtype=np.float32,
    )
    return np.clip(scores, 0.0, 1.0)
