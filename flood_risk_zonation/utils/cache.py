"""
GeoParquet caching utilities for the Flood Risk Zonation System.

Functions
---------
save_geodataframe   — persist a GeoDataFrame to GeoParquet.
load_geodataframe   — load a GeoDataFrame from GeoParquet.
cache_key           — produce a deterministic string key from BoundingBox + PipelineConfig.
is_cached           — check whether a cache entry exists.
get_cache_path      — return the full path for a cache entry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd


def save_geodataframe(gdf: gpd.GeoDataFrame, path: Path | str) -> None:
    """
    Persist a GeoDataFrame to GeoParquet format.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to save.
    path : Path | str
        Destination file path (should end in .parquet or .geoparquet).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path)


def load_geodataframe(path: Path | str) -> gpd.GeoDataFrame:
    """
    Load a GeoDataFrame from a GeoParquet file.

    Parameters
    ----------
    path : Path | str
        Source file path.

    Returns
    -------
    gpd.GeoDataFrame
        The loaded GeoDataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    return gpd.read_parquet(path)


def cache_key(bounding_box: object, config: object) -> str:
    """
    Produce a deterministic SHA-256 cache key from a BoundingBox and PipelineConfig.

    The key is derived from a JSON-serialised dict of the relevant fields,
    ensuring the same inputs always produce the same key.

    Parameters
    ----------
    bounding_box : BoundingBox
        Geographic extent.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    str
        A 16-character hex string (first 16 chars of SHA-256 digest).
    """
    key_dict = {
        "min_lon": bounding_box.min_lon,
        "min_lat": bounding_box.min_lat,
        "max_lon": bounding_box.max_lon,
        "max_lat": bounding_box.max_lat,
        "cell_size_meters": config.cell_size_meters,
        "model_type": config.model_type,
        "random_seed": config.random_seed,
    }
    serialised = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def is_cached(key: str, cache_dir: Path | str) -> bool:
    """
    Check whether a cache entry exists for the given key.

    Parameters
    ----------
    key : str
        Cache key produced by :func:`cache_key`.
    cache_dir : Path | str
        Directory where cache files are stored.

    Returns
    -------
    bool
        True if the cache file exists, False otherwise.
    """
    return get_cache_path(key, cache_dir).exists()


def get_cache_path(key: str, cache_dir: Path | str) -> Path:
    """
    Return the full file path for a cache entry.

    Parameters
    ----------
    key : str
        Cache key produced by :func:`cache_key`.
    cache_dir : Path | str
        Directory where cache files are stored.

    Returns
    -------
    Path
        Full path to the cache file (e.g. data/cache/<key>.parquet).
    """
    return Path(cache_dir) / f"{key}.parquet"
