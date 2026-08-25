"""
Export utilities for the Flood Risk Zonation System.

Supports HTML (Folium map), GeoJSON, CSV, and PDF report export.
"""
from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd

from flood_risk_zonation.features.extractor import FEATURE_COLUMNS


def export_html(folium_map: folium.Map, output_path: Path | str) -> None:
    """Save a Folium map as a self-contained HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folium_map.save(str(output_path))


def export_geojson(scored_grid: gpd.GeoDataFrame, output_path: Path | str) -> None:
    """Save the scored GeoDataFrame as a GeoJSON FeatureCollection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_grid.to_file(str(output_path), driver="GeoJSON")


def export_csv(scored_grid: gpd.GeoDataFrame, output_path: Path | str) -> None:
    """Save feature columns and risk outputs as CSV (no geometry column)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cell_id", "centroid_lat", "centroid_lon"] + FEATURE_COLUMNS + ["risk_score", "risk_class"]
    available = [c for c in cols if c in scored_grid.columns]
    scored_grid[available].to_csv(str(output_path), index=False)
