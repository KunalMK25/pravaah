"""
Folium map construction for PRAVAAH.
"""
from __future__ import annotations

import logging

import folium
import geopandas as gpd
import numpy as np

from flood_risk_zonation.visualization.explainability import build_cell_explanation
from flood_risk_zonation.visualization.layers import (
    RISK_COLOR_MAP,
    add_drainage_lines_layer,
    add_habitation_layer,
    add_population_density_layer,
    add_rainfall_heatmap_layer,
    add_red_zone_layer,
    add_risk_choropleth_layer,
    add_spatial_zone_layer,
    add_relocation_candidate_layer,
    add_water_bodies_layer,
)

logger = logging.getLogger(__name__)

# Maximum number of cells that receive interactive explainability
# (tooltip + popup). Beyond this, performance degrades noticeably.
_MAX_EXPLAINABLE_CELLS = 100


class FloodRiskMapBuilder:
    """Builds interactive Folium maps from scored grid GeoDataFrames."""

    RISK_COLOR_MAP = RISK_COLOR_MAP

    def build_choropleth_map(
        self,
        scored_grid: gpd.GeoDataFrame,
        center: tuple[float, float],
        zoom_start: int = 12,
        use_google_maps: bool = False,
        google_maps_api_key: str | None = None,
        water_bodies: gpd.GeoDataFrame | None = None,
        model_bounds: dict | None = None,
        # SIH / Phase 2 arguments
        exposure_results: list | None = None,
        relocation_results: list | None = None,
        show_red_zones: bool = True,
        # Phase 3 — spatial zones + candidates
        zoned_grid: gpd.GeoDataFrame | None = None,
        relocation_candidates: dict | None = None,
        show_spatial_zones: bool = True,
        zone_filter: list[str] | None = None,
        # Emergency response — facilities + evacuation routes
        show_emergency_facilities: bool = False,
        hospitals: list | None = None,
        shelters: list | None = None,
        evacuation_routes: list | None = None,
    ) -> folium.Map:
        """
        Construct a Folium map with risk choropleth, drainage lines,
        rainfall heatmap, population, and water body layers.

        Parameters
        ----------
        scored_grid : gpd.GeoDataFrame
            Scored grid with risk_class, risk_score, feature columns, and
            optional is_coastal_tsunami_risk column.
        center : tuple[float, float]
            (lat, lon) map center.
        zoom_start : int
            Initial zoom level.
        model_bounds : dict | None
            Fitted normalisation bounds from WeightedSusceptibilityModel
            ({feature: (lower, upper)}). Passed to per-cell explanation for
            accurate bar scaling. If None, falls back to fixed reference ranges.
        exposure_results : list[ExposureResult] | None
            SIH exposure results for habitation marker overlay.
        relocation_results : list[RelocationPriorityResult] | None
            SIH relocation priorities for marker colouring.
        show_red_zones : bool
            If True, render a dedicated Red Zone overlay layer.
        """
        m = folium.Map(location=list(center), zoom_start=zoom_start, tiles="OpenStreetMap")

        # Layer 1: Risk classification choropleth (color fill, no interactivity)
        add_risk_choropleth_layer(m, scored_grid)

        # Layer 2: Spatial zones (RED/YELLOW/GREEN) — Phase 3
        if show_spatial_zones and zoned_grid is not None:
            add_spatial_zone_layer(m, zoned_grid, zone_filter=zone_filter)
        elif show_red_zones:
            # Fall back to legacy red-zone layer when zoned_grid not yet computed
            add_red_zone_layer(m, scored_grid)

        # Layer 3: Drainage lines
        add_drainage_lines_layer(m)

        # Layer 4: Rainfall heatmap (toggle)
        add_rainfall_heatmap_layer(m, scored_grid)

        # Layer 5: Population density (toggle)
        add_population_density_layer(m, scored_grid)

        # Layer 6: Water bodies (toggle)
        if water_bodies is not None:
            add_water_bodies_layer(m, water_bodies)

        # Layer 7: Habitation markers (Phase 2)
        if exposure_results:
            add_habitation_layer(m, exposure_results, relocation_results)

        # Layer 8: Relocation candidate areas (Phase 3)
        if relocation_candidates and exposure_results:
            add_relocation_candidate_layer(m, relocation_candidates, exposure_results)

        # Layer 9: Emergency facilities (hospitals, shelters)
        if show_emergency_facilities:
            from flood_risk_zonation.visualization.layers import add_emergency_facilities_layer
            add_emergency_facilities_layer(m, hospitals=hospitals, shelters=shelters)

        # Layer 10: Evacuation routes (hazard-aware)
        if evacuation_routes:
            from flood_risk_zonation.visualization.layers import add_evacuation_routes_layer
            add_evacuation_routes_layer(m, evacuation_routes=evacuation_routes)

        # Per-cell hover tooltips + click popups
        self.add_cell_explainability_layer(m, scored_grid, model_bounds=model_bounds)

        # Layer control added last so it indexes all layers
        folium.LayerControl(collapsed=False).add_to(m)

        return m

    def add_cell_explainability_layer(
        self,
        folium_map: folium.Map,
        grid: gpd.GeoDataFrame,
        model_bounds: dict | None = None,
    ) -> folium.Map:
        """
        Add hover tooltips (quick summary) and click popups (full breakdown)
        to each grid cell.

        - All risk classes get a tooltip (visible on hover).
        - All risk classes get a popup (visible on click).
        - Water cells show a water-specific explanation (no factor bars).
        - Coastal cells show the ⚠️ Tsunami Risk badge.
        - Capped at _MAX_EXPLAINABLE_CELLS total (prioritises High-risk cells,
          then others) to keep page load times acceptable.

        Parameters
        ----------
        model_bounds : dict | None
            {feature_name: (lower_5th_pct, upper_95th_pct)} from the fitted
            WeightedSusceptibilityModel. Used to scale factor bars relative
            to the actual data distribution, not fixed ranges.
        """
        fg = folium.FeatureGroup(name="Cell Info (hover/click)", show=True)

        # Prioritise cells for display: High first, then Water, Medium, Low
        priority_order = {"High": 0, "Water": 1, "Medium": 2, "Low": 3}
        grid_sorted = grid.copy()
        grid_sorted["_priority"] = grid_sorted["risk_class"].map(
            lambda c: priority_order.get(str(c), 9)
        )
        grid_sorted = grid_sorted.sort_values("_priority").head(_MAX_EXPLAINABLE_CELLS)

        for _, row in grid_sorted.iterrows():
            try:
                tooltip_html, popup_html = build_cell_explanation(row, model_bounds)
            except Exception as e:
                logger.debug("Skipping cell explanation: %s", e)
                continue

            # Use folium.Polygon (not GeoJson) for per-cell interactivity.
            # GeoJson bulk-renders multiple features as one Leaflet layer and
            # can miss individual mouse events. Polygon creates one L.polygon
            # per cell, guaranteeing tooltip/popup fire correctly on hover/click.
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Extract exterior ring coords as (lat, lon) pairs for folium.Polygon
            try:
                if geom.geom_type == "Polygon":
                    coords = [(lat, lon) for lon, lat in geom.exterior.coords]
                elif geom.geom_type == "MultiPolygon":
                    # Use largest polygon
                    largest = max(geom.geoms, key=lambda g: g.area)
                    coords = [(lat, lon) for lon, lat in largest.exterior.coords]
                else:
                    continue
            except Exception:
                continue

            folium.Polygon(
                locations=coords,
                color="transparent",
                fill=True,
                fill_color="transparent",
                fill_opacity=0.0,
                weight=0,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(fg)

        fg.add_to(folium_map)
        return folium_map

    # ── Backward-compat alias ─────────────────────────────────────────────────
    def add_popup_layer(self, folium_map: folium.Map, grid: gpd.GeoDataFrame) -> folium.Map:
        """Deprecated alias — calls add_cell_explainability_layer."""
        return self.add_cell_explainability_layer(folium_map, grid)
