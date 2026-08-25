"""
Individual layer builder functions for the PRAVAAH interactive map.
Uses bulk GeoJson rendering for performance with large grids.
"""
from __future__ import annotations
import json
from pathlib import Path
import folium
import folium.plugins
import geopandas as gpd
import numpy as np

RISK_COLOR_MAP = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
    "Water": "#3498db",
}


def add_risk_choropleth_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Render risk zones using bulk GeoJson per class — fast for any grid size.

    Rendered as non-interactive (no tooltip/popup) so that mouse events
    pass through to the per-cell explainability layer added on top.
    """
    fg = folium.FeatureGroup(name="Risk Classification", show=True)
    for risk_class, color in RISK_COLOR_MAP.items():
        subset = scored_grid[scored_grid["risk_class"] == risk_class]
        if len(subset) == 0:
            continue
        folium.GeoJson(
            subset.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 0.3,
                "fillOpacity": 0.6,
            },
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_rainfall_heatmap_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add a rainfall intensity heatmap layer using cell centroids."""
    if "rainfall_mean_mm" not in scored_grid.columns:
        return folium_map
    rain_vals = scored_grid["rainfall_mean_mm"].values.astype(float)
    rain_max = float(rain_vals.max()) if rain_vals.max() > 0 else 1.0
    heat_data = [
        [row["centroid_lat"], row["centroid_lon"], float(row.get("rainfall_mean_mm", 0)) / rain_max]
        for _, row in scored_grid.iterrows()
    ]
    fg = folium.FeatureGroup(name="Rainfall Intensity", show=False)
    folium.plugins.HeatMap(
        heat_data, min_opacity=0.3, max_zoom=18, radius=25, blur=20,
        gradient={0.2: "#ffffb2", 0.5: "#fd8d3c", 0.8: "#e31a1c", 1.0: "#800026"},
    ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_drainage_lines_layer(folium_map: folium.Map, drainage_path: str | None = None) -> folium.Map:
    """Add OSM drainage lines as a layer. Searches data/drainage_lines/ for any geojson."""
    import logging
    logger = logging.getLogger(__name__)

    # Find any drainage geojson file
    drain_dir = Path("data/drainage_lines")
    if drainage_path is None:
        files = list(drain_dir.glob("*.geojson")) if drain_dir.exists() else []
        if not files:
            return folium_map
        drainage_path = str(files[0])

    path = Path(drainage_path)
    if not path.exists():
        return folium_map

    try:
        with open(path) as f:
            geojson_data = json.load(f)
        if not geojson_data.get("features"):
            return folium_map
        type_colors = {
            "drain": "#1a6faf", "canal": "#2980b9", "stream": "#5dade2",
            "river": "#1b4f72", "ditch": "#7fb3d3",
        }
        fg = folium.FeatureGroup(name="Drainage Lines", show=True)
        for feature in geojson_data["features"]:
            wtype = feature.get("properties", {}).get("waterway", "drain")
            color = type_colors.get(wtype, "#2980b9")
            weight = 3 if wtype in ("river", "canal") else 2
            folium.GeoJson(
                feature,
                style_function=lambda _, c=color, w=weight: {"color": c, "weight": w, "opacity": 0.85},
            ).add_to(fg)
        fg.add_to(folium_map)
    except Exception as e:
        logger.warning("Failed to load drainage lines: %s", e)
    return folium_map


def add_population_density_layer(folium_map: folium.Map, scored_grid: gpd.GeoDataFrame) -> folium.Map:
    """Add population density overlay."""
    if "population_density" not in scored_grid.columns:
        return folium_map
    fg = folium.FeatureGroup(name="Population Density", show=False)
    pop_max = float(scored_grid["population_density"].max()) or 1.0
    for _, row in scored_grid.iterrows():
        intensity = float(row.get("population_density", 0)) / pop_max
        r = int(255 * intensity)
        color = f"#{r:02x}4444"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {"fillColor": c, "color": "none", "fillOpacity": 0.4},
        ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


def add_water_bodies_layer(folium_map: folium.Map, water_bodies: gpd.GeoDataFrame) -> folium.Map:
    """Add water bodies overlay."""
    if water_bodies is None or len(water_bodies) == 0:
        return folium_map
    fg = folium.FeatureGroup(name="Water Bodies", show=False)
    folium.GeoJson(
        water_bodies.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#3498db", "color": "#2980b9", "weight": 1, "fillOpacity": 0.5,
        },
    ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map


# ── PRAVAAH habitation & relocation visualization layers ─────────────────────

PRIORITY_COLOR_MAP = {
    "CRITICAL": "#c0392b",   # dark red
    "HIGH":     "#e67e22",   # orange
    "MEDIUM":   "#f1c40f",   # yellow
    "LOW":      "#2ecc71",   # green
}

PRIORITY_ICON_MAP = {
    "CRITICAL": "exclamation-triangle",
    "HIGH":     "exclamation-circle",
    "MEDIUM":   "info-circle",
    "LOW":      "check-circle",
}


def add_habitation_layer(
    folium_map: folium.Map,
    exposure_results: list,
    relocation_results: list | None = None,
) -> folium.Map:
    """
    Add habitation markers to the map, colour-coded by relocation priority.

    Parameters
    ----------
    folium_map : folium.Map
    exposure_results : list[ExposureResult]
        Per-habitation exposure data.
    relocation_results : list[RelocationPriorityResult] | None
        If provided, markers are coloured by priority class.
        If None, markers are coloured by hazard class.

    Returns
    -------
    folium.Map
    """
    if not exposure_results:
        return folium_map

    # Build lookup maps
    rel_map: dict = {}
    if relocation_results:
        rel_map = {r.hab_id: r for r in relocation_results}

    hazard_color = {
        "High": "#e74c3c",
        "Medium": "#f39c12",
        "Low": "#2ecc71",
        "Water": "#3498db",
    }

    fg = folium.FeatureGroup(name="Habitations", show=True)

    for exp in exposure_results:
        rel = rel_map.get(exp.hab_id)

        if rel:
            color = PRIORITY_COLOR_MAP.get(rel.priority_class, "#999999")
            icon_name = PRIORITY_ICON_MAP.get(rel.priority_class, "home")
            priority_label = rel.priority_class
            score_label = f"{rel.relocation_score:.2f}"
            action_short = rel.recommended_action[:80] + "…" if len(rel.recommended_action) > 80 else rel.recommended_action
        else:
            color = hazard_color.get(exp.hazard_class, "#999999")
            icon_name = "home"
            priority_label = "—"
            score_label = "—"
            action_short = "Run PRAVAAH analysis for full recommendation."

        pop_str = (
            f"{exp.population_exposed:,} (OSM)" if exp.population_source == "osm_tag" and exp.population_exposed
            else "Unknown"
        )
        hab_name = exp.name or f"Unnamed ({exp.hab_type})"

        tooltip_html = (
            f"<b>{hab_name}</b><br>"
            f"Type: {exp.hab_type}<br>"
            f"Hazard: {exp.hazard_class} ({exp.hazard_score:.1f})<br>"
            f"Priority: <b style='color:{color}'>{priority_label}</b>"
        )

        popup_html = f"""
        <div style="font-family:sans-serif;font-size:12px;min-width:220px;max-width:280px">
          <h4 style="margin:0 0 6px 0;color:#1a252f">{hab_name}</h4>
          <table style="width:100%;border-collapse:collapse">
            <tr><td style="color:#666;padding:2px 4px">Type</td><td style="padding:2px 4px">{exp.hab_type}</td></tr>
            <tr style="background:#f8f8f8"><td style="color:#666;padding:2px 4px">Hazard Score</td><td style="padding:2px 4px">{exp.hazard_score:.1f} / 100</td></tr>
            <tr><td style="color:#666;padding:2px 4px">Hazard Class</td><td style="padding:2px 4px"><b style="color:{hazard_color.get(exp.hazard_class,'#333')}">{exp.hazard_class}</b></td></tr>
            <tr style="background:#f8f8f8"><td style="color:#666;padding:2px 4px">Population</td><td style="padding:2px 4px">{pop_str}</td></tr>
            <tr><td style="color:#666;padding:2px 4px">Relocation Priority</td><td style="padding:2px 4px"><b style="color:{color}">{priority_label}</b> ({score_label})</td></tr>
          </table>
          <p style="margin:6px 0 0 0;font-size:11px;color:#555;border-top:1px solid #eee;padding-top:4px">{action_short}</p>
        </div>
        """

        try:
            folium.Marker(
                location=[exp.lat, exp.lon],
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color="red" if priority_label == "CRITICAL" else
                                        "orange" if priority_label == "HIGH" else
                                        "blue" if priority_label == "LOW" else "gray",
                                 icon="home", prefix="fa"),
            ).add_to(fg)
        except Exception:
            # Fallback: simple circle marker
            folium.CircleMarker(
                location=[exp.lat, exp.lon],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fg)

    fg.add_to(folium_map)
    return folium_map


def add_red_zone_layer(
    folium_map: folium.Map,
    scored_grid: gpd.GeoDataFrame,
) -> folium.Map:
    """
    Highlight High-risk cells as explicit 'Red Zone' cells with
    a distinct pulsing outline style (thicker border, stronger fill).
    """
    red_cells = scored_grid[scored_grid["risk_class"] == "High"]
    if len(red_cells) == 0:
        return folium_map

    fg = folium.FeatureGroup(name="🔴 Red Zones", show=True)
    folium.GeoJson(
        red_cells.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#c0392b",
            "color": "#922b21",
            "weight": 1.5,
            "fillOpacity": 0.55,
        },
    ).add_to(fg)
    fg.add_to(folium_map)
    return folium_map
