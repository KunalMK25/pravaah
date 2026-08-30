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


# ── PRAVAAH Phase 3 — Spatial Zone Layers ─────────────────────────────────────

# Operational zone colours (distinct from the ML risk class colours)
ZONE_FILL_COLORS = {
    "RED":    "#c0392b",
    "YELLOW": "#f39c12",
    "GREEN":  "#27ae60",
    "WATER":  "#2980b9",
}
ZONE_BORDER_COLORS = {
    "RED":    "#922b21",
    "YELLOW": "#b7770d",
    "GREEN":  "#1e8449",
    "WATER":  "#1a5276",
}


def add_spatial_zone_layer(
    folium_map: folium.Map,
    zoned_grid: gpd.GeoDataFrame,
    zone_filter: list[str] | None = None,
) -> folium.Map:
    """
    Add the RED / YELLOW / GREEN / WATER spatial attention zone layer.

    This layer is SEPARATE from the underlying ML risk classification.
    It represents the operational attention zones:
      🟥 RED    — primary hazard zone (based on ML High class)
      🟨 YELLOW — secondary attention zone (adjacent to RED or Medium class)
      🟩 GREEN  — lower-risk area / potential safe zone
      🔵 WATER  — permanent water body

    The layer name includes the zone type so users can toggle each
    separately via the Folium layer control.

    Parameters
    ----------
    zoned_grid : gpd.GeoDataFrame
        Grid with a ``spatial_zone`` column (output of classify_spatial_zones).
    zone_filter : list[str] | None
        If specified, only render these zones (e.g. ["RED", "YELLOW"]).
        Default: render all four zones.
    """
    if "spatial_zone" not in zoned_grid.columns:
        return folium_map

    zones_to_show = zone_filter or ["RED", "YELLOW", "GREEN", "WATER"]
    zone_labels = {
        "RED":    "🟥 RED — Primary Hazard Zone",
        "YELLOW": "🟨 YELLOW — Secondary Attention Zone",
        "GREEN":  "🟩 GREEN — Lower-Risk / Potential Safe Area",
        "WATER":  "🔵 WATER — Permanent Water Body",
    }

    for zone in zones_to_show:
        subset = zoned_grid[zoned_grid["spatial_zone"] == zone]
        if len(subset) == 0:
            continue
        fill   = ZONE_FILL_COLORS.get(zone, "#aaaaaa")
        border = ZONE_BORDER_COLORS.get(zone, "#666666")
        label  = zone_labels.get(zone, zone)

        # GREEN zone shown by default off (it can be large and noisy)
        show_default = zone in ("RED", "YELLOW")

        fg = folium.FeatureGroup(name=label, show=show_default)
        folium.GeoJson(
            subset.__geo_interface__,
            style_function=lambda _, f=fill, b=border: {
                "fillColor":   f,
                "color":       b,
                "weight":      0.8,
                "fillOpacity": 0.45 if zone == "GREEN" else 0.55,
            },
        ).add_to(fg)
        fg.add_to(folium_map)

    return folium_map


def add_relocation_candidate_layer(
    folium_map: folium.Map,
    relocation_candidates: dict,
    source_exposures: list,
) -> folium.Map:
    """
    Add relocation candidate area markers to the map.

    Candidates are shown as green circle markers with a letter label.
    Clicking a candidate shows its key metrics and candidate score.

    Parameters
    ----------
    relocation_candidates : dict
        hab_id → list[RelocationCandidate]
    source_exposures : list[ExposureResult]
        Used to show the source habitation name in the candidate popup.
    """
    if not relocation_candidates:
        return folium_map

    exp_map = {e.hab_id: e for e in source_exposures}
    fg = folium.FeatureGroup(name="🟩 Relocation Candidate Areas", show=False)

    rank_labels = ["A", "B", "C", "D", "E"]

    for hab_id, candidates in relocation_candidates.items():
        exp = exp_map.get(hab_id)
        source_name = exp.name if exp else hab_id

        for rank, cand in enumerate(candidates[:5]):
            label = rank_labels[rank] if rank < len(rank_labels) else str(rank + 1)
            score_pct = int(cand.candidate_score * 100)

            tooltip_html = (
                f"<b>Candidate {label}</b> for <i>{source_name}</i><br>"
                f"Score: {score_pct}% | "
                f"Distance: {cand.distance_km:.1f} km | "
                f"Area: {cand.area_km2:.2f} km²"
            )
            popup_html = f"""
            <div style="font-family:sans-serif;font-size:12px;min-width:200px;max-width:260px">
              <h4 style="margin:0 0 6px 0;color:#1e8449">
                Relocation Candidate {label}
              </h4>
              <p style="margin:0 0 4px 0;font-size:11px;color:#666">
                For: <i>{source_name}</i>
              </p>
              <table style="width:100%;border-collapse:collapse">
                <tr><td style="color:#666;padding:2px 4px">Candidate Score</td>
                    <td style="padding:2px 4px"><b>{score_pct}%</b></td></tr>
                <tr style="background:#f0f9f0">
                  <td style="color:#666;padding:2px 4px">Distance</td>
                  <td style="padding:2px 4px">{cand.distance_km:.1f} km</td></tr>
                <tr><td style="color:#666;padding:2px 4px">Safe Area</td>
                    <td style="padding:2px 4px">{cand.area_km2:.2f} km²</td></tr>
                <tr style="background:#f0f9f0">
                  <td style="color:#666;padding:2px 4px">Hazard</td>
                  <td style="padding:2px 4px">{cand.mean_hazard_score:.0f}/100</td></tr>
              </table>
              <p style="margin:6px 0 0 0;font-size:10px;color:#555;border-top:1px solid #eee;padding-top:4px">
                {cand.notes[:120]}{"…" if len(cand.notes) > 120 else ""}
              </p>
              <p style="margin:4px 0 0 0;font-size:9px;color:#999">
                ⚠ Decision-support candidate only — not an official relocation site.
              </p>
            </div>
            """

            # Star marker for rank-1, circle for others
            radius = 10 if rank == 0 else 7
            color  = "#27ae60" if rank == 0 else "#52be80"

            folium.CircleMarker(
                location=[cand.centroid_lat, cand.centroid_lon],
                radius=radius,
                color="#1e8449",
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=2,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=280),
            ).add_to(fg)

            # Label the candidate with its letter
            try:
                folium.Marker(
                    location=[cand.centroid_lat, cand.centroid_lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size:9px;font-weight:bold;color:white;'
                             f'background:#27ae60;border-radius:50%;width:14px;height:14px;'
                             f'display:flex;align-items:center;justify-content:center;'
                             f'border:1px solid #1e8449">{label}</div>',
                        icon_size=(14, 14),
                        icon_anchor=(7, 7),
                    ),
                ).add_to(fg)
            except Exception:
                pass   # DivIcon sometimes not available — skip

    fg.add_to(folium_map)
    return folium_map


# ── Emergency Facilities & Evacuation Routes Layers ──────────────────────────

def add_emergency_facilities_layer(
    folium_map: folium.Map,
    hospitals: list | None = None,
    shelters: list | None = None,
) -> folium.Map:
    """
    Add emergency facility markers (hospitals, shelters) to the map.

    Hospitals are displayed with red cross icons in a "Hospitals" FeatureGroup.
    Shelters are displayed with house icons in a "Shelters" FeatureGroup.
    Both groups can be toggled independently via the layer control.

    Parameters
    ----------
    folium_map : folium.Map
        Map to which layers will be added.
    hospitals : list[EmergencyFacility] | None
        List of hospital facilities. If None, hospital layer is skipped.
    shelters : list[EmergencyFacility] | None
        List of shelter facilities. If None, shelter layer is skipped.

    Returns
    -------
    folium.Map
        Map with new facility layers added.
    """
    import logging
    logger = logging.getLogger(__name__)

    if hospitals:
        fg_hospitals = folium.FeatureGroup(name="🏥 Hospitals", show=True)
        for facility in hospitals:
            try:
                # Extract metadata for popup
                osm_link = (
                    f'<a href="https://www.openstreetmap.org/node/{facility.osm_id}" '
                    f'target="_blank">View in OSM</a>'
                    if facility.osm_id
                    else ""
                )
                contact = facility.metadata.get("contact:phone", "Not available")
                operator = facility.metadata.get("operator", "Unknown")

                popup_html = f"""
                <div style="font-family:sans-serif;font-size:12px;min-width:200px;max-width:280px">
                  <h4 style="margin:0 0 6px 0;color:#c0392b">🏥 {facility.name}</h4>
                  <table style="width:100%;border-collapse:collapse">
                    <tr><td style="color:#666;padding:2px 4px">Type</td>
                        <td style="padding:2px 4px">{facility.facility_type.title()}</td></tr>
                    <tr style="background:#f8f8f8">
                      <td style="color:#666;padding:2px 4px">Operator</td>
                      <td style="padding:2px 4px">{operator}</td></tr>
                    <tr><td style="color:#666;padding:2px 4px">Phone</td>
                        <td style="padding:2px 4px">{contact}</td></tr>
                    <tr style="background:#f8f8f8">
                      <td style="color:#666;padding:2px 4px">Coordinates</td>
                      <td style="padding:2px 4px">{facility.latitude:.4f}, {facility.longitude:.4f}</td></tr>
                    <tr><td style="color:#666;padding:2px 4px">Source</td>
                        <td style="padding:2px 4px">{facility.source}</td></tr>
                  </table>
                  <p style="margin:6px 0 0 0;font-size:10px;color:#555;border-top:1px solid #eee;padding-top:4px">
                    {osm_link}
                  </p>
                </div>
                """

                tooltip_html = f"<b>{facility.name}</b><br>🏥 {facility.facility_type.title()}"

                folium.Marker(
                    location=[facility.latitude, facility.longitude],
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color="red", icon="plus", prefix="fa"),
                ).add_to(fg_hospitals)
            except Exception as e:
                logger.warning("Failed to add hospital marker for %s: %s", facility.name, e)

        fg_hospitals.add_to(folium_map)

    if shelters:
        fg_shelters = folium.FeatureGroup(name="🏠 Shelters", show=True)
        for facility in shelters:
            try:
                osm_link = (
                    f'<a href="https://www.openstreetmap.org/node/{facility.osm_id}" '
                    f'target="_blank">View in OSM</a>'
                    if facility.osm_id
                    else ""
                )
                operator = facility.metadata.get("operator", "Unknown")
                capacity = facility.metadata.get("capacity", "Unknown")

                popup_html = f"""
                <div style="font-family:sans-serif;font-size:12px;min-width:200px;max-width:280px">
                  <h4 style="margin:0 0 6px 0;color:#27ae60">🏠 {facility.name}</h4>
                  <table style="width:100%;border-collapse:collapse">
                    <tr><td style="color:#666;padding:2px 4px">Type</td>
                        <td style="padding:2px 4px">{facility.facility_type.title()}</td></tr>
                    <tr style="background:#f8f8f8">
                      <td style="color:#666;padding:2px 4px">Operator</td>
                      <td style="padding:2px 4px">{operator}</td></tr>
                    <tr><td style="color:#666;padding:2px 4px">Capacity</td>
                        <td style="padding:2px 4px">{capacity}</td></tr>
                    <tr style="background:#f8f8f8">
                      <td style="color:#666;padding:2px 4px">Coordinates</td>
                      <td style="padding:2px 4px">{facility.latitude:.4f}, {facility.longitude:.4f}</td></tr>
                    <tr><td style="color:#666;padding:2px 4px">Source</td>
                        <td style="padding:2px 4px">{facility.source}</td></tr>
                  </table>
                  <p style="margin:6px 0 0 0;font-size:10px;color:#555;border-top:1px solid #eee;padding-top:4px">
                    {osm_link}
                  </p>
                </div>
                """

                tooltip_html = f"<b>{facility.name}</b><br>🏠 {facility.facility_type.title()}"

                folium.Marker(
                    location=[facility.latitude, facility.longitude],
                    tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color="green", icon="home", prefix="fa"),
                ).add_to(fg_shelters)
            except Exception as e:
                logger.warning("Failed to add shelter marker for %s: %s", facility.name, e)

        fg_shelters.add_to(folium_map)

    return folium_map


def add_evacuation_routes_layer(
    folium_map: folium.Map,
    evacuation_routes: list | None = None,
) -> folium.Map:
    """
    Add evacuation route polylines to the map.

    Routes are colour-coded based on hazard exposure:
      - GREEN (mostly safe): RED exposure <= 5%
      - YELLOW (mixed):      RED exposure 5–20%
      - ORANGE (hazardous):  RED exposure > 20%

    Route popups show:
      - Origin & destination habitation/facility names
      - Distance (km) and routing method
      - Hazard exposure breakdown (RED/YELLOW/GREEN/WATER %)
      - Route status

    Parameters
    ----------
    folium_map : folium.Map
        Map to which routes will be added.
    evacuation_routes : list[EvacuationRoute] | None
        List of EvacuationRoute objects. If None, layer is skipped.

    Returns
    -------
    folium.Map
        Map with evacuation routes layer added.
    """
    import logging
    logger = logging.getLogger(__name__)

    if not evacuation_routes:
        return folium_map

    fg = folium.FeatureGroup(name="🚨 Evacuation Routes", show=True)

    for route in evacuation_routes:
        if route.status != "FOUND":
            # Skip unsuccessful routes (no geometry to display)
            continue

        if not route.route_geometry or len(route.route_geometry) < 2:
            logger.debug("Route %s has insufficient geometry", route.hab_id)
            continue

        # Determine route colour based on RED zone exposure
        red_exposure = route.hazard_exposure.get("RED", 0.0)
        if red_exposure <= 5.0:
            route_color = "#27ae60"  # green
            route_weight = 3
        elif red_exposure <= 20.0:
            route_color = "#f39c12"  # orange
            route_weight = 3
        else:
            route_color = "#e74c3c"  # red
            route_weight = 3

        # Build popup with route details
        hazard_str = route.hazard_summary()
        popup_html = f"""
        <div style="font-family:sans-serif;font-size:11px;min-width:240px;max-width:320px">
          <h4 style="margin:0 0 6px 0;color:#1a252f">
            🚨 Evacuation Route
          </h4>
          <table style="width:100%;border-collapse:collapse">
            <tr><td style="color:#666;padding:2px 4px;width:40%"><b>From</b></td>
                <td style="padding:2px 4px"><i>{route.hab_name}</i></td></tr>
            <tr style="background:#f8f8f8">
              <td style="color:#666;padding:2px 4px"><b>To</b></td>
              <td style="padding:2px 4px"><i>{route.facility_name}</i> ({route.facility_type.title()})</td></tr>
            <tr><td style="color:#666;padding:2px 4px"><b>Distance</b></td>
                <td style="padding:2px 4px"><b>{route.distance_km:.2f} km</b></td></tr>
            <tr style="background:#f8f8f8">
              <td style="color:#666;padding:2px 4px"><b>Method</b></td>
              <td style="padding:2px 4px">{route.routing_method.replace("_", " ").title()}</td></tr>
            <tr><td style="color:#666;padding:2px 4px;vertical-align:top"><b>Hazard</b></td>
                <td style="padding:2px 4px"><code style="font-size:10px;color:#555">{hazard_str}</code></td></tr>
            <tr style="background:#f8f8f8">
              <td style="color:#666;padding:2px 4px"><b>Status</b></td>
              <td style="padding:2px 4px">{route.status}</td></tr>
          </table>
          <p style="margin:6px 0 0 0;font-size:9px;color:#999;border-top:1px solid #eee;padding-top:4px">
            ⚠ <b>Decision-support only</b> — not an official evacuation order. Authority decision-makers must verify facility capacity and issue official orders.
          </p>
        </div>
        """

        # Draw polyline for the route
        try:
            folium.PolyLine(
                locations=route.route_geometry,
                color=route_color,
                weight=route_weight,
                opacity=0.8,
                popup=folium.Popup(popup_html, max_width=350),
            ).add_to(fg)
        except Exception as e:
            logger.warning("Failed to draw route polyline for %s: %s", route.hab_id, e)
            continue

        # Add start marker (origin habitation)
        try:
            if route.route_geometry:
                start_lat, start_lon = route.route_geometry[0]
                tooltip_start = f"<b>Start:</b> {route.hab_name}"
                folium.CircleMarker(
                    location=[start_lat, start_lon],
                    radius=5,
                    color="#1a252f",
                    fill=True,
                    fill_color="#3498db",
                    fill_opacity=0.8,
                    weight=2,
                    tooltip=folium.Tooltip(tooltip_start, sticky=True),
                ).add_to(fg)
        except Exception as e:
            logger.debug("Failed to add start marker: %s", e)

        # Add end marker (destination facility)
        try:
            if route.route_geometry:
                end_lat, end_lon = route.route_geometry[-1]
                tooltip_end = f"<b>Destination:</b> {route.facility_name}"
                folium.CircleMarker(
                    location=[end_lat, end_lon],
                    radius=6,
                    color="#1a252f",
                    fill=True,
                    fill_color="#2ecc71",
                    fill_opacity=0.9,
                    weight=2,
                    tooltip=folium.Tooltip(tooltip_end, sticky=True),
                ).add_to(fg)
        except Exception as e:
            logger.debug("Failed to add end marker: %s", e)

    fg.add_to(folium_map)
    return folium_map
