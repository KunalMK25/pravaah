"""
PRAVAAH — Predictive Risk & Vulnerability Assessment for At-Risk Habitations

AI-powered geospatial decision-support system that identifies hazard-based red
zones, evaluates vulnerable habitations, assesses exposure and carrying-capacity
stress, and prioritises intervention or relocation using explainable spatial
intelligence.

Run with: streamlit run app.py
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from flood_risk_zonation.config import BoundingBox, PipelineConfig, validate_bbox_size
from flood_risk_zonation.exceptions import FloodRiskError
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
from flood_risk_zonation.ingest.sample_data import DEMO_REGIONS, DemoRegion
from flood_risk_zonation.pipeline import FloodRiskPipeline, _load_land_mask
from flood_risk_zonation.sih_pipeline import SIHPipeline
from flood_risk_zonation.scoring.susceptibility import (
    WeightedSusceptibilityModel,
    RandomForestSusceptibilityModel,  # noqa: F401
)
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PRAVAAH — Hazard & Habitation Intelligence",
    page_icon="🌊",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌊 PRAVAAH")
st.sidebar.caption("Predictive Risk & Vulnerability Assessment for At-Risk Habitations")
st.sidebar.markdown("---")

# ── Region Selection ──────────────────────────────────────────────────────────
PRESET_REGIONS = {
    "Gottigere, Bangalore": {
        "min_lon": 77.55, "min_lat": 12.84, "max_lon": 77.62, "max_lat": 12.91,
        "area_name": "Gottigere, Bangalore",
        "offline_key": "Bangalore (Gottigere)",
    },
    "Chennai Marina (Coastal)": {
        "min_lon": 80.24, "min_lat": 12.98, "max_lon": 80.31, "max_lat": 13.05,
        "area_name": "Chennai Marina, Chennai",
        "offline_key": "Chennai Marina (Coastal)",
    },
    "Dal Lake, Srinagar": {
        "min_lon": 74.83, "min_lat": 34.07, "max_lon": 74.90, "max_lat": 34.14,
        "area_name": "Dal Lake, Srinagar",
        "offline_key": "Dal Lake, Srinagar",
    },
    "Puri, Odisha (Cyclone Coast)": {
        "min_lon": 85.80, "min_lat": 19.77, "max_lon": 85.87, "max_lat": 19.84,
        "area_name": "Puri, Odisha",
        "offline_key": None,
    },
    "✏️ Custom Region": {
        "min_lon": 77.55, "min_lat": 12.84, "max_lon": 77.62, "max_lat": 12.91,
        "area_name": "",
        "offline_key": None,
    },
}

st.sidebar.subheader("Region Selection")
selected_preset = st.sidebar.selectbox(
    "Select Region",
    list(PRESET_REGIONS.keys()),
    index=0,
)
preset = PRESET_REGIONS[selected_preset]
is_custom = selected_preset == "✏️ Custom Region"

offline_key = preset["offline_key"]
has_offline = offline_key is not None and offline_key in DEMO_REGIONS

if has_offline:
    use_offline = st.sidebar.checkbox(
        "📦 Use offline sample data",
        value=False,
        help="Use pre-bundled synthetic data — no network required.",
    )
else:
    use_offline = False

offline_region: DemoRegion | None = None
if use_offline and has_offline:
    offline_region = DEMO_REGIONS[offline_key]
    st.sidebar.info(f"📦 Offline mode — **{selected_preset}**")

st.sidebar.markdown("---")

area_name_input = st.sidebar.text_input(
    "Area Name (for report)",
    value=preset["area_name"] if not is_custom else "",
    disabled=not is_custom,
)

col1, col2 = st.sidebar.columns(2)
min_lon = col1.number_input("Min Lon", value=float(preset["min_lon"]),
    min_value=-180.0, max_value=180.0, step=0.01, disabled=not is_custom or use_offline)
min_lat = col2.number_input("Min Lat", value=float(preset["min_lat"]),
    min_value=-90.0, max_value=90.0, step=0.01, disabled=not is_custom or use_offline)
max_lon = col1.number_input("Max Lon", value=float(preset["max_lon"]),
    min_value=-180.0, max_value=180.0, step=0.01, disabled=not is_custom or use_offline)
max_lat = col2.number_input("Max Lat", value=float(preset["max_lat"]),
    min_value=-90.0, max_value=90.0, step=0.01, disabled=not is_custom or use_offline)

st.sidebar.subheader("Grid Resolution")
resolution_map = {"250m": 250, "500m": 500, "1000m": 1000}
resolution_label = st.sidebar.selectbox("Cell Size", list(resolution_map.keys()), index=1)
cell_size = resolution_map[resolution_label]

st.sidebar.subheader("Hazard Model")
method_choice = st.sidebar.radio(
    "Susceptibility Model",
    ["Ensemble (WSI + RF)", "Weighted Index (WSI)", "Random Forest (ML)"],
    index=0,
    help=(
        "Ensemble: blends WSI + Random Forest — recommended.\n"
        "WSI: transparent weighted index, instant results.\n"
        "RF: Random Forest only."
    ),
)
_model_type_map = {
    "Ensemble (WSI + RF)": "ensemble",
    "Weighted Index (WSI)": "weighted_susceptibility",
    "Random Forest (ML)": "random_forest",
}
selected_model_type = _model_type_map[method_choice]

st.sidebar.subheader("Classification Thresholds")
low_threshold = st.sidebar.slider("Low / Medium boundary", 10.0, 49.0, 33.0, 1.0)
medium_threshold = st.sidebar.slider("Medium / High boundary", 51.0, 90.0, 66.0, 1.0)

st.sidebar.subheader("Habitation Intelligence")
run_sih = st.sidebar.checkbox(
    "🏘️ Include Habitation Analysis",
    value=True,
    help="Fetch settlements from OSM and run exposure, vulnerability, and relocation assessment.",
)

run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# Land mask status
try:
    _load_land_mask()
    st.sidebar.markdown(
        '<p style="color:#2ecc71;font-size:12px;margin-top:5px;text-align:center">'
        '🟢 <b>Land/Sea Mask</b>: Active</p>', unsafe_allow_html=True
    )
except Exception:
    st.sidebar.markdown(
        '<p style="color:#e74c3c;font-size:12px;margin-top:5px;text-align:center">'
        '🔴 <b>Land/Sea Mask</b>: Error</p>', unsafe_allow_html=True
    )

# Bbox size preview
if not use_offline:
    from math import cos, radians as _rad
    _center_lat = (float(min_lat) + float(max_lat)) / 2.0
    _h = (float(max_lat) - float(min_lat)) * 111.32
    _w = (float(max_lon) - float(min_lon)) * 111.32 * cos(_rad(_center_lat))
    if _w > 0 and _h > 0:
        from flood_risk_zonation.config import BBOX_MIN_SIDE_KM, BBOX_MAX_SIDE_KM
        _ok = (_w >= BBOX_MIN_SIDE_KM and _h >= BBOX_MIN_SIDE_KM
               and _w <= BBOX_MAX_SIDE_KM and _h <= BBOX_MAX_SIDE_KM)
        st.sidebar.caption(f"{'✅' if _ok else '⚠️'} {_w:.1f} km × {_h:.1f} km")

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🌊 PRAVAAH")
st.markdown(
    "**Predictive Risk & Vulnerability Assessment for At-Risk Habitations**  \n"
    "*AI-powered geospatial decision support for hazard red zones, "
    "vulnerable habitations, carrying capacity, and relocation prioritisation.*"
)
st.markdown("---")

if "result" not in st.session_state:
    st.session_state.result = None
if "sih_result" not in st.session_state:
    st.session_state.sih_result = None
if "fallback_warning" not in st.session_state:
    st.session_state.fallback_warning = False

if run_button:
    st.session_state.fallback_warning = False
    st.session_state.sih_result = None
    try:
        if offline_region is not None:
            bbox = offline_region.bbox
        else:
            bbox = BoundingBox(
                min_lon=float(min_lon), min_lat=float(min_lat),
                max_lon=float(max_lon), max_lat=float(max_lat),
            )

        config = PipelineConfig(
            cell_size_meters=float(cell_size),
            model_type=selected_model_type,
            rf_n_estimators=100,
            low_threshold=float(low_threshold),
            medium_threshold=float(medium_threshold),
            use_cache=False,
            allow_network=not use_offline,
        )

        if not use_offline:
            size_error = validate_bbox_size(bbox)
            if size_error:
                st.error(f"📐 **Invalid area size** — {size_error}")
                st.stop()

        pipeline = FloodRiskPipeline(config)

        with st.status("Running PRAVAAH analysis…", expanded=True) as status:
            progress = st.write

            if use_offline and offline_region is not None:
                st.write("📦 Loading offline sample data…")
                from flood_risk_zonation.ingest.sample_data import (
                    get_demo_elevation, get_demo_rainfall, get_demo_water_bodies,
                )
                from flood_risk_zonation.ingest.population import load_population

                elevation = get_demo_elevation(offline_region, resolution_m=30.0)
                rainfall = get_demo_rainfall(offline_region)
                water_bodies = get_demo_water_bodies(offline_region)
                population = load_population(bbox, data_dir=str(config.cache_dir))
                provenance = {
                    "elevation": "offline_sample", "rainfall": "offline_sample",
                    "water_bodies": "offline_sample", "population": population.source,
                }
                result = pipeline.run_from_ingested_data(
                    bounding_box=bbox, elevation=elevation, rainfall=rainfall,
                    water_bodies=water_bodies, population=population,
                    provenance=provenance, data_tier=3, progress_callback=progress,
                )
            else:
                result = pipeline.run(bbox, progress_callback=progress)
                if result.data_provenance.get("water_bodies") == "fallback":
                    st.session_state.fallback_warning = True

            # Habitation intelligence
            if run_sih:
                st.write("🏘️ Running habitation intelligence…")
                sih_pipe = SIHPipeline(
                    config=config,
                    allow_network=not use_offline,
                )
                sih_result = sih_pipe.run_sih_stages(
                    result, bbox, progress_callback=progress
                )
                st.session_state.sih_result = sih_result
                n_habs = len(sih_result.habitation_dataset.habitations)
                n_red = len(sih_result.red_zone_habitations)
                n_crit = len(sih_result.critical_habitations)
                st.write(
                    f"🏘️ {n_habs} settlements assessed | "
                    f"{n_red} in red zone | {n_crit} critical priority"
                )

            status.update(label="✅ PRAVAAH analysis complete", state="complete", expanded=False)

        st.session_state.result = result
        sih_r = st.session_state.sih_result
        if sih_r:
            n_c = len(sih_r.critical_habitations)
            n_h = sum(1 for r in sih_r.relocation_results if r.priority_class == "HIGH")
            if n_c > 0:
                st.error(
                    f"🚨 **{n_c} CRITICAL habitation(s)** require immediate relocation consideration. "
                    "See the **Relocation Priority** tab.",
                    icon="🚨",
                )
            elif n_h > 0:
                st.warning(
                    f"⚠️ **{n_h} HIGH priority habitation(s)** require evacuation planning.",
                    icon="⚠️",
                )
            else:
                st.success(
                    f"✅ Analysis complete — {result.cell_count} cells in "
                    f"{result.pipeline_duration_seconds:.1f}s (Tier {result.data_tier} data) | "
                    f"{len(sih_r.habitation_dataset.habitations)} habitations assessed"
                )
        else:
            st.success(
                f"✅ Analysis complete — {result.cell_count} cells in "
                f"{result.pipeline_duration_seconds:.1f}s (Tier {result.data_tier} data)"
            )

    except FloodRiskError as exc:
        st.error(f"Analysis error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        logger.exception("Unhandled exception in PRAVAAH pipeline")

# Fallback warning banner
if st.session_state.get("fallback_warning"):
    st.warning(
        "⚠️ **OSM Overpass unreachable** — water body data unavailable for this run. "
        "Risk scores are based on elevation, terrain, and rainfall only. "
        "Try again later, or enable **offline sample data** for a reliable demo.",
        icon="⚠️",
    )

result = st.session_state.result
sih_result = st.session_state.sih_result

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_hazard, tab_habitations, tab_relocation, tab_stats, tab_factors, tab_data, tab_methodology = st.tabs([
    "🗺️ Hazard Map",
    "🏘️ Habitations",
    "🚨 Relocation Priority",
    "📊 Risk Statistics",
    "📈 Factor Weights",
    "📋 Data & Export",
    "📖 Methodology",
])

# ─── Tab 1: Hazard Map ────────────────────────────────────────────────────────
with tab_hazard:
    if result is not None:
        center = result.bounding_box.center
        builder = FloodRiskMapBuilder()
        _model = result.analysis_result.model
        _model_bounds = None
        if hasattr(_model, "lower_") and hasattr(_model, "upper_"):
            _model_bounds = {
                f: (_model.lower_[f], _model.upper_[f])
                for f in _model.lower_ if f in _model.upper_
            }

        exp_list = sih_result.exposure_results if sih_result else None
        rel_list = sih_result.relocation_results if sih_result else None

        m = builder.build_choropleth_map(
            result.scored_grid,
            center=center,
            zoom_start=11,
            model_bounds=_model_bounds,
            exposure_results=exp_list,
            relocation_results=rel_list,
            show_red_zones=True,
        )
        st.components.v1.html(m._repr_html_(), height=620, scrolling=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown("🔴 **High Risk** (Red Zone)")
        c2.markdown("🟡 **Medium Risk**")
        c3.markdown("🟢 **Low Risk**")
        c4.markdown("🔵 **Water Body**")
        if sih_result and sih_result.habitation_dataset.habitations:
            st.caption(
                "🏘️ Habitation markers colour-coded by relocation priority — click a marker for details."
            )
    else:
        st.info(
            "Configure parameters in the sidebar and click **Run Analysis** to generate a map.\n\n"
            "PRAVAAH will identify hazard zones, locate vulnerable habitations, and prioritise "
            "relocation needs — all with transparent, explainable scoring."
        )

# ─── Tab 2: Habitations ───────────────────────────────────────────────────────
with tab_habitations:
    if sih_result is not None and sih_result.exposure_results:
        exp_list = sih_result.exposure_results
        rel_map = {r.hab_id: r for r in sih_result.relocation_results}
        vuln_map = {v.hab_id: v for v in sih_result.vulnerability_results}
        cap_map = {c.hab_id: c for c in sih_result.capacity_results}

        st.subheader("Habitation Exposure Summary")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Habitations", len(exp_list))
        col_b.metric(
            "In Red Zone",
            sum(1 for e in exp_list if e.is_in_red_zone),
            delta="High Risk" if any(e.is_in_red_zone for e in exp_list) else "None",
            delta_color="inverse",
        )
        col_c.metric(
            "CRITICAL Priority",
            sum(1 for r in sih_result.relocation_results if r.priority_class == "CRITICAL"),
            delta_color="inverse",
        )
        col_d.metric(
            "HIGH Priority",
            sum(1 for r in sih_result.relocation_results if r.priority_class == "HIGH"),
            delta_color="inverse",
        )

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            hazard_filter = st.multiselect(
                "Filter by Hazard Class",
                ["High", "Medium", "Low", "Water"],
                default=["High", "Medium", "Low", "Water"],
            )
        with filter_col2:
            priority_filter = st.multiselect(
                "Filter by Relocation Priority",
                ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            )

        PRIORITY_EMOJI = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "🔔", "LOW": "✅"}

        rows = []
        for exp in exp_list:
            rel = rel_map.get(exp.hab_id)
            vuln = vuln_map.get(exp.hab_id)
            cap = cap_map.get(exp.hab_id)
            priority = rel.priority_class if rel else "—"
            if exp.hazard_class not in hazard_filter:
                continue
            if rel and rel.priority_class not in priority_filter:
                continue
            pop = (
                f"{exp.population_exposed:,}"
                if exp.population_source == "osm_tag" and exp.population_exposed
                else "UNKNOWN"
            )
            rows.append({
                "Name": exp.name or "Unnamed",
                "Type": exp.hab_type,
                "Hazard Class": exp.hazard_class,
                "Hazard Score": round(exp.hazard_score, 1),
                "Priority": f"{PRIORITY_EMOJI.get(priority, '')} {priority}",
                "Relocation Score": round(rel.relocation_score, 3) if rel else "—",
                "Vulnerability": vuln.vulnerability_class if vuln else "—",
                "Capacity": cap.capacity_status if cap else "—",
                "Population": pop,
                "Lat": round(exp.lat, 5),
                "Lon": round(exp.lon, 5),
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No habitations match the current filters.")

        # Detail panel
        st.subheader("Habitation Detail")
        hab_names = [
            f"{PRIORITY_EMOJI.get(rel_map.get(e.hab_id, type('o', (), {'priority_class': '—'})()).priority_class, '')} "
            f"{e.name or 'Unnamed'} ({e.hab_type})"
            for e in exp_list
        ]
        selected_hab_idx = st.selectbox(
            "Select a habitation for detailed breakdown",
            range(len(exp_list)),
            format_func=lambda i: hab_names[i],
        )
        if selected_hab_idx is not None:
            exp = exp_list[selected_hab_idx]
            rel = rel_map.get(exp.hab_id)
            vuln = vuln_map.get(exp.hab_id)
            cap = cap_map.get(exp.hab_id)

            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown(f"#### {exp.name or 'Unnamed Habitation'}")
                st.markdown(f"**Type:** {exp.hab_type} | **Location:** {exp.lat:.5f}°N, {exp.lon:.5f}°E")
                _hazard_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Water": "🔵"}
                st.markdown(
                    f"**Hazard:** {_hazard_color.get(exp.hazard_class, '')} {exp.hazard_class} "
                    f"(score: {exp.hazard_score:.1f}/100)"
                )
                pop_str = (
                    f"{exp.population_exposed:,} *(from OSM)*"
                    if exp.population_source == "osm_tag" and exp.population_exposed
                    else "**UNKNOWN** *(not in OSM data)*"
                )
                st.markdown(f"**Population Exposed:** {pop_str}")
                st.markdown(f"**In Red Zone:** {'Yes 🔴' if exp.is_in_red_zone else 'No'}")

            with col_r:
                if rel:
                    _pc = rel.priority_class
                    _pc_color = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "🔔", "LOW": "✅"}
                    st.markdown(f"#### Relocation Priority: {_pc_color.get(_pc, '')} **{_pc}**")
                    st.progress(rel.relocation_score, text=f"Relocation score: {rel.relocation_score:.3f}")
                    st.info(rel.recommended_action)

            if vuln:
                st.markdown("##### Vulnerability Breakdown")
                vuln_cols = st.columns(len(vuln.component_scores))
                for i, (comp, score) in enumerate(vuln.component_scores.items()):
                    w = vuln.component_weights.get(comp, 0)
                    vuln_cols[i].metric(
                        comp.replace("_", " ").title(),
                        f"{score:.2f}",
                        delta=f"weight {w:.0%}",
                        delta_color="off",
                    )
                st.markdown(
                    f"**Overall Vulnerability:** {vuln.vulnerability_class} "
                    f"(score: {vuln.vulnerability_score:.3f})"
                )

            if cap:
                st.markdown("##### Carrying Capacity")
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Status", cap.capacity_status)
                cc2.metric(
                    "Safe Area", f"{cap.safe_area_km2:.2f} km²",
                    help=f"Low-risk land within {cap.search_radius_km:.0f}km radius",
                )
                cc3.metric(
                    "Nearest Road",
                    f"{cap.nearest_road_km:.1f} km" if cap.nearest_road_km >= 0 else "Not found",
                )
                cc4.metric(
                    "Nearest Healthcare",
                    f"{cap.nearest_healthcare_km:.1f} km" if cap.nearest_healthcare_km >= 0 else "Not found",
                )
                st.caption(cap.notes)

            if rel and rel.explanation:
                with st.expander("📋 Full Explainability Report"):
                    st.code(rel.explanation, language=None)

    elif result is not None and not run_sih:
        st.info(
            "Enable **Include Habitation Analysis** in the sidebar and re-run "
            "to see settlement exposure, vulnerability, and relocation assessments."
        )
    else:
        st.info("Run the analysis first.")

# ─── Tab 3: Relocation Priority ───────────────────────────────────────────────
with tab_relocation:
    if sih_result is not None and sih_result.relocation_results:
        rel_list = sih_result.relocation_results
        exp_map_r = {e.hab_id: e for e in sih_result.exposure_results}
        cap_map_r = {c.hab_id: c for c in sih_result.capacity_results}
        vuln_map_r = {v.hab_id: v for v in sih_result.vulnerability_results}

        PRIORITY_COLORS_CHART = {
            "CRITICAL": "#c0392b",
            "HIGH": "#e67e22",
            "MEDIUM": "#f1c40f",
            "LOW": "#2ecc71",
        }

        st.subheader("Relocation Priority Overview")
        priority_counts: dict[str, int] = {}
        for r in rel_list:
            priority_counts[r.priority_class] = priority_counts.get(r.priority_class, 0) + 1

        col_crit, col_high, col_med, col_low = st.columns(4)
        col_crit.metric("🚨 CRITICAL", priority_counts.get("CRITICAL", 0))
        col_high.metric("⚠️ HIGH", priority_counts.get("HIGH", 0))
        col_med.metric("🔔 MEDIUM", priority_counts.get("MEDIUM", 0))
        col_low.metric("✅ LOW", priority_counts.get("LOW", 0))

        if priority_counts:
            fig, ax = plt.subplots(figsize=(6, 3))
            labels = [k for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if k in priority_counts]
            vals = [priority_counts[k] for k in labels]
            colors_chart = [PRIORITY_COLORS_CHART[k] for k in labels]
            ax.barh(labels[::-1], vals[::-1], color=colors_chart[::-1])
            ax.set_xlabel("Number of Habitations")
            ax.set_title("Relocation Priority Distribution")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for i, v in enumerate(vals[::-1]):
                ax.text(v + 0.05, i, str(v), va="center", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Priority Ranked Habitations")
        st.caption(
            "Declared weights: Hazard 35% | Vulnerability 30% | Capacity Stress 20% | Exposure 15%"
        )

        table_rows = []
        for rank, rel in enumerate(rel_list, 1):
            exp = exp_map_r.get(rel.hab_id)
            cap = cap_map_r.get(rel.hab_id)
            pop = (
                f"{rel.population_exposed:,}"
                if rel.population_source == "osm_tag" and rel.population_exposed
                else "UNKNOWN"
            )
            safe = f"{cap.safe_area_km2:.2f} km²" if cap else "—"
            table_rows.append({
                "Rank": rank,
                "Priority": rel.priority_class,
                "Name": rel.name or "Unnamed",
                "Score": round(rel.relocation_score, 3),
                "Hazard": round(rel.hazard_score, 1),
                "Vulnerability": round(rel.vulnerability_score, 3),
                "Cap. Stress": round(1 - rel.capacity_score, 3),
                "Population": pop,
                "Safe Area": safe,
                "Recommended Action": (
                    rel.recommended_action[:60] + "…"
                    if len(rel.recommended_action) > 60
                    else rel.recommended_action
                ),
            })

        if table_rows:
            df_rel = pd.DataFrame(table_rows)

            def _highlight_priority(row):
                bg = {
                    "CRITICAL": "background-color: #fdecea",
                    "HIGH":     "background-color: #fef5e7",
                    "MEDIUM":   "background-color: #fef9e7",
                    "LOW":      "background-color: #eafaf1",
                }
                return [bg.get(row["Priority"], "")] * len(row)

            st.dataframe(
                df_rel.style.apply(_highlight_priority, axis=1),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("ℹ️ Relocation Priority Methodology"):
            st.markdown("""
**Formula (all weights declared):**
```
relocation_score =
    0.35 × hazard_score/100
  + 0.30 × vulnerability_score
  + 0.20 × (1 − capacity_score)
  + 0.15 × exposure_component
```

| Score | Class | Action |
|-------|-------|--------|
| < 0.25 | LOW | Routine monitoring |
| 0.25–0.50 | MEDIUM | Preparedness / monitoring |
| 0.50–0.75 | HIGH | Priority intervention / evacuation planning |
| > 0.75 | CRITICAL | Immediate relocation priority |

**Guardrails (deterministic, documented):**
- Capacity CRITICAL: +0.10 score bonus
- Coastal/tsunami flag: HIGH → CRITICAL escalation
- Unknown population + non-high hazard: capped at HIGH (precautionary principle)

No black-box AI. Every rule is visible here and in the source code.
""")

    elif result is not None and not run_sih:
        st.info("Enable **Include Habitation Analysis** in the sidebar and re-run.")
    else:
        st.info("Run the analysis first.")

# ─── Tab 4: Risk Statistics ───────────────────────────────────────────────────
with tab_stats:
    if result is not None:
        dist = result.risk_distribution
        color_map_s = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c", "Water": "#3498db"}
        dist_no_water = {k: v for k, v in dist.items() if k != "Water"}
        labels_s = list(dist_no_water.keys())
        counts_s = list(dist_no_water.values())
        bar_colors_s = [color_map_s.get(l, "#999") for l in labels_s]

        if counts_s:
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots()
                ax.bar(labels_s, counts_s, color=bar_colors_s)
                ax.set_ylabel("Cell Count")
                ax.set_title("Risk Class Distribution (excl. Water)")
                st.pyplot(fig)
                plt.close(fig)
            with col_b:
                fig2, ax2 = plt.subplots()
                ax2.pie(counts_s, labels=labels_s, colors=bar_colors_s,
                        autopct="%1.1f%%", startangle=90)
                ax2.set_title("Risk Class Share")
                st.pyplot(fig2)
                plt.close(fig2)
        else:
            st.info("The selected area contains only permanent-water cells.")

        n_water = dist.get("Water", 0)
        if n_water:
            st.info(
                f"ℹ️ {n_water} cells classified as permanent water bodies — "
                "excluded from flood risk statistics."
            )
    else:
        st.info("Run the analysis first.")

# ─── Tab 5: Factor Weights ────────────────────────────────────────────────────
with tab_factors:
    if result is not None:
        fi = result.analysis_result.feature_importances
        fig, ax = plt.subplots(figsize=(8, 5))
        feats = list(fi.keys())
        imps = list(fi.values())
        ax.barh(feats[::-1], imps[::-1], color="#3498db")
        ax.set_xlabel("Importance")
        ax.set_title("Hazard Susceptibility Factor Weights")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(result.analysis_result.validation_note)

        ar = result.analysis_result
        if ar.method in ("random_forest", "ensemble") and ar.mean_cv_auc is not None:
            st.markdown("**Cross-Validation Results (5-fold stratified)**")
            cols = st.columns(5)
            cols[0].metric("AUC-ROC", f"{ar.mean_cv_auc:.3f}")
            cols[1].metric("F1 Score", f"{ar.mean_cv_f1:.3f}")
            cols[2].metric("Accuracy", f"{ar.mean_cv_accuracy:.3f}" if ar.mean_cv_accuracy else "—")
            cols[3].metric("Precision", f"{ar.mean_cv_precision:.3f}" if ar.mean_cv_precision else "—")
            cols[4].metric("Recall", f"{ar.mean_cv_recall:.3f}" if ar.mean_cv_recall else "—")
            if ar.cv_auc_scores:
                n_folds = len(ar.cv_auc_scores)
                fold_df = pd.DataFrame({
                    "Fold": [f"Fold {i+1}" for i in range(n_folds)],
                    "AUC-ROC":   [f"{v:.3f}" for v in ar.cv_auc_scores],
                    "F1":        [f"{v:.3f}" for v in (ar.cv_f1_scores or [])],
                    "Accuracy":  [f"{v:.3f}" for v in (ar.cv_accuracy_scores or ["—"]*n_folds)],
                    "Precision": [f"{v:.3f}" for v in (ar.cv_precision_scores or ["—"]*n_folds)],
                    "Recall":    [f"{v:.3f}" for v in (ar.cv_recall_scores or ["—"]*n_folds)],
                })
                st.dataframe(fold_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run the analysis first.")

# ─── Tab 6: Data & Export ─────────────────────────────────────────────────────
with tab_data:
    if result is not None:
        st.subheader("Grid Data")
        display_cols = (
            ["cell_id", "centroid_lat", "centroid_lon"]
            + FEATURE_COLUMNS
            + ["risk_score", "risk_class"]
        )
        available = [c for c in display_cols if c in result.scored_grid.columns]
        df = result.scored_grid[available].copy()

        risk_filter_d = st.multiselect(
            "Filter by Risk Class", ["Low", "Medium", "High", "Water"],
            default=["Low", "Medium", "High", "Water"],
        )
        df_filtered = df[df["risk_class"].isin(risk_filter_d)]
        st.dataframe(df_filtered, use_container_width=True)

        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
        with col_dl1:
            csv_buf = io.StringIO()
            df_filtered.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Hazard Grid CSV", csv_buf.getvalue(),
                file_name="PRAVAAH_Hazard_Grid.csv", mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                "⬇️ Hazard GeoJSON", result.scored_grid.to_json(),
                file_name="PRAVAAH_Hazard_Map.geojson", mime="application/json",
            )

        # Habitation intelligence export
        if sih_result and sih_result.relocation_results:
            with col_dl3:
                rows_export = []
                exp_m = {e.hab_id: e for e in sih_result.exposure_results}
                vuln_m = {v.hab_id: v for v in sih_result.vulnerability_results}
                cap_m = {c.hab_id: c for c in sih_result.capacity_results}
                for rel in sih_result.relocation_results:
                    exp = exp_m.get(rel.hab_id)
                    vuln = vuln_m.get(rel.hab_id)
                    cap = cap_m.get(rel.hab_id)
                    rows_export.append({
                        "hab_id": rel.hab_id,
                        "name": rel.name,
                        "type": exp.hab_type if exp else "",
                        "lat": exp.lat if exp else "",
                        "lon": exp.lon if exp else "",
                        "hazard_score": rel.hazard_score,
                        "hazard_class": exp.hazard_class if exp else "",
                        "vulnerability_score": rel.vulnerability_score,
                        "vulnerability_class": vuln.vulnerability_class if vuln else "",
                        "capacity_score": rel.capacity_score,
                        "capacity_status": cap.capacity_status if cap else "",
                        "safe_area_km2": cap.safe_area_km2 if cap else "",
                        "nearest_road_km": cap.nearest_road_km if cap else "",
                        "nearest_healthcare_km": cap.nearest_healthcare_km if cap else "",
                        "relocation_score": rel.relocation_score,
                        "priority_class": rel.priority_class,
                        "population_exposed": rel.population_exposed or "UNKNOWN",
                        "population_source": rel.population_source,
                        "recommended_action": rel.recommended_action,
                        "contributing_factors": " | ".join(rel.contributing_factors),
                    })
                hab_csv = io.StringIO()
                pd.DataFrame(rows_export).to_csv(hab_csv, index=False)
                st.download_button(
                    "⬇️ Relocation Priority CSV",
                    hab_csv.getvalue(),
                    file_name="PRAVAAH_Relocation_Priority.csv",
                    mime="text/csv",
                )

        with col_dl4:
            if st.button("📄 Generate PDF Report"):
                with st.spinner("Generating PRAVAAH report…"):
                    try:
                        import tempfile
                        from flood_risk_zonation.visualization.pdf_report import export_pdf_report

                        area_name = (
                            area_name_input.strip()
                            or preset.get("area_name", "")
                            or (
                                f"Lat {result.bounding_box.min_lat:.3f}–"
                                f"{result.bounding_box.max_lat:.3f}"
                            )
                        )
                        with tempfile.TemporaryDirectory() as tmpdir:
                            pdf_path = export_pdf_report(
                                result,
                                Path(tmpdir) / "PRAVAAH_Risk_Assessment.pdf",
                                area_name=area_name,
                                data_tier=result.data_tier,
                                sih_result=sih_result,
                            )
                            pdf_bytes = Path(pdf_path).read_bytes()
                        st.download_button(
                            "⬇️ Download PDF Report",
                            pdf_bytes,
                            file_name="PRAVAAH_Risk_Assessment.pdf",
                            mime="application/pdf",
                        )
                        st.success("PRAVAAH report generated successfully.")
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
    else:
        st.info("Run the analysis first.")

# ─── Tab 7: Methodology ───────────────────────────────────────────────────────
with tab_methodology:
    st.markdown("""
## PRAVAAH — System Methodology

### Core Decision Logic
```
HAZARD  +  EXPOSURE  +  VULNERABILITY  +  CAPACITY STRESS  =  RELOCATION PRIORITY
```

PRAVAAH builds upon an existing geospatial risk-assessment foundation and extends it
with habitation-level exposure, vulnerability, carrying-capacity, and relocation
intelligence. Every score uses declared, auditable weights — no black-box AI.

---

### Hazard Analysis

| Dataset | Source | Resolution |
|---|---|---|
| Elevation (DEM) | NASA SRTM (local GeoTIFF; API/synthetic fallback) | ~30m |
| Rainfall | Synthetic (GPM-structured) | 0.1° |
| Water Bodies | OpenStreetMap Overpass API (locally cached) | Vector |
| Drainage Capacity | Synthetic (population-density-correlated) | per-cell |
| Population Density | Synthetic fallback | ~1km |

Ten conditioning factors are computed per grid cell → normalised → combined via a
Weighted Susceptibility Index (WSI) and optionally blended with a Random Forest model
(Ensemble, default).

---

### Habitation Intelligence

#### Settlement Ingestion
- Source: OpenStreetMap Overpass API (place= nodes: city, town, village, hamlet, suburb, neighbourhood)
- Results cached locally by bounding box
- Fallback: synthetic placeholder points when OSM is unavailable

#### Exposure Analysis
- Each settlement is spatially overlaid against the nearest hazard grid cells
- Metrics: mean hazard score, dominant hazard class, % high-risk cells
- **Population:** OSM `population` tag only → labelled `osm_tag`. Absent → `UNKNOWN`. Never fabricated.

#### Vulnerability Assessment (Transparent Weighted Index)

| Component | Weight | Direction |
|---|---|---|
| Hazard severity | 30% | Higher = more vulnerable |
| Low elevation | 15% | Lower elevation = more vulnerable |
| Water proximity | 15% | Closer to water = more vulnerable |
| Poor drainage | 15% | Lower capacity = more vulnerable |
| Population exposure | 10% | Known exposed population = more vulnerable |
| Road accessibility | 10% | Farther road = more vulnerable |
| Healthcare access | 5% | Farther healthcare = more vulnerable |

Classes: **LOW** (< 0.25) | **MEDIUM** (< 0.50) | **HIGH** (< 0.75) | **CRITICAL** (≥ 0.75)

#### Carrying Capacity Assessment

| Component | Weight | Measurement |
|---|---|---|
| Safe area nearby | 45% | Low-risk land within 5 km radius |
| Road accessibility | 30% | Distance to nearest primary/secondary road (OSM) |
| Healthcare access | 25% | Distance to nearest hospital/clinic (OSM) |

Status: **ADEQUATE** (≥ 0.60) | **STRESSED** (≥ 0.35) | **CRITICAL** (< 0.35)

#### Relocation Priority

```
relocation_score =
    0.35 × hazard_score/100
  + 0.30 × vulnerability_score
  + 0.20 × (1 − capacity_score)
  + 0.15 × exposure_component
```

Guardrails: capacity escalation (+0.10), coastal escalation (HIGH → CRITICAL),
unknown population cap (non-high hazard capped at HIGH).

---

### Data Tiers
- **Tier 1**: Real SRTM + OSM data
- **Tier 2**: Partial real data
- **Tier 3**: Fully synthetic (demo/offline mode)

---

### Scientific Honesty
- Population is **UNKNOWN** when not in OSM — never fabricated
- ML metrics are cross-validation on WSI pseudo-labels, not real flood events
- Shelter capacity is **unavailable** — no curated national dataset
- Road/healthcare distances are **straight-line** (Euclidean), not routed
- All weights are **declared in source code**

---

### Known Limitations
- Drainage capacity is synthetic
- OSM population coverage is sparse for many Indian settlements
- No temporal forecasting in current version
- Habitation completeness depends on OSM mapping quality for the study area
""")
