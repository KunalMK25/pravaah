"""
PRAVAAH — Predictive Risk & Vulnerability Assessment for At-Risk Habitations
Phase 3: RED/YELLOW/GREEN spatial zones + relocation candidates + bounded agentic decision support.
Run with: streamlit run app.py
"""
from __future__ import annotations
import io, logging, os
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
from flood_risk_zonation.scoring.susceptibility import WeightedSusceptibilityModel, RandomForestSusceptibilityModel  # noqa
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="PRAVAAH — Hazard & Habitation Intelligence", page_icon="\U0001f30a", layout="wide")

st.sidebar.title("\U0001f30a PRAVAAH")
st.sidebar.caption("Predictive Risk & Vulnerability Assessment for At-Risk Habitations")
st.sidebar.markdown("---")

PRESET_REGIONS = {
    "Gottigere, Bangalore": {"min_lon": 77.55, "min_lat": 12.84, "max_lon": 77.62, "max_lat": 12.91, "area_name": "Gottigere, Bangalore", "offline_key": "Bangalore (Gottigere)"},
    "Chennai Marina (Coastal)": {"min_lon": 80.24, "min_lat": 12.98, "max_lon": 80.31, "max_lat": 13.05, "area_name": "Chennai Marina, Chennai", "offline_key": "Chennai Marina (Coastal)"},
    "Dal Lake, Srinagar": {"min_lon": 74.83, "min_lat": 34.07, "max_lon": 74.90, "max_lat": 34.14, "area_name": "Dal Lake, Srinagar", "offline_key": "Dal Lake, Srinagar"},
    "Puri, Odisha (Cyclone Coast)": {"min_lon": 85.80, "min_lat": 19.77, "max_lon": 85.87, "max_lat": 19.84, "area_name": "Puri, Odisha", "offline_key": None},
    "\u270f\ufe0f Custom Region": {"min_lon": 77.55, "min_lat": 12.84, "max_lon": 77.62, "max_lat": 12.91, "area_name": "", "offline_key": None},
}

st.sidebar.subheader("Region Selection")
selected_preset = st.sidebar.selectbox("Select Region", list(PRESET_REGIONS.keys()), index=0)
preset = PRESET_REGIONS[selected_preset]
is_custom = "Custom" in selected_preset

offline_key = preset["offline_key"]
has_offline = offline_key is not None and offline_key in DEMO_REGIONS
use_offline = st.sidebar.checkbox("\U0001f4e6 Use offline sample data", value=False, help="Pre-bundled synthetic data.") if has_offline else False
offline_region: DemoRegion | None = None
if use_offline and has_offline:
    offline_region = DEMO_REGIONS[offline_key]
    st.sidebar.info(f"\U0001f4e6 Offline mode — **{selected_preset}**")

st.sidebar.markdown("---")
area_name_input = st.sidebar.text_input("Area Name (for report)", value=preset["area_name"] if not is_custom else "", disabled=not is_custom)
col1, col2 = st.sidebar.columns(2)
min_lon = col1.number_input("Min Lon", value=float(preset["min_lon"]), min_value=-180.0, max_value=180.0, step=0.01, disabled=not is_custom or use_offline)
min_lat = col2.number_input("Min Lat", value=float(preset["min_lat"]), min_value=-90.0, max_value=90.0, step=0.01, disabled=not is_custom or use_offline)
max_lon = col1.number_input("Max Lon", value=float(preset["max_lon"]), min_value=-180.0, max_value=180.0, step=0.01, disabled=not is_custom or use_offline)
max_lat = col2.number_input("Max Lat", value=float(preset["max_lat"]), min_value=-90.0, max_value=90.0, step=0.01, disabled=not is_custom or use_offline)

st.sidebar.subheader("Grid Resolution")
cell_size = {"250m": 250, "500m": 500, "1000m": 1000}[st.sidebar.selectbox("Cell Size", ["250m","500m","1000m"], index=1)]
st.sidebar.subheader("Hazard Model")
selected_model_type = {"Ensemble (WSI + RF)": "ensemble", "Weighted Index (WSI)": "weighted_susceptibility", "Random Forest (ML)": "random_forest"}[
    st.sidebar.radio("Susceptibility Model", ["Ensemble (WSI + RF)", "Weighted Index (WSI)", "Random Forest (ML)"], index=0)]
st.sidebar.subheader("Classification Thresholds")
low_threshold = st.sidebar.slider("Low / Medium boundary", 10.0, 49.0, 33.0, 1.0)
medium_threshold = st.sidebar.slider("Medium / High boundary", 51.0, 90.0, 66.0, 1.0)
st.sidebar.subheader("Habitation Intelligence")
run_sih = st.sidebar.checkbox("\U0001f3d8\ufe0f Include Habitation Analysis", value=True)
run_phase3 = st.sidebar.checkbox("\U0001f916 Spatial Zones + AI Decision Support", value=True,
    help="RED/YELLOW/GREEN zone classification, relocation candidate discovery, and agentic decision support.")

_llm_provider = os.environ.get("PRAVAAH_LLM_PROVIDER", "none").lower()
_has_llm = ((_llm_provider == "openai" and bool(os.environ.get("OPENAI_API_KEY"))) or
            (_llm_provider == "anthropic" and bool(os.environ.get("ANTHROPIC_API_KEY"))))
if _has_llm:
    st.sidebar.markdown(f'<p style="color:#2ecc71;font-size:11px;text-align:center">\U0001f916 AI: {_llm_provider.title()} connected</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p style="color:#95a5a6;font-size:11px;text-align:center">\U0001f916 AI: Rule-based (no LLM key)</p>', unsafe_allow_html=True)

run_button = st.sidebar.button("\U0001f680 Run Analysis", type="primary", use_container_width=True)
try:
    _load_land_mask()
    st.sidebar.markdown('<p style="color:#2ecc71;font-size:12px;margin-top:5px;text-align:center">\U0001f7e2 Land/Sea Mask: Active</p>', unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown('<p style="color:#e74c3c;font-size:12px;margin-top:5px;text-align:center">\U0001f534 Land/Sea Mask: Error</p>', unsafe_allow_html=True)

if not use_offline:
    from math import cos, radians as _rad
    _cl = (float(min_lat)+float(max_lat))/2.0
    _h = (float(max_lat)-float(min_lat))*111.32
    _w = (float(max_lon)-float(min_lon))*111.32*cos(_rad(_cl))
    if _w>0 and _h>0:
        from flood_risk_zonation.config import BBOX_MIN_SIDE_KM, BBOX_MAX_SIDE_KM
        _ok = _w>=BBOX_MIN_SIDE_KM and _h>=BBOX_MIN_SIDE_KM and _w<=BBOX_MAX_SIDE_KM and _h<=BBOX_MAX_SIDE_KM
        st.sidebar.caption(f"{'\u2705' if _ok else '\u26a0\ufe0f'} {_w:.1f} km \u00d7 {_h:.1f} km")

st.title("\U0001f30a PRAVAAH")
st.markdown("**Predictive Risk & Vulnerability Assessment for At-Risk Habitations**  \n"
            "*Geospatial ML hazard engine \u00b7 Spatial RED/YELLOW/GREEN zones \u00b7 Vulnerable habitation intelligence \u00b7 Bounded agentic decision support.*")
st.markdown("---")

for _k,_v in [("result",None),("sih_result",None),("full_result",None),("fallback_warning",False)]:
    if _k not in st.session_state: st.session_state[_k]=_v

PRIORITY_EMOJI = {"CRITICAL":"\U0001f6a8","HIGH":"\u26a0\ufe0f","MEDIUM":"\U0001f514","LOW":"\u2705"}
ZONE_EMOJI    = {"RED":"\U0001f7e5","YELLOW":"\U0001f7e8","GREEN":"\U0001f7e9","WATER":"\U0001f535"}
PCOLORS = {"CRITICAL":"#c0392b","HIGH":"#e67e22","MEDIUM":"#f1c40f","LOW":"#2ecc71"}

if run_button:
    st.session_state.fallback_warning = False
    st.session_state.sih_result = None
    st.session_state.full_result = None
    try:
        bbox = offline_region.bbox if offline_region else BoundingBox(float(min_lon),float(min_lat),float(max_lon),float(max_lat))
        config = PipelineConfig(cell_size_meters=float(cell_size), model_type=selected_model_type,
            rf_n_estimators=100, low_threshold=float(low_threshold), medium_threshold=float(medium_threshold),
            use_cache=False, allow_network=not use_offline)
        if not use_offline:
            err = validate_bbox_size(bbox)
            if err: st.error(f"\U0001f4d0 Invalid area size — {err}"); st.stop()
        pipeline = FloodRiskPipeline(config)
        with st.status("Running PRAVAAH analysis\u2026", expanded=True) as status:
            progress = st.write
            if use_offline and offline_region:
                st.write("\U0001f4e6 Loading offline sample data\u2026")
                from flood_risk_zonation.ingest.sample_data import get_demo_elevation,get_demo_rainfall,get_demo_water_bodies
                from flood_risk_zonation.ingest.population import load_population
                elevation=get_demo_elevation(offline_region,resolution_m=30.0)
                rainfall=get_demo_rainfall(offline_region)
                water_bodies=get_demo_water_bodies(offline_region)
                population=load_population(bbox,data_dir=str(config.cache_dir))
                result=pipeline.run_from_ingested_data(bounding_box=bbox,elevation=elevation,rainfall=rainfall,
                    water_bodies=water_bodies,population=population,
                    provenance={"elevation":"offline_sample","rainfall":"offline_sample","water_bodies":"offline_sample","population":population.source},
                    data_tier=3,progress_callback=progress)
            else:
                result = pipeline.run(bbox,progress_callback=progress)
                if result.data_provenance.get("water_bodies")=="fallback": st.session_state.fallback_warning=True
            sih_result=None; full_result=None
            if run_sih:
                st.write("\U0001f3d8\ufe0f Running habitation intelligence\u2026")
                sih_pipe=SIHPipeline(config=config,allow_network=not use_offline)
                sih_result=sih_pipe.run_sih_stages(result,bbox,progress_callback=progress)
                n_h=len(sih_result.habitation_dataset.habitations)
                st.write(f"\U0001f3d8\ufe0f {n_h} settlements \u00b7 {len(sih_result.red_zone_habitations)} red zone \u00b7 {len(sih_result.critical_habitations)} critical")
                if run_phase3:
                    st.write("\U0001f5fa\ufe0f Classifying spatial zones (RED/YELLOW/GREEN)\u2026")
                    full_result=sih_pipe.run_phase3(sih_result,progress_callback=progress,run_agents=True)
                    st.write(f"\U0001f5fa\ufe0f \U0001f7e5{full_result.red_zone_count} RED \U0001f7e8{full_result.yellow_zone_count} YELLOW \U0001f7e9{full_result.green_zone_count} GREEN "
                             f"\u00b7 {sum(len(v) for v in full_result.relocation_candidates.values())} candidates \u00b7 {len(full_result.agent_decisions)} AI decisions")
            status.update(label="\u2705 PRAVAAH analysis complete",state="complete",expanded=False)
        st.session_state.result=result; st.session_state.sih_result=sih_result; st.session_state.full_result=full_result
        if sih_result:
            nc=len(sih_result.critical_habitations); nh=sum(1 for r in sih_result.relocation_results if r.priority_class=="HIGH")
            if nc>0: st.error(f"\U0001f6a8 **{nc} CRITICAL habitation(s)** — see Relocation Priority and AI Decision Support tabs.",icon="\U0001f6a8")
            elif nh>0: st.warning(f"\u26a0\ufe0f **{nh} HIGH priority habitation(s)** require evacuation planning.",icon="\u26a0\ufe0f")
            else: st.success(f"\u2705 {result.cell_count} cells \u00b7 {result.pipeline_duration_seconds:.1f}s \u00b7 Tier {result.data_tier} \u00b7 {len(sih_result.habitation_dataset.habitations)} habitations assessed")
        else: st.success(f"\u2705 {result.cell_count} cells \u00b7 {result.pipeline_duration_seconds:.1f}s \u00b7 Tier {result.data_tier}")
    except FloodRiskError as exc: st.error(f"Analysis error: {exc}")
    except Exception as exc: st.error(f"Unexpected error: {exc}"); logger.exception("Unhandled exception")

if st.session_state.get("fallback_warning"):
    st.warning("\u26a0\ufe0f OSM Overpass unreachable — water body data unavailable. Try again later or enable offline mode.",icon="\u26a0\ufe0f")

result=st.session_state.result; sih_result=st.session_state.sih_result; full_result=st.session_state.full_result

tab_hazard,tab_zones,tab_habs,tab_reloc,tab_ai,tab_stats,tab_factors,tab_data,tab_meth = st.tabs([
    "\U0001f5fa\ufe0f Hazard Map","\U0001f3a8 Spatial Zones","\U0001f3d8\ufe0f Habitations",
    "\U0001f6a8 Relocation Priority","\U0001f916 AI Decision Support",
    "\U0001f4ca Risk Statistics","\U0001f4c8 Factor Weights","\U0001f4cb Data & Export","\U0001f4d6 Methodology"])

# ── Tab 1: Hazard Map ─────────────────────────────────────────────────────────
with tab_hazard:
    if result is not None:
        center=result.bounding_box.center; builder=FloodRiskMapBuilder()
        _model=result.analysis_result.model; _mb=None
        if hasattr(_model,"lower_") and hasattr(_model,"upper_"):
            _mb={f:(_model.lower_[f],_model.upper_[f]) for f in _model.lower_ if f in _model.upper_}
        exp_list=sih_result.exposure_results if sih_result else None
        rel_list=sih_result.relocation_results if sih_result else None
        zg=full_result.zoned_grid if full_result else None
        cands=full_result.relocation_candidates if full_result else None
        m=builder.build_choropleth_map(result.scored_grid,center=center,zoom_start=11,model_bounds=_mb,
            exposure_results=exp_list,relocation_results=rel_list,
            show_red_zones=(zg is None),zoned_grid=zg,relocation_candidates=cands,show_spatial_zones=(zg is not None))
        st.components.v1.html(m._repr_html_(),height=630,scrolling=False)
        if zg is not None:
            c1,c2,c3,c4=st.columns(4)
            c1.markdown("\U0001f7e5 **RED** — Primary Hazard"); c2.markdown("\U0001f7e8 **YELLOW** — Secondary Attention")
            c3.markdown("\U0001f7e9 **GREEN** — Lower-Risk / Potential Safe Area"); c4.markdown("\U0001f535 **WATER** — Water Body")
            st.caption("Spatial zones are an operational layer derived from the ML model. YELLOW = adjacent to RED or Medium class. GREEN = lower-risk (not an official shelter). "
                       "\U0001f3d8\ufe0f Click habitation markers for details. \U0001f7e9 Green circles = relocation candidate areas.")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.markdown("\U0001f534 **High Risk**"); c2.markdown("\U0001f7e1 **Medium Risk**"); c3.markdown("\U0001f7e2 **Low Risk**"); c4.markdown("\U0001f535 **Water Body**")
    else:
        st.info("Configure parameters in the sidebar and click **Run Analysis** to generate a map.")

# ── Tab 2: Spatial Zones ──────────────────────────────────────────────────────
with tab_zones:
    if full_result is not None and full_result.zoned_grid is not None:
        zg=full_result.zoned_grid
        st.subheader("RED / YELLOW / GREEN Zone Summary")
        st.caption("Operational attention zones derived from the ML hazard model. The underlying risk_score/risk_class are preserved unchanged.")
        cz1,cz2,cz3,cz4=st.columns(4)
        cz1.metric("\U0001f7e5 RED (Primary Hazard)",full_result.red_zone_count,help="Cells with risk_class=High")
        cz2.metric("\U0001f7e8 YELLOW (Secondary)",full_result.yellow_zone_count,help="Adjacent to RED or Medium class")
        cz3.metric("\U0001f7e9 GREEN (Lower-Risk)",full_result.green_zone_count,help="Not RED, YELLOW, or Water")
        cz4.metric("\U0001f535 WATER",int((zg["spatial_zone"]=="WATER").sum()) if "spatial_zone" in zg.columns else 0)
        with st.expander("\u2139\ufe0f Zone Definitions"):
            st.markdown("""| Zone | Meaning | Based on |\n|------|---------|---------|\n"""
                "| \U0001f7e5 **RED** | Primary hazard zone | ML risk_class = High |\n"
                "| \U0001f7e8 **YELLOW** | Secondary attention | Adjacent to RED (8-neighbour) or risk_class = Medium |\n"
                "| \U0001f7e9 **GREEN** | Lower-risk / potential safe area | Not RED, YELLOW, or Water |\n"
                "| \U0001f535 **WATER** | Permanent water body | risk_class = Water |\n\n"
                "**YELLOW** cells are NOT confirmed flood cells. **GREEN** areas are NOT guaranteed safe — they warrant further evaluation.")
        if "spatial_zone" in zg.columns:
            zd=zg["spatial_zone"].value_counts().to_dict()
            zc={"RED":"#c0392b","YELLOW":"#f39c12","GREEN":"#27ae60","WATER":"#2980b9"}
            zo=[z for z in ["RED","YELLOW","GREEN","WATER"] if z in zd]
            fig,ax=plt.subplots(figsize=(7,3))
            ax.bar(zo,[zd[z] for z in zo],color=[zc[z] for z in zo],edgecolor="white",linewidth=0.8)
            ax.set_ylabel("Grid Cells"); ax.set_title("Spatial Zone Distribution")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for i,z in enumerate(zo): ax.text(i,zd[z]+0.3,str(zd[z]),ha="center",va="bottom",fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        if sih_result and sih_result.exposure_results:
            st.subheader("Habitation Zone Assignments")
            zone_rows=[{"Name":e.name or "Unnamed","Spatial Zone":full_result.habitation_zones.get(e.hab_id,"?"),
                "Underlying Hazard":e.hazard_class,"Hazard Score":round(e.hazard_score,1),
                "Priority":(next((r.priority_class for r in sih_result.relocation_results if r.hab_id==e.hab_id),None) or "—"),
                "In Red Zone":"Yes" if e.is_in_red_zone else "No"} for e in sih_result.exposure_results]
            if zone_rows: st.dataframe(pd.DataFrame(zone_rows),use_container_width=True,hide_index=True)
        if full_result.relocation_candidates:
            st.subheader("Relocation Candidate Areas")
            st.caption("GREEN-zone areas for HIGH/CRITICAL habitations. Decision-support recommendations only — not officially designated sites.")
            cr=[]
            for hid,cands in full_result.relocation_candidates.items():
                exp=next((e for e in (sih_result.exposure_results if sih_result else []) if e.hab_id==hid),None)
                for rank,c in enumerate(cands,1):
                    cr.append({"Source":exp.name if exp else hid,"Rank":rank,"Score":f"{c.candidate_score:.3f}",
                        "Distance (km)":f"{c.distance_km:.1f}","Safe Area (km\u00b2)":f"{c.area_km2:.2f}",
                        "Hazard":f"{c.mean_hazard_score:.0f}/100","Notes":c.notes[:80]+"\u2026" if len(c.notes)>80 else c.notes})
            if cr: st.dataframe(pd.DataFrame(cr),use_container_width=True,hide_index=True)
    elif result is not None and not run_phase3:
        st.info("Enable **Spatial Zones + AI Decision Support** in the sidebar and re-run.")
    else: st.info("Run the analysis first.")

# ── Tab 3: Habitations ────────────────────────────────────────────────────────
with tab_habs:
    if sih_result is not None and sih_result.exposure_results:
        exp_list=sih_result.exposure_results; rel_map={r.hab_id:r for r in sih_result.relocation_results}
        vuln_map={v.hab_id:v for v in sih_result.vulnerability_results}; cap_map={c.hab_id:c for c in sih_result.capacity_results}
        st.subheader("Habitation Exposure Summary")
        ca,cb,cc,cd=st.columns(4)
        ca.metric("Total Habitations",len(exp_list)); cb.metric("In Red Zone",sum(1 for e in exp_list if e.is_in_red_zone),delta_color="inverse")
        cc.metric("CRITICAL Priority",sum(1 for r in sih_result.relocation_results if r.priority_class=="CRITICAL"),delta_color="inverse")
        cd.metric("HIGH Priority",sum(1 for r in sih_result.relocation_results if r.priority_class=="HIGH"),delta_color="inverse")
        fc1,fc2=st.columns(2)
        hf=fc1.multiselect("Filter Hazard",["High","Medium","Low","Water"],default=["High","Medium","Low","Water"])
        pf=fc2.multiselect("Filter Priority",["CRITICAL","HIGH","MEDIUM","LOW"],default=["CRITICAL","HIGH","MEDIUM","LOW"])
        rows=[]
        for e in exp_list:
            rel=rel_map.get(e.hab_id); vuln=vuln_map.get(e.hab_id); cap=cap_map.get(e.hab_id)
            pri=rel.priority_class if rel else "—"
            if e.hazard_class not in hf: continue
            if rel and rel.priority_class not in pf: continue
            pop=f"{e.population_exposed:,}" if e.population_source=="osm_tag" and e.population_exposed else "UNKNOWN"
            zone=full_result.habitation_zones.get(e.hab_id,"—") if full_result else "—"
            rows.append({"Name":e.name or "Unnamed","Type":e.hab_type,"Zone":zone,"Hazard":e.hazard_class,
                "Score":round(e.hazard_score,1),"Priority":f"{PRIORITY_EMOJI.get(pri,'')} {pri}",
                "Reloc.Score":round(rel.relocation_score,3) if rel else "—","Vuln.":vuln.vulnerability_class if vuln else "—",
                "Capacity":cap.capacity_status if cap else "—","Population":pop,"Lat":round(e.lat,5),"Lon":round(e.lon,5)})
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        else: st.info("No habitations match filters.")
        st.subheader("Habitation Detail")
        hnames=[f"{PRIORITY_EMOJI.get(rel_map.get(e.hab_id,type('o',(),{'priority_class':'—'})()).priority_class,'')} {e.name or 'Unnamed'} ({e.hab_type})" for e in exp_list]
        si=st.selectbox("Select habitation",range(len(exp_list)),format_func=lambda i:hnames[i])
        if si is not None:
            exp=exp_list[si]; rel=rel_map.get(exp.hab_id); vuln=vuln_map.get(exp.hab_id); cap=cap_map.get(exp.hab_id)
            zone=full_result.habitation_zones.get(exp.hab_id,"—") if full_result else "—"
            cL,cR=st.columns(2)
            with cL:
                st.markdown(f"#### {exp.name or 'Unnamed Habitation'}")
                st.markdown(f"**Type:** {exp.hab_type} | **Spatial Zone:** {ZONE_EMOJI.get(zone,'')} **{zone}**")
                hce={"High":"\U0001f534","Medium":"\U0001f7e1","Low":"\U0001f7e2","Water":"\U0001f535"}
                st.markdown(f"**Hazard:** {hce.get(exp.hazard_class,'')} {exp.hazard_class} ({exp.hazard_score:.1f}/100)")
                pop_s=(f"{exp.population_exposed:,} *(OSM)*" if exp.population_source=="osm_tag" and exp.population_exposed else "**UNKNOWN** *(not in OSM)*")
                st.markdown(f"**Population:** {pop_s}  \n**In Red Zone:** {'Yes \U0001f534' if exp.is_in_red_zone else 'No'}")
            with cR:
                if rel:
                    st.markdown(f"#### Priority: {PRIORITY_EMOJI.get(rel.priority_class,'')} **{rel.priority_class}**")
                    st.progress(rel.relocation_score,text=f"Relocation score: {rel.relocation_score:.3f}")
                    st.info(rel.recommended_action)
            if vuln:
                st.markdown("##### Vulnerability")
                vcols=st.columns(len(vuln.component_scores))
                for i,(comp,sc) in enumerate(vuln.component_scores.items()):
                    vcols[i].metric(comp.replace("_"," ").title(),f"{sc:.2f}",delta=f"wt {vuln.component_weights.get(comp,0):.0%}",delta_color="off")
                st.markdown(f"**Overall:** {vuln.vulnerability_class} ({vuln.vulnerability_score:.3f})")
            if cap:
                st.markdown("##### Carrying Capacity")
                cc1,cc2,cc3,cc4=st.columns(4)
                cc1.metric("Status",cap.capacity_status); cc2.metric("Safe Area",f"{cap.safe_area_km2:.2f} km\u00b2")
                cc3.metric("Nearest Road",f"{cap.nearest_road_km:.1f} km" if cap.nearest_road_km>=0 else "Not found")
                cc4.metric("Healthcare",f"{cap.nearest_healthcare_km:.1f} km" if cap.nearest_healthcare_km>=0 else "Not found")
                st.caption(cap.notes)
            if rel and rel.explanation:
                with st.expander("\U0001f4cb Full Explainability"): st.code(rel.explanation,language=None)
            # AI decision panel inline
            if full_result and full_result.agent_decisions:
                dec=full_result.get_decision_for(exp.hab_id)
                if dec:
                    st.markdown("---"); st.markdown("#### \U0001f916 AI Decision Support")
                    st.caption("AI-ASSISTED — not an official evacuation order.")
                    if not dec.ai_assisted: st.caption(f"\u2699\ufe0f {dec.fallback_reason}")
                    st.markdown(f"**Summary:** {dec.summary}")
                    st.info(f"**Recommended action:** {dec.recommended_action}")
                    if dec.candidate_areas:
                        with st.expander(f"\U0001f7e9 {len(dec.candidate_areas)} Relocation Candidate Area(s)"):
                            for rank,cand in enumerate(dec.candidate_areas[:3],1):
                                st.markdown(f"**Candidate {rank}** — score {cand.candidate_score:.3f} | {cand.distance_km:.1f} km | {cand.area_km2:.2f} km\u00b2")
                                st.markdown(cand.notes)
    elif result is not None and not run_sih: st.info("Enable Habitation Analysis in the sidebar.")
    else: st.info("Run the analysis first.")

# ── Tab 4: Relocation Priority ────────────────────────────────────────────────
with tab_reloc:
    if sih_result is not None and sih_result.relocation_results:
        rel_list=sih_result.relocation_results; exp_map_r={e.hab_id:e for e in sih_result.exposure_results}; cap_map_r={c.hab_id:c for c in sih_result.capacity_results}
        st.subheader("Relocation Priority Overview")
        pc={}
        for r in rel_list: pc[r.priority_class]=pc.get(r.priority_class,0)+1
        c1,c2,c3,c4=st.columns(4)
        c1.metric("\U0001f6a8 CRITICAL",pc.get("CRITICAL",0)); c2.metric("\u26a0\ufe0f HIGH",pc.get("HIGH",0)); c3.metric("\U0001f514 MEDIUM",pc.get("MEDIUM",0)); c4.metric("\u2705 LOW",pc.get("LOW",0))
        fig,ax=plt.subplots(figsize=(6,3))
        labs=[k for k in ["CRITICAL","HIGH","MEDIUM","LOW"] if k in pc]; vals=[pc[k] for k in labs]
        ax.barh(labs[::-1],vals[::-1],color=[PCOLORS[k] for k in labs[::-1]])
        ax.set_xlabel("Number of Habitations"); ax.set_title("Relocation Priority Distribution")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        for i,v in enumerate(vals[::-1]): ax.text(v+0.05,i,str(v),va="center",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.subheader("Priority Ranked Habitations")
        st.caption("Declared weights: Hazard 35% | Vulnerability 30% | Capacity Stress 20% | Exposure 15%")
        tr=[]
        for rank,rel in enumerate(rel_list,1):
            exp=exp_map_r.get(rel.hab_id); cap=cap_map_r.get(rel.hab_id)
            zone=full_result.habitation_zones.get(rel.hab_id,"—") if full_result else "—"
            pop=f"{rel.population_exposed:,}" if rel.population_source=="osm_tag" and rel.population_exposed else "UNKNOWN"
            nc=len(full_result.relocation_candidates.get(rel.hab_id,[])) if full_result else 0
            tr.append({"Rank":rank,"Priority":rel.priority_class,"Zone":zone,"Name":rel.name or "Unnamed","Score":round(rel.relocation_score,3),
                "Hazard":round(rel.hazard_score,1),"Vuln.":round(rel.vulnerability_score,3),"Cap.Stress":round(1-rel.capacity_score,3),
                "Population":pop,"Safe Area":f"{cap.safe_area_km2:.2f} km\u00b2" if cap else "—","Candidates":nc})
        if tr:
            df_rel=pd.DataFrame(tr)
            def _hp(row):
                bg={"CRITICAL":"background-color:#fdecea","HIGH":"background-color:#fef5e7","MEDIUM":"background-color:#fef9e7","LOW":"background-color:#eafaf1"}
                return [bg.get(row["Priority"],"")] * len(row)
            st.dataframe(df_rel.style.apply(_hp,axis=1),use_container_width=True,hide_index=True)
        with st.expander("\u2139\ufe0f Formula"):
            st.code("relocation_score =\n    0.35 x hazard_score/100\n  + 0.30 x vulnerability_score\n  + 0.20 x (1 - capacity_score)\n  + 0.15 x exposure_component",language=None)
    elif result is not None and not run_sih: st.info("Enable Habitation Analysis in the sidebar.")
    else: st.info("Run the analysis first.")

# ── Tab 5: AI Decision Support ────────────────────────────────────────────────
with tab_ai:
    if full_result is not None and full_result.agent_decisions:
        decisions=full_result.agent_decisions
        st.subheader("\U0001f916 PRAVAAH AI-Assisted Decision Support")
        if _has_llm: st.success(f"AI model active ({_llm_provider.title()})",icon="\U0001f916")
        else:
            st.info("\u2139\ufe0f **Rule-based mode** — LLM not configured. "
                "Set `PRAVAAH_LLM_PROVIDER` + API key to enable AI explanations. "
                "All structured analysis and recommendations are fully functional.",icon="\u2139\ufe0f")
        st.caption("AI-ASSISTED DECISION SUPPORT — recommendations only. Not an official evacuation order. All scores from the deterministic PRAVAAH pipeline.")
        ov=[]
        for hid,dec in decisions.items():
            ov.append({"Name":dec.hab_name,"Priority":dec.priority_class,"Zone":dec.spatial_zone,
                "Score":round(dec.relocation_score,3),"Candidates":len(dec.candidate_areas),"AI":"\u2705" if dec.ai_assisted else "Rule-based"})
        if ov:
            def _hd(row):
                bg={"CRITICAL":"background-color:#fdecea","HIGH":"background-color:#fef5e7","MEDIUM":"background-color:#fef9e7","LOW":"background-color:#eafaf1"}
                return [bg.get(row["Priority"],"")] * len(row)
            st.dataframe(pd.DataFrame(ov).style.apply(_hd,axis=1),use_container_width=True,hide_index=True)
        st.subheader("Decision Detail")
        _dec_list=sorted(decisions.items(),key=lambda kv:{"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}.get(kv[1].priority_class,9))
        _dec_names=[f"{dec.priority_class} — {dec.hab_name}" for _,dec in _dec_list]
        sdi=st.selectbox("Select habitation for AI detail",range(len(_dec_list)),format_func=lambda i:_dec_names[i])
        if sdi is not None:
            _,dec=_dec_list[sdi]
            st.markdown(f"### {PRIORITY_EMOJI.get(dec.priority_class,'')} {dec.hab_name} — {ZONE_EMOJI.get(dec.spatial_zone,'')} {dec.spatial_zone}")
            st.progress(dec.relocation_score,text=f"Relocation score: {dec.relocation_score:.3f}")
            if not dec.ai_assisted: st.caption(f"\u2699\ufe0f {dec.fallback_reason}")
            st.info(f"**Recommended action:** {dec.recommended_action}",icon="\U0001f4cb")
            if dec.evidence:
                st.markdown("#### Agent Evidence")
                ev_tabs=st.tabs([f"\U0001f50d {e.agent_name}" for e in dec.evidence])
                for etab,ev in zip(ev_tabs,dec.evidence):
                    with etab:
                        sc={"CRITICAL":"\U0001f534","HIGH":"\U0001f7e0","MEDIUM":"\U0001f7e1","LOW":"\U0001f7e2"}
                        st.markdown(f"**Severity:** {sc.get(ev.severity,'')} {ev.severity}")
                        st.markdown(ev.summary)
                        if ev.key_factors: st.markdown("**Key factors:**\n" + "\n".join(f"  \u2022 {f}" for f in ev.key_factors))
                        if ev.metrics:
                            with st.expander("\U0001f4ca Raw metrics"): st.json(ev.metrics)
                        st.caption("AI-assisted" if ev.ai_assisted else "Rule-based fallback")
            if dec.candidate_areas:
                st.markdown("#### \U0001f7e9 Relocation Candidate Areas")
                st.caption("Potential lower-risk areas for evaluation. Decision-support recommendations only.")
                rl=["A","B","C","D","E"]
                for rank,cand in enumerate(dec.candidate_areas[:5]):
                    lbl=rl[rank] if rank<len(rl) else str(rank+1)
                    with st.expander(f"Candidate {lbl} — Score {cand.candidate_score:.3f} | {cand.distance_km:.1f} km | {cand.area_km2:.2f} km\u00b2",expanded=(rank==0)):
                        cc1,cc2,cc3,cc4=st.columns(4)
                        cc1.metric("Score",f"{cand.candidate_score:.3f}"); cc2.metric("Distance",f"{cand.distance_km:.1f} km")
                        cc3.metric("Safe Area",f"{cand.area_km2:.2f} km\u00b2"); cc4.metric("Hazard",f"{cand.mean_hazard_score:.0f}/100")
                        st.markdown(cand.notes); st.caption(f"Data: {cand.data_provenance}")
                if dec.top_candidate_reason: st.success(f"**Top recommendation:** {dec.top_candidate_reason}")
    elif full_result is not None and not full_result.agent_decisions: st.info("No agent decisions. Check habitation analysis ran successfully.")
    elif result is not None and not run_phase3: st.info("Enable Spatial Zones + AI in the sidebar.")
    else: st.info("Run the analysis first.")

# ── Tab 6: Risk Statistics ────────────────────────────────────────────────────
with tab_stats:
    if result is not None:
        dist=result.risk_distribution; cm_s={"Low":"#2ecc71","Medium":"#f39c12","High":"#e74c3c","Water":"#3498db"}
        dnw={k:v for k,v in dist.items() if k!="Water"}; ls=list(dnw.keys()); cs=list(dnw.values()); bs=[cm_s.get(l,"#999") for l in ls]
        if cs:
            ca,cb=st.columns(2)
            with ca:
                fig,ax=plt.subplots(); ax.bar(ls,cs,color=bs); ax.set_ylabel("Cell Count"); ax.set_title("Risk Class Distribution (excl. Water)")
                st.pyplot(fig); plt.close(fig)
            with cb:
                fig2,ax2=plt.subplots(); ax2.pie(cs,labels=ls,colors=bs,autopct="%1.1f%%",startangle=90); ax2.set_title("Risk Class Share")
                st.pyplot(fig2); plt.close(fig2)
        nw=dist.get("Water",0)
        if nw: st.info(f"\u2139\ufe0f {nw} cells classified as permanent water bodies.")
    else: st.info("Run the analysis first.")

# ── Tab 7: Factor Weights ─────────────────────────────────────────────────────
with tab_factors:
    if result is not None:
        fi=result.analysis_result.feature_importances
        fig,ax=plt.subplots(figsize=(8,5)); feats=list(fi.keys()); imps=list(fi.values())
        ax.barh(feats[::-1],imps[::-1],color="#3498db"); ax.set_xlabel("Importance"); ax.set_title("Hazard Susceptibility Factor Weights")
        st.pyplot(fig); plt.close(fig); st.caption(result.analysis_result.validation_note)
        ar=result.analysis_result
        if ar.method in ("random_forest","ensemble") and ar.mean_cv_auc is not None:
            st.markdown("**Cross-Validation Results (5-fold)**")
            cols=st.columns(5)
            cols[0].metric("AUC-ROC",f"{ar.mean_cv_auc:.3f}"); cols[1].metric("F1",f"{ar.mean_cv_f1:.3f}")
            cols[2].metric("Accuracy",f"{ar.mean_cv_accuracy:.3f}" if ar.mean_cv_accuracy else "—")
            cols[3].metric("Precision",f"{ar.mean_cv_precision:.3f}" if ar.mean_cv_precision else "—")
            cols[4].metric("Recall",f"{ar.mean_cv_recall:.3f}" if ar.mean_cv_recall else "—")
            if ar.cv_auc_scores:
                nf=len(ar.cv_auc_scores)
                st.dataframe(pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(nf)],
                    "AUC":[f"{v:.3f}" for v in ar.cv_auc_scores],"F1":[f"{v:.3f}" for v in (ar.cv_f1_scores or [])],
                    "Acc":[f"{v:.3f}" for v in (ar.cv_accuracy_scores or ["—"]*nf)],
                    "Prec":[f"{v:.3f}" for v in (ar.cv_precision_scores or ["—"]*nf)],
                    "Rec":[f"{v:.3f}" for v in (ar.cv_recall_scores or ["—"]*nf)]}),use_container_width=True,hide_index=True)
    else: st.info("Run the analysis first.")

# ── Tab 8: Data & Export ──────────────────────────────────────────────────────
with tab_data:
    if result is not None:
        st.subheader("Grid Data")
        disp_cols=["cell_id","centroid_lat","centroid_lon"]+FEATURE_COLUMNS+["risk_score","risk_class"]
        avail=[c for c in disp_cols if c in result.scored_grid.columns]
        df=result.scored_grid[avail].copy()
        rfd=st.multiselect("Filter by Risk Class",["Low","Medium","High","Water"],default=["Low","Medium","High","Water"])
        dff=df[df["risk_class"].isin(rfd)]; st.dataframe(dff,use_container_width=True)
        d1,d2,d3,d4=st.columns(4)
        with d1:
            cb2=io.StringIO(); dff.to_csv(cb2,index=False)
            st.download_button("\u2b07\ufe0f Hazard Grid CSV",cb2.getvalue(),file_name="PRAVAAH_Hazard_Grid.csv",mime="text/csv")
        with d2:
            st.download_button("\u2b07\ufe0f Hazard GeoJSON",result.scored_grid.to_json(),file_name="PRAVAAH_Hazard_Map.geojson",mime="application/json")
        if sih_result and sih_result.relocation_results:
            with d3:
                re=[]
                for rel in sih_result.relocation_results:
                    exp=next((e for e in sih_result.exposure_results if e.hab_id==rel.hab_id),None)
                    vuln=next((v for v in sih_result.vulnerability_results if v.hab_id==rel.hab_id),None)
                    cap=next((c for c in sih_result.capacity_results if c.hab_id==rel.hab_id),None)
                    zone=full_result.habitation_zones.get(rel.hab_id,"") if full_result else ""
                    re.append({"hab_id":rel.hab_id,"name":rel.name,"spatial_zone":zone,"type":exp.hab_type if exp else "",
                        "lat":exp.lat if exp else "","lon":exp.lon if exp else "","hazard_score":rel.hazard_score,
                        "hazard_class":exp.hazard_class if exp else "","vulnerability_score":rel.vulnerability_score,
                        "vulnerability_class":vuln.vulnerability_class if vuln else "","capacity_score":rel.capacity_score,
                        "capacity_status":cap.capacity_status if cap else "","safe_area_km2":cap.safe_area_km2 if cap else "",
                        "nearest_road_km":cap.nearest_road_km if cap else "","nearest_healthcare_km":cap.nearest_healthcare_km if cap else "",
                        "relocation_score":rel.relocation_score,"priority_class":rel.priority_class,
                        "population_exposed":rel.population_exposed or "UNKNOWN","population_source":rel.population_source,
                        "recommended_action":rel.recommended_action,"contributing_factors":" | ".join(rel.contributing_factors),
                        "num_candidates":len(full_result.relocation_candidates.get(rel.hab_id,[])) if full_result else 0})
                hc=io.StringIO(); pd.DataFrame(re).to_csv(hc,index=False)
                st.download_button("\u2b07\ufe0f Relocation Priority CSV",hc.getvalue(),file_name="PRAVAAH_Relocation_Priority.csv",mime="text/csv")
        with d4:
            if st.button("\U0001f4c4 Generate PDF Report"):
                with st.spinner("Generating PRAVAAH report\u2026"):
                    try:
                        import tempfile
                        from flood_risk_zonation.visualization.pdf_report import export_pdf_report
                        area_name=(area_name_input.strip() or preset.get("area_name","") or f"Lat {result.bounding_box.min_lat:.3f}\u2013{result.bounding_box.max_lat:.3f}")
                        with tempfile.TemporaryDirectory() as td:
                            pp=export_pdf_report(result,Path(td)/"PRAVAAH_Risk_Assessment.pdf",area_name=area_name,data_tier=result.data_tier,sih_result=sih_result)
                            pb=Path(pp).read_bytes()
                        st.download_button("\u2b07\ufe0f Download PDF",pb,file_name="PRAVAAH_Risk_Assessment.pdf",mime="application/pdf")
                        st.success("Report generated.")
                    except Exception as e: st.error(f"Report failed: {e}")
    else: st.info("Run the analysis first.")

# ── Tab 9: Methodology ────────────────────────────────────────────────────────
with tab_meth:
    st.markdown("""
## PRAVAAH — System Methodology

### Architecture
```
Geospatial Data → ML Hazard Engine → Hazard Classification
       ↓                                      ↓
  Habitation Overlay                  RED/YELLOW/GREEN Zones
       ↓                                      ↓
  Exposure Analysis              Relocation Candidate Discovery
       ↓                                      ↓
  Vulnerability                    Agentic Decision Support
       ↓                              (5 bounded agents)
  Carrying Capacity                          ↓
       ↓                            Explainable Decision
  Relocation Priority                        ↓
       └─────────────────────────→ Authority Dashboard
```

### Spatial Zone System
| Zone | Meaning | Derived from |
|------|---------|-------------|
| 🟥 RED | Primary hazard zone | ML risk_class = High |
| 🟨 YELLOW | Secondary attention zone | 8-neighbour adjacent to RED, or risk_class = Medium |
| 🟩 GREEN | Lower-risk / potential safe area | Not RED, YELLOW, or Water |
| 🔵 WATER | Permanent water body | risk_class = Water |

**Key principle:** The underlying `risk_score` and `risk_class` are never modified.
`spatial_zone` is an additional operational column.

A YELLOW cell is NOT a confirmed flood cell.
A GREEN area is NOT a guaranteed safe zone — it requires further evaluation.

### Agentic Architecture
PRAVAAH uses a bounded, cost-conscious agentic layer:

| Agent | Responsibility | When invoked |
|-------|---------------|--------------|
| HazardAnalyst | Interprets hazard metrics, identifies dominant factors | All priorities |
| ExposureAnalyst | Interprets population exposure | MEDIUM+ priorities |
| VulnerabilityAnalyst | Explains vulnerability component scores | HIGH+ priorities |
| CapacityAnalyst | Identifies capacity constraints | HIGH+ priorities |
| RelocationPlanner | Synthesises evidence, recommends candidates | HIGH+ priorities |

**Cost controls:** LLM only invoked for HIGH/CRITICAL habitations. MAX 5 agents per habitation.
MAX 50 habitations per run. Deterministic fallback when LLM unavailable.

**LLM setup (optional):** Set `PRAVAAH_LLM_PROVIDER=openai` or `anthropic` + API key.

### Relocation Candidate Scoring
```
candidate_score =
    0.30 × (1 - hazard_score/100)
  + 0.25 × norm(area_km2)
  + 0.20 × (1 - norm(distance_km))
  + 0.15 × road_accessibility_factor
  + 0.10 × healthcare_accessibility_factor
  − 0.05 × population_pressure_factor
```
Candidates drawn from GREEN zones only. Decision-support recommendations, not official designations.

### Scientific Honesty
- Population is **UNKNOWN** when not in OSM — never fabricated
- ML metrics are CV on WSI pseudo-labels, not validated against real flood events
- Shelter capacity is **unavailable** — no curated national dataset
- Road/healthcare distances are straight-line (Euclidean), not routed
- All weights declared in source code and visible here
- AI recommendations are clearly labelled as decision-support, not official orders
""")
