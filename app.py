"""
PRAVAAH-AI — Predictive Risk & Vulnerability Assessment for At-Risk Habitations
Full intelligence system: hazard + spatial zones + habitation intelligence
+ live weather + forecast + historical validation + scenarios + SHAP + agentic AI.
Run with: streamlit run app.py
"""
from __future__ import annotations
import io, logging, os
from dotenv import load_dotenv

load_dotenv()

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

st.set_page_config(page_title="PRAVAAH-AI — Hazard & Habitation Intelligence",
                   page_icon="assets/pravaah-ai-icon.png", layout="wide")

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("\U0001f30a PRAVAAH-AI")
st.sidebar.caption("Predictive Risk & Vulnerability Assessment for At-Risk Habitations (PRAVAAH-AI)")
st.sidebar.markdown("---")

PRESET_REGIONS = {
    "Gottigere, Bangalore": {"min_lon":77.55,"min_lat":12.84,"max_lon":77.62,"max_lat":12.91,"area_name":"Gottigere, Bangalore","offline_key":"Bangalore (Gottigere)","active_flood_override":False},
    "Chennai Marina (Coastal)": {"min_lon":80.24,"min_lat":12.98,"max_lon":80.31,"max_lat":13.05,"area_name":"Chennai Marina, Chennai","offline_key":"Chennai Marina (Coastal)","active_flood_override":False},
    "Dal Lake, Srinagar": {"min_lon":74.83,"min_lat":34.07,"max_lon":74.90,"max_lat":34.14,"area_name":"Dal Lake, Srinagar","offline_key":"Dal Lake, Srinagar","active_flood_override":False},
    "Puri, Odisha (Cyclone Coast)": {"min_lon":85.80,"min_lat":19.77,"max_lon":85.87,"max_lat":19.84,"area_name":"Puri, Odisha","offline_key":None,"active_flood_override":False},
    "Indian Hilly Region (Sikkim)": {"min_lon":87.50,"min_lat":27.40,"max_lon":88.57,"max_lat":28.47,"area_name":"Sikkim Foothills (Himalayan Region)","offline_key":"Indian Hilly Region","active_flood_override":False},
    "Nepal Recent Flood-Affected Area": {"min_lon":85.40,"min_lat":28.10,"max_lon":86.47,"max_lat":29.17,"area_name":"Rasuwa District, Bhote Koshi River (Nepal)","offline_key":"Nepal Flood Area","active_flood_override":True},
    "Indian Ocean Open Water": {"min_lon":71.50,"min_lat":9.50,"max_lon":72.57,"max_lat":10.57,"area_name":"Arabian Sea Open Water (Demonstration)","offline_key":"Indian Ocean","active_flood_override":False},
    "\u270f\ufe0f Custom Region": {"min_lon":77.55,"min_lat":12.84,"max_lon":77.62,"max_lat":12.91,"area_name":"","offline_key":None,"active_flood_override":False},
}
st.sidebar.subheader("Region Selection")
selected_preset = st.sidebar.selectbox("Select Region", list(PRESET_REGIONS.keys()), index=0)
preset = PRESET_REGIONS[selected_preset]
is_custom = "Custom" in selected_preset

offline_key = preset["offline_key"]
has_offline = offline_key is not None and offline_key in DEMO_REGIONS
use_offline = st.sidebar.checkbox("\U0001f4e6 Use offline sample data", value=False) if has_offline else False
offline_region: DemoRegion | None = None
if use_offline and has_offline:
    offline_region = DEMO_REGIONS[offline_key]
    st.sidebar.info(f"\U0001f4e6 Offline mode \u2014 **{selected_preset}**")

st.sidebar.markdown("---")
area_name_input = st.sidebar.text_input("Area Name (for report)", value=preset["area_name"] if not is_custom else "", disabled=not is_custom)
col1, col2 = st.sidebar.columns(2)
min_lon = col1.number_input("Min Lon", value=float(preset["min_lon"]), min_value=-180.0, max_value=180.0, step=0.01, disabled=not is_custom or use_offline)
min_lat = col2.number_input("Min Lat", value=float(preset["min_lat"]), min_value=-90.0, max_value=90.0, step=0.01, disabled=not is_custom or use_offline)
max_lon = col1.number_input("Max Lon", value=float(preset["max_lon"]), min_value=-180.0, max_value=180.0, step=0.01, disabled=not is_custom or use_offline)
max_lat = col2.number_input("Max Lat", value=float(preset["max_lat"]), min_value=-90.0, max_value=90.0, step=0.01, disabled=not is_custom or use_offline)

st.sidebar.subheader("Grid Resolution")
cell_size = {"250m":250,"500m":500,"1000m":1000}[st.sidebar.selectbox("Cell Size", ["250m","500m","1000m"], index=1)]
st.sidebar.subheader("Hazard Model")
selected_model_type = {"Ensemble (WSI + RF)":"ensemble","Weighted Index (WSI)":"weighted_susceptibility","Random Forest (ML)":"random_forest"}[
    st.sidebar.radio("Susceptibility Model", ["Ensemble (WSI + RF)","Weighted Index (WSI)","Random Forest (ML)"], index=0)]
st.sidebar.subheader("Thresholds")
low_threshold = st.sidebar.slider("Low/Medium boundary", 10.0, 49.0, 33.0, 1.0)
medium_threshold = st.sidebar.slider("Medium/High boundary", 51.0, 90.0, 66.0, 1.0)
st.sidebar.subheader("Satellite Data (Optional)")
sentinel1_geotiff_path = st.sidebar.text_input("🛰️ Sentinel-1 GeoTIFF (flood mask)", value="", placeholder="e.g., data/sentinel1_flood_mask.tif")
sentinel1_geojson_path = st.sidebar.text_input("🛰️ Sentinel-1 GeoJSON (polygons)", value="", placeholder="e.g., data/sentinel1_flood.geojson")
st.sidebar.caption("Leave empty to skip Sentinel-1 satellite data (optional)")
st.sidebar.subheader("Population Data Providers (Phase 1B)")
use_worldpop = st.sidebar.checkbox("🌍 WorldPop (gridded raster)", value=True)
use_osm = st.sidebar.checkbox("🗺️ OpenStreetMap (habitation tags)", value=True)
use_synthetic = st.sidebar.checkbox("🔄 Synthetic Fallback (if all fail)", value=True)
st.sidebar.caption("Population estimates feed into habitation exposure analysis")
st.sidebar.subheader("Intelligence Layers")
run_sih = st.sidebar.checkbox("\U0001f3d8\ufe0f Habitation Analysis", value=True)
run_phase3 = st.sidebar.checkbox("\U0001f5fa\ufe0f Spatial Zones + Candidates", value=True)
run_agents = st.sidebar.checkbox("\U0001f916 AI Decision Support", value=True)
run_weather = st.sidebar.checkbox("\U0001f327\ufe0f Live Weather", value=True)
run_forecast = st.sidebar.checkbox("\U0001f4c8 Forecast (24\u201372h)", value=True)
run_validation = st.sidebar.checkbox("\U0001f4da Historical Validation", value=False)

st.sidebar.subheader("🚨 Emergency Response")
enable_emergency_facilities = st.sidebar.checkbox("Emergency Facilities", value=True)
show_hospitals = show_shelters = False
if enable_emergency_facilities:
    show_hospitals = st.sidebar.checkbox("🏥 Hospitals", value=True)
    show_shelters = st.sidebar.checkbox("🏠 Shelters", value=True)
enable_evacuation_routes = st.sidebar.checkbox("Hazard-Aware Evacuation Routes", value=False)
evacuation_priority_filter = []
if enable_evacuation_routes:
    evacuation_priority_filter = st.sidebar.multiselect(
        "Route for priorities", ["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH"]
    )

_llm_provider = os.environ.get("PRAVAAH_LLM_PROVIDER","none").lower()
_has_llm = ((_llm_provider=="openai" and bool(os.environ.get("OPENAI_API_KEY"))) or
            (_llm_provider=="anthropic" and bool(os.environ.get("ANTHROPIC_API_KEY"))) or
            (_llm_provider=="groq" and bool(os.environ.get("GROQ_API_KEY"))))
_weather_key = bool(os.environ.get("OPENWEATHER_API_KEY","").strip())
if _has_llm:
    st.sidebar.markdown(f'<p style="color:#2ecc71;font-size:11px;text-align:center">\U0001f916 AI: {_llm_provider.title()}</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p style="color:#95a5a6;font-size:11px;text-align:center">\U0001f916 AI: Rule-based</p>', unsafe_allow_html=True)
if _weather_key:
    st.sidebar.markdown('<p style="color:#2ecc71;font-size:11px;text-align:center">\U0001f327\ufe0f Weather: API key set</p>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p style="color:#95a5a6;font-size:11px;text-align:center">\U0001f327\ufe0f Weather: No key (unavailable)</p>', unsafe_allow_html=True)

run_button = st.sidebar.button("\U0001f680 Run Analysis", type="primary", use_container_width=True)
try:
    _load_land_mask()
    st.sidebar.markdown('<p style="color:#2ecc71;font-size:11px;margin-top:4px;text-align:center">\U0001f7e2 Land Mask: Active</p>', unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown('<p style="color:#e74c3c;font-size:11px;margin-top:4px;text-align:center">\U0001f534 Land Mask: Error</p>', unsafe_allow_html=True)
if not use_offline:
    from math import cos, radians as _rad
    _cl = (float(min_lat)+float(max_lat))/2.0
    _h = (float(max_lat)-float(min_lat))*111.32
    _w = (float(max_lon)-float(min_lon))*111.32*cos(_rad(_cl))
    if _w>0 and _h>0:
        from flood_risk_zonation.config import BBOX_MIN_SIDE_KM, BBOX_MAX_SIDE_KM
        _ok = _w>=BBOX_MIN_SIDE_KM and _h>=BBOX_MIN_SIDE_KM and _w<=BBOX_MAX_SIDE_KM and _h<=BBOX_MAX_SIDE_KM
        st.sidebar.caption(f"{'✅' if _ok else '⚠️'} {_w:.1f} km × {_h:.1f} km")

# ── Main panel ───────────────────────────────────────────────────────────────
st.title("\U0001f30a PRAVAAH-AI")
st.markdown("**Predictive Risk & Vulnerability Assessment for At-Risk Habitations**  \n"
            "*Hazard ML · Spatial Zones · Habitation Intelligence · Live Weather · Forecasting · What-If Scenarios · Agentic AI*")
st.markdown("---")

for _k,_v in [("result",None),("sih_result",None),("full_result",None),
              ("weather_data",None),("forecast_result",None),
              ("validation_result",None),("scenario_results",{}),
              ("fallback_warning",False)]:
    if _k not in st.session_state: st.session_state[_k]=_v

PCOLORS={"CRITICAL":"#c0392b","HIGH":"#e67e22","MEDIUM":"#f1c40f","LOW":"#2ecc71"}
PEMOJI={"CRITICAL":"\U0001f6a8","HIGH":"⚠️","MEDIUM":"\U0001f514","LOW":"✅"}
ZEMOJI={"RED":"\U0001f7e5","YELLOW":"\U0001f7e8","GREEN":"\U0001f7e9","WATER":"\U0001f535"}

if run_button:
    st.session_state.update({"fallback_warning":False,"sih_result":None,"full_result":None,
                             "weather_data":None,"forecast_result":None,"validation_result":None,
                             "scenario_results":{}})
    try:
        bbox = offline_region.bbox if offline_region else BoundingBox(float(min_lon),float(min_lat),float(max_lon),float(max_lat))
        config = PipelineConfig(cell_size_meters=float(cell_size),model_type=selected_model_type,
            rf_n_estimators=50,cv_folds=3,low_threshold=float(low_threshold),medium_threshold=float(medium_threshold),
            use_cache=False,allow_network=not use_offline,
            population_config={
                "worldpop": {"enabled": use_worldpop},
                "osm": {"enabled": use_osm},
                "synthetic": {"enabled": use_synthetic},
            })
        # Add optional Sentinel-1 paths to config (will be None if not provided)
        config.sentinel1_geotiff_path = sentinel1_geotiff_path if sentinel1_geotiff_path.strip() else None
        config.sentinel1_geojson_path = sentinel1_geojson_path if sentinel1_geojson_path.strip() else None
        if not use_offline:
            err = validate_bbox_size(bbox)
            if err: st.error(f"📐 {err}"); st.stop()
        pipeline = FloodRiskPipeline(config)
        with st.status("Running PRAVAAH-AI analysis…", expanded=True) as status:
            p = st.write
            
            # Active Flood Verification Gate (MUST execute BEFORE pipeline)
            active_flood_result = None
            if not use_offline:
                p("Checking for active flooding...")
                try:
                    from flood_risk_zonation.verification.active_flood_check import check_active_flooding
                    area_name = area_name_input if area_name_input.strip() else f"Lat {bbox.min_lat:.2f} to {bbox.max_lat:.2f}"
                    # Get active_flood_override from preset if available
                    override_enabled = preset.get("active_flood_override", False)
                    active_flood_result = check_active_flooding(
                        location_name=area_name,
                        lat=(bbox.min_lat + bbox.max_lat) / 2.0,
                        lon=(bbox.min_lon + bbox.max_lon) / 2.0,
                        active_flood_override=override_enabled,
                    )
                    if active_flood_result.is_active_flood_gate():
                        p(f"🚨 **ACTIVE FLOODING DETECTED**: {active_flood_result.summary}")
                        st.session_state.active_flood_result = active_flood_result
                        status.update(label="Active flooding detected - pipeline skipped", state="complete")
                        # Stop execution to prevent expensive pipeline
                        st.stop()
                    else:
                        p(f"✓ No active flooding - proceeding with analysis")
                except Exception as afe:
                    p(f"⚠️ Verification error: {afe}")
                    p("Proceeding with normal analysis (safety fallback)")
                    active_flood_result = None
            
            if use_offline and offline_region:
                p("📦 Loading offline data…")
                from flood_risk_zonation.ingest.sample_data import get_demo_elevation,get_demo_rainfall,get_demo_water_bodies
                from flood_risk_zonation.ingest.population import load_population
                elevation=get_demo_elevation(offline_region,resolution_m=30.0)
                rainfall=get_demo_rainfall(offline_region)
                water_bodies=get_demo_water_bodies(offline_region)
                population=load_population(bbox,data_dir=str(config.cache_dir))
                result=pipeline.run_from_ingested_data(bounding_box=bbox,elevation=elevation,rainfall=rainfall,
                    water_bodies=water_bodies,population=population,
                    provenance={"elevation":"offline_sample","rainfall":"offline_sample",
                                "water_bodies":"offline_sample","population":population.source},
                    data_tier=3,progress_callback=p)
            else:
                result=pipeline.run(bbox,progress_callback=p)
                if result.data_provenance.get("water_bodies")=="fallback": st.session_state.fallback_warning=True

            # Phase 2: Habitation intelligence
            sih_result=None; full_result=None
            if run_sih:
                p("\U0001f3d8\ufe0f Habitation intelligence…")
                sih_pipe=SIHPipeline(config=config,allow_network=not use_offline)
                sih_result=sih_pipe.run_sih_stages(result,bbox,progress_callback=p)
                n_h=len(sih_result.habitation_dataset.habitations)
                p(f"\U0001f3d8\ufe0f {n_h} settlements · {len(sih_result.red_zone_habitations)} red zone · {len(sih_result.critical_habitations)} critical")
                # Phase 3: Zones + candidates + agents
                if run_phase3:
                    p("\U0001f5fa\ufe0f Spatial zones + candidates…")
                    full_result=sih_pipe.run_phase3(sih_result,progress_callback=p,run_agents=run_agents)
                    p(f"\U0001f5fa\ufe0f \U0001f7e5{full_result.red_zone_count} \U0001f7e8{full_result.yellow_zone_count} \U0001f7e9{full_result.green_zone_count} · {len(full_result.agent_decisions)} AI decisions")

            # Live weather
            weather_data=None
            if run_weather and not use_offline:
                p("\U0001f327\ufe0f Fetching live weather…")
                try:
                    from flood_risk_zonation.weather.client import fetch_weather
                    clat,clon=bbox.center
                    weather_data=fetch_weather(clat,clon)
                    p(f"\U0001f327\ufe0f Weather: {weather_data.data_status} · adjustment={weather_data.dynamic_risk_adjustment:.2f}")
                except Exception as we:
                    p(f"\U0001f327\ufe0f Weather unavailable: {we}")

            # Forecast
            forecast_result=None
            if run_forecast and weather_data is not None:
                p("\U0001f4c8 Generating 24–72h forecast…")
                try:
                    from flood_risk_zonation.forecast.engine import generate_forecast
                    forecast_result=generate_forecast(result,weather_data)
                    max_zone=max((h.spatial_zone for h in forecast_result.horizons if h.spatial_zone!="WATER"),
                                 key=lambda z:{"RED":3,"YELLOW":2,"GREEN":1,"WATER":0}.get(z,0),default="GREEN")
                    p(f"\U0001f4c8 Forecast: max projected zone={max_zone}")
                except Exception as fe:
                    p(f"\U0001f4c8 Forecast failed: {fe}")

            # Historical validation
            validation_result=None
            if run_validation:
                p("\U0001f4da Historical validation…")
                try:
                    from flood_risk_zonation.validation.events import run_validation as _rv
                    validation_result=_rv(result)
                    p(f"\U0001f4da Validation: {validation_result.data_status}")
                except Exception as vae:
                    p(f"\U0001f4da Validation failed: {vae}")

            status.update(label="✅ PRAVAAH-AI complete", state="complete", expanded=False)

        st.session_state.update({"result":result,"sih_result":sih_result,"full_result":full_result,
                                 "weather_data":weather_data,"forecast_result":forecast_result,
                                 "validation_result":validation_result})
        # Active Flood Emergency Banner
        if st.session_state.get('active_flood_result'):
            afr = st.session_state.active_flood_result
            if afr.is_active_flood_gate():
                st.markdown(
                    f'<div style="background-color:#c0392b; padding:20px; border-radius:8px; '
                    f'border:3px solid #a93226; color:white; text-align:center; '
                    f'box-shadow: 0 0 20px rgba(192,57,43,0.8)">'
                    f'<h2 style="margin:0; color:white; font-weight:bold;">?? ACTIVE FLOODING DETECTED ??</h2>'
                    f'<p style="margin:10px 0 0 0; font-size:16px;">{afr.summary}</p>'
                    f'<p style="margin:8px 0 0 0; font-size:13px; opacity:0.9;">'
                    f'<strong>Location:</strong> {afr.location_name}<br/>'
                    f'<strong>Verified:</strong> {afr.verification_timestamp.strftime("%Y-%m-%d %H:%M UTC")}<br/>'
                    f'<strong>Confidence:</strong> {afr.confidence:.0%}'
                    f'</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown('---')
                if afr.primary_evidence:
                    with st.expander('?? Evidence Details', expanded=False):
                        st.write(f'**Source:** {afr.primary_evidence.source}')
                        st.write(f'**Title:** {afr.primary_evidence.title}')
                        st.write(f'**Evidence:** {afr.primary_evidence.evidence_text}')
                        if afr.primary_evidence.timestamp:
                            st.write(f'**Published:** {afr.primary_evidence.timestamp.strftime("%Y-%m-%d %H:%M UTC")}')
                st.warning('?? The predictive analysis below is for reference only. Current flooding is an active situation that requires immediate official emergency response.', icon='??')
                st.markdown('---')
        
        # Summary alert
        if sih_result:
            nc=len(sih_result.critical_habitations); nh=sum(1 for r in sih_result.relocation_results if r.priority_class=="HIGH")
            if nc>0: st.error(f"\U0001f6a8 **{nc} CRITICAL** habitation(s) — see Relocation Priority tab.", icon="\U0001f6a8")
            elif nh>0: st.warning(f"⚠️ **{nh} HIGH priority** habitation(s).", icon="⚠️")
            else: st.success(f"✅ {result.cell_count} cells · {result.pipeline_duration_seconds:.1f}s · Tier {result.data_tier} · {len(sih_result.habitation_dataset.habitations)} habitations")
        else: st.success(f"✅ {result.cell_count} cells · {result.pipeline_duration_seconds:.1f}s · Tier {result.data_tier}")
    except FloodRiskError as exc: st.error(f"Analysis error: {exc}")
    except Exception as exc: st.error(f"Unexpected error: {exc}"); logger.exception("Unhandled")

if st.session_state.get("fallback_warning"):
    st.warning("⚠️ OSM Overpass unreachable — water body data unavailable.", icon="⚠️")

result=st.session_state.result; sih_result=st.session_state.sih_result
full_result=st.session_state.full_result; weather_data=st.session_state.weather_data
forecast_result=st.session_state.forecast_result; validation_result=st.session_state.validation_result
scenario_results=st.session_state.scenario_results

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_map,tab_zones,tab_habs,tab_reloc,tab_weather,tab_fc,tab_val,tab_scenarios,tab_ai,tab_xai,tab_data,tab_meth = st.tabs([
    "\U0001f5fa\ufe0f Hazard Map","\U0001f3a8 Spatial Zones","\U0001f3d8\ufe0f Habitations",
    "\U0001f6a8 Relocation","\U0001f327\ufe0f Weather","\U0001f4c8 Forecast",
    "\U0001f4da Validation","\U0001f9ea Scenarios","\U0001f916 AI Support",
    "\U0001f4ca Explainability","\U0001f4cb Data & Export","\U0001f4d6 Methodology"])

# ── Tab 1: Hazard Map ────────────────────────────────────────────────────────
with tab_map:
    if result is not None:
        center=result.bounding_box.center; builder=FloodRiskMapBuilder()
        _model=result.analysis_result.model; _mb=None
        if hasattr(_model,"lower_") and hasattr(_model,"upper_"):
            _mb={f:(_model.lower_[f],_model.upper_[f]) for f in _model.lower_ if f in _model.upper_}
        exp_list=sih_result.exposure_results if sih_result else None
        rel_list=sih_result.relocation_results if sih_result else None
        zg=full_result.zoned_grid if full_result else None
        cands=full_result.relocation_candidates if full_result else None
        
        # Emergency Response: Load facilities and compute evacuation routes
        emergency_facilities = {"hospitals": [], "shelters": []}
        evacuation_routes_list = []
        if enable_emergency_facilities or enable_evacuation_routes:
            try:
                from flood_risk_zonation.capacity.emergency_response import (
                    load_emergency_facilities, compute_evacuation_routes
                )
                from flood_risk_zonation.utils.routing import build_road_graph
                
                # Create BBox from study area
                study_bbox = BoundingBox(
                    min_lon=float(min_lon), min_lat=float(min_lat),
                    max_lon=float(max_lon), max_lat=float(max_lat)
                )
                
                # Load facilities
                if enable_emergency_facilities:
                    emergency_facilities = load_emergency_facilities(
                        bbox=study_bbox, cache_dir=".cache/osm", allow_network=True
                    )
                
                # Compute evacuation routes
                if enable_evacuation_routes and sih_result and evacuation_priority_filter:
                    road_points = []
                    try:
                        road_graph = build_road_graph(road_points) if road_points else None
                    except Exception:
                        road_graph = None
                    
                    evacuation_routes_list = compute_evacuation_routes(
                        sih_result=sih_result,
                        zoned_grid=zg if zg is not None else None,
                        facilities_dict=emergency_facilities,
                        road_graph=road_graph,
                        priority_filter=evacuation_priority_filter,
                    )
            except Exception as e:
                st.warning(f"Emergency response feature error: {e}")
        
        m=builder.build_choropleth_map(result.scored_grid,center=center,zoom_start=11,model_bounds=_mb,
            exposure_results=exp_list,relocation_results=rel_list,
            show_red_zones=(zg is None),zoned_grid=zg,relocation_candidates=cands,show_spatial_zones=(zg is not None),
            show_emergency_facilities=enable_emergency_facilities,
            hospitals=emergency_facilities.get("hospitals"),
            shelters=emergency_facilities.get("shelters"),
            evacuation_routes=evacuation_routes_list if enable_evacuation_routes else None)
        st.components.v1.html(m._repr_html_(),height=630,scrolling=False)
        if zg is not None:
            c1,c2,c3,c4=st.columns(4)
            c1.markdown("\U0001f7e5 **RED** — Primary Hazard"); c2.markdown("\U0001f7e8 **YELLOW** — Secondary Attention")
            c3.markdown("\U0001f7e9 **GREEN** — Lower-Risk / Potential Safe"); c4.markdown("\U0001f535 **WATER**")
            if weather_data and weather_data.data_status != "UNAVAILABLE":
                adj=weather_data.dynamic_risk_adjustment
                if adj>0: st.info(f"\U0001f327\ufe0f **Live weather:** {weather_data.dynamic_risk_reason} (adjustment: {adj:.2f})")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.markdown("\U0001f534 High Risk"); c2.markdown("\U0001f7e1 Medium"); c3.markdown("\U0001f7e2 Low"); c4.markdown("\U0001f535 Water")
        
        # ── Sentinel-1 Satellite Validation Metrics ──
        if result.sentinel1_comparison_metrics is not None:
            s1_metrics = result.sentinel1_comparison_metrics
            if s1_metrics.comparison_status == "COMPUTED":
                st.markdown("---")
                st.subheader("🛰️ Sentinel-1 Satellite Validation")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("IoU (Intersection over Union)", f"{s1_metrics.iou:.3f}" if s1_metrics.iou is not None else "N/A")
                mc2.metric("Precision", f"{s1_metrics.precision:.3f}" if s1_metrics.precision is not None else "N/A")
                mc3.metric("Recall", f"{s1_metrics.recall:.3f}" if s1_metrics.recall is not None else "N/A")
                mc4.metric("F1 Score", f"{s1_metrics.f1_score:.3f}" if s1_metrics.f1_score is not None else "N/A")
                
                # Confusion matrix details
                with st.expander("📊 Validation Details"):
                    vc1, vc2, vc3, vc4 = st.columns(4)
                    vc1.metric("True Positives", s1_metrics.true_positives)
                    vc2.metric("False Positives", s1_metrics.false_positives)
                    vc3.metric("False Negatives", s1_metrics.false_negatives)
                    vc4.metric("True Negatives", s1_metrics.true_negatives)
                    
                    st.markdown("**Coverage & Inundation:**")
                    vv1, vv2, vv3 = st.columns(3)
                    vv1.metric("Satellite Coverage", f"{s1_metrics.coverage_fraction*100:.1f}%")
                    vv2.metric("Sentinel-1 Inundation", f"{s1_metrics.sentinel1_inundation_fraction*100:.1f}%")
                    vv3.metric("Model Inundation", f"{s1_metrics.model_inundation_fraction*100:.1f}%")
                    
                    if s1_metrics.limitations:
                        st.markdown("**Limitations:**")
                        for lim in s1_metrics.limitations:
                            st.caption(f"• {lim}")
            elif s1_metrics.comparison_status == "UNAVAILABLE":
                st.info(f"🛰️ Sentinel-1 data unavailable: {s1_metrics.error_reason or 'No satellite data provided'}")
        elif result.sentinel1_observation is not None:
            obs = result.sentinel1_observation
            st.markdown("---")
            st.subheader("🛰️ Sentinel-1 Satellite Observation")
            s1c1, s1c2, s1c3 = st.columns(3)
            s1c1.metric("Status", obs.observation_status)
            s1c2.metric("Platform", obs.platform)
            s1c3.metric("Confidence", f"{obs.confidence:.2f}")
            if obs.observation_status == "OBSERVED":
                st.info(f"Sentinel-1 observation acquired but metrics not computed (inundation: {obs.inundation_fraction*100:.1f}% if flood_observed={obs.flood_observed})")

    else:
        st.info("Configure parameters in the sidebar and click **Run Analysis**.")

# ── Tab 2: Spatial Zones ─────────────────────────────────────────────────────
with tab_zones:
    if full_result is not None and full_result.zoned_grid is not None:
        zg=full_result.zoned_grid
        st.subheader("RED / YELLOW / GREEN Zone Summary")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("\U0001f7e5 RED",full_result.red_zone_count)
        c2.metric("\U0001f7e8 YELLOW",full_result.yellow_zone_count)
        c3.metric("\U0001f7e9 GREEN",full_result.green_zone_count)
        c4.metric("\U0001f535 WATER",int((zg["spatial_zone"]=="WATER").sum()) if "spatial_zone" in zg.columns else 0)
        with st.expander("ℹ️ Zone Definitions"):
            st.markdown("| Zone | Meaning | Based on |\n|------|---------|----------|\n"
                "| \U0001f7e5 **RED** | Primary hazard zone | ML risk_class = High |\n"
                "| \U0001f7e8 **YELLOW** | Secondary attention | 8-neighbour adjacent to RED or risk_class = Medium |\n"
                "| \U0001f7e9 **GREEN** | Lower-risk / potential safe area | Not RED, YELLOW, or Water |\n"
                "| \U0001f535 **WATER** | Permanent water body | risk_class = Water |\n\n"
                "GREEN ≠ officially safe. YELLOW ≠ confirmed flood. Underlying risk_class/score are preserved.")
        if "spatial_zone" in zg.columns:
            zd=zg["spatial_zone"].value_counts().to_dict()
            zo=[z for z in ["RED","YELLOW","GREEN","WATER"] if z in zd]
            zc={"RED":"#c0392b","YELLOW":"#f39c12","GREEN":"#27ae60","WATER":"#2980b9"}
            fig,ax=plt.subplots(figsize=(7,3))
            ax.bar(zo,[zd[z] for z in zo],color=[zc[z] for z in zo],edgecolor="white",linewidth=0.8)
            ax.set_ylabel("Grid Cells"); ax.set_title("Spatial Zone Distribution")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for i,z in enumerate(zo): ax.text(i,zd[z]+0.3,str(zd[z]),ha="center",va="bottom",fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        if sih_result and sih_result.exposure_results:
            st.subheader("Habitation Zone Assignments")
            rows=[{"Name":e.name or "Unnamed","Zone":full_result.habitation_zones.get(e.hab_id,"?"),
                "Hazard Class":e.hazard_class,"Score":round(e.hazard_score,1),
                "Priority":(next((r.priority_class for r in sih_result.relocation_results if r.hab_id==e.hab_id),"—")),
                "In Red":"Yes" if e.is_in_red_zone else "No"} for e in sih_result.exposure_results]
            if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        if full_result.relocation_candidates:
            st.subheader("Relocation Candidate Areas")
            st.caption("GREEN-zone areas for HIGH/CRITICAL habitations. Decision-support only — not officially designated sites.")
            cr=[]
            for hid,cands in full_result.relocation_candidates.items():
                exp=next((e for e in (sih_result.exposure_results if sih_result else []) if e.hab_id==hid),None)
                for rank,c in enumerate(cands,1):
                    cr.append({"For":exp.name if exp else hid,"Rank":rank,"Score":f"{c.candidate_score:.3f}",
                        "Dist (km)":f"{c.distance_km:.1f}","Area (km²)":f"{c.area_km2:.2f}",
                        "Hazard":f"{c.mean_hazard_score:.0f}/100","Notes":c.notes[:80]})
            if cr: st.dataframe(pd.DataFrame(cr),use_container_width=True,hide_index=True)
    elif result is not None: st.info("Enable **Spatial Zones** in the sidebar.")
    else: st.info("Run the analysis first.")

# ── Tab 3: Habitations ───────────────────────────────────────────────────────
with tab_habs:
    if sih_result is not None and sih_result.exposure_results:
        exp_list=sih_result.exposure_results; rel_map={r.hab_id:r for r in sih_result.relocation_results}
        vuln_map={v.hab_id:v for v in sih_result.vulnerability_results}; cap_map={c.hab_id:c for c in sih_result.capacity_results}
        ca,cb,cc,cd=st.columns(4)
        ca.metric("Total",len(exp_list)); cb.metric("In Red Zone",sum(1 for e in exp_list if e.is_in_red_zone),delta_color="inverse")
        cc.metric("CRITICAL",sum(1 for r in sih_result.relocation_results if r.priority_class=="CRITICAL"),delta_color="inverse")
        cd.metric("HIGH",sum(1 for r in sih_result.relocation_results if r.priority_class=="HIGH"),delta_color="inverse")
        fc1,fc2=st.columns(2)
        hf=fc1.multiselect("Hazard",["High","Medium","Low","Water"],default=["High","Medium","Low","Water"])
        pf=fc2.multiselect("Priority",["CRITICAL","HIGH","MEDIUM","LOW"],default=["CRITICAL","HIGH","MEDIUM","LOW"])
        rows=[]
        for e in exp_list:
            rel=rel_map.get(e.hab_id); vuln=vuln_map.get(e.hab_id); cap=cap_map.get(e.hab_id)
            pri=rel.priority_class if rel else "—"
            if e.hazard_class not in hf: continue
            if rel and rel.priority_class not in pf: continue
            zone=full_result.habitation_zones.get(e.hab_id,"—") if full_result else "—"
            pop=f"{e.population_exposed:,}" if e.population_source=="osm_tag" and e.population_exposed else "UNKNOWN"
            rows.append({"Name":e.name or "Unnamed","Type":e.hab_type,"Zone":zone,"Hazard":e.hazard_class,
                "Score":round(e.hazard_score,1),"Priority":f"{PEMOJI.get(pri,'')} {pri}",
                "Vuln.":vuln.vulnerability_class if vuln else "—","Cap.":cap.capacity_status if cap else "—","Pop.":pop})
        if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        else: st.info("No habitations match filters.")
        st.subheader("Detail")
        hnames=[f"{PEMOJI.get(rel_map.get(e.hab_id,type('o',(),{'priority_class':'—'})()).priority_class,'')} {e.name or 'Unnamed'} ({e.hab_type})" for e in exp_list]
        si=st.selectbox("Select habitation",range(len(exp_list)),format_func=lambda i:hnames[i])
        if si is not None:
            e=exp_list[si]; rel=rel_map.get(e.hab_id); vuln=vuln_map.get(e.hab_id); cap=cap_map.get(e.hab_id)
            zone=full_result.habitation_zones.get(e.hab_id,"—") if full_result else "—"
            cL,cR=st.columns(2)
            with cL:
                st.markdown(f"**{e.name or 'Unnamed'}** — {e.hab_type} | Zone: {ZEMOJI.get(zone,'')} **{zone}**")
                hce={"High":"\U0001f534","Medium":"\U0001f7e1","Low":"\U0001f7e2","Water":"\U0001f535"}
                st.markdown(f"Hazard: {hce.get(e.hazard_class,'')} {e.hazard_class} ({e.hazard_score:.1f}/100)")
                pop_s=f"{e.population_exposed:,} *(OSM)*" if e.population_source=="osm_tag" and e.population_exposed else "**UNKNOWN**"
                red_zone_label = "Yes 🔴" if e.is_in_red_zone else "No"
                st.markdown(f"Population: {pop_s} | Red Zone: {red_zone_label}")
            with cR:
                if rel:
                    st.markdown(f"Priority: {PEMOJI.get(rel.priority_class,'')} **{rel.priority_class}**")
                    st.progress(rel.relocation_score,text=f"Score: {rel.relocation_score:.3f}")
                    st.info(rel.recommended_action)
            if vuln:
                vcols=st.columns(len(vuln.component_scores))
                for i,(comp,sc) in enumerate(vuln.component_scores.items()):
                    vcols[i].metric(comp.replace("_"," ").title(),f"{sc:.2f}",delta=f"{vuln.component_weights.get(comp,0):.0%}",delta_color="off")
                st.markdown(f"Vulnerability: {vuln.vulnerability_class} ({vuln.vulnerability_score:.3f})")
            if cap:
                cc1,cc2,cc3,cc4=st.columns(4)
                cc1.metric("Capacity",cap.capacity_status); cc2.metric("Safe Area",f"{cap.safe_area_km2:.2f} km²")
                cc3.metric("Road",f"{cap.nearest_road_km:.1f} km" if cap.nearest_road_km>=0 else "?")
                cc4.metric("Healthcare",f"{cap.nearest_healthcare_km:.1f} km" if cap.nearest_healthcare_km>=0 else "?")
            if rel and rel.explanation:
                with st.expander("📋 Explainability"): st.code(rel.explanation,language=None)
            if full_result and full_result.agent_decisions:
                dec=full_result.get_decision_for(e.hab_id)
                if dec:
                    st.markdown("---"); st.markdown("#### \U0001f916 AI Decision Support")
                    if not dec.ai_assisted: st.caption(f"⚙️ {dec.fallback_reason}")
                    st.info(f"**{dec.recommended_action}**"); st.markdown(dec.summary)
                    if dec.candidate_areas:
                        with st.expander(f"\U0001f7e9 {len(dec.candidate_areas)} Candidate Area(s)"):
                            for rank,cnd in enumerate(dec.candidate_areas[:3],1):
                                st.markdown(f"**{rank}.** Score {cnd.candidate_score:.3f} · {cnd.distance_km:.1f} km · {cnd.area_km2:.2f} km²")
                                st.markdown(cnd.notes)
    elif result is not None: st.info("Enable Habitation Analysis.")
    else: st.info("Run the analysis first.")

# ── Tab 4: Relocation Priority ───────────────────────────────────────────────
with tab_reloc:
    if sih_result is not None and sih_result.relocation_results:
        rel_list=sih_result.relocation_results; exp_map_r={e.hab_id:e for e in sih_result.exposure_results}; cap_map_r={c.hab_id:c for c in sih_result.capacity_results}
        pc={}
        for r in rel_list: pc[r.priority_class]=pc.get(r.priority_class,0)+1
        c1,c2,c3,c4=st.columns(4)
        c1.metric("\U0001f6a8 CRITICAL",pc.get("CRITICAL",0)); c2.metric("⚠️ HIGH",pc.get("HIGH",0))
        c3.metric("\U0001f514 MEDIUM",pc.get("MEDIUM",0)); c4.metric("✅ LOW",pc.get("LOW",0))
        fig,ax=plt.subplots(figsize=(6,3))
        labs=[k for k in ["CRITICAL","HIGH","MEDIUM","LOW"] if k in pc]; vals=[pc[k] for k in labs]
        ax.barh(labs[::-1],vals[::-1],color=[PCOLORS[k] for k in labs[::-1]])
        ax.set_xlabel("Habitations"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.caption("Formula: Hazard 35% + Vulnerability 30% + Capacity Stress 20% + Exposure 15%")
        tr=[]
        for rank,rel in enumerate(rel_list,1):
            exp=exp_map_r.get(rel.hab_id); cap=cap_map_r.get(rel.hab_id)
            zone=full_result.habitation_zones.get(rel.hab_id,"—") if full_result else "—"
            pop=f"{rel.population_exposed:,}" if rel.population_source=="osm_tag" and rel.population_exposed else "UNKNOWN"
            nc=len(full_result.relocation_candidates.get(rel.hab_id,[])) if full_result else 0
            tr.append({"Rank":rank,"Priority":rel.priority_class,"Zone":zone,"Name":rel.name or "Unnamed",
                "Score":round(rel.relocation_score,3),"Hazard":round(rel.hazard_score,1),
                "Vuln.":round(rel.vulnerability_score,3),"Cap.Stress":round(1-rel.capacity_score,3),
                "Pop.":pop,"Candidates":nc})
        if tr:
            def _hp(row):
                bg={"CRITICAL":"background-color:#fdecea","HIGH":"background-color:#fef5e7","MEDIUM":"background-color:#fef9e7","LOW":"background-color:#eafaf1"}
                return [bg.get(row["Priority"],"")] * len(row)
            st.dataframe(pd.DataFrame(tr).style.apply(_hp,axis=1),use_container_width=True,hide_index=True)
    elif result is not None: st.info("Enable Habitation Analysis.")
    else: st.info("Run the analysis first.")

# ── Tab 5: Live Weather ──────────────────────────────────────────────────────
with tab_weather:
    if weather_data is not None:
        status=weather_data.data_status
        STATUS_COLOR={"LIVE":"#27ae60","CACHED":"#f39c12","FALLBACK":"#e67e22","UNAVAILABLE":"#95a5a6"}
        _sc=STATUS_COLOR.get(status,"#333")
        st.markdown(f"### \U0001f327\ufe0f Live Weather \u2014 <span style='color:{_sc};font-weight:bold'>{status}</span>",
                    unsafe_allow_html=True)
        if weather_data.location_name:
            st.caption(f"Location: **{weather_data.location_name}** | Source: {weather_data.source} | Updated: {weather_data.fetched_at}")
        if status == "UNAVAILABLE":
            st.warning("Weather data unavailable. Set OPENWEATHER_API_KEY environment variable for live weather.")
            st.markdown(f"*Reason:* {weather_data.dynamic_risk_reason}")
        else:
            curr=weather_data.current
            if curr:
                c1,c2,c3,c4=st.columns(4)
                c1.metric("\U0001f327\ufe0f Rainfall (3h)", f"{curr.rainfall_mm:.1f} mm" if curr.rainfall_mm>=0 else "—")
                c2.metric("\U0001f321\ufe0f Temperature", f"{curr.temperature_c:.1f} °C" if curr.temperature_c>-900 else "—")
                c3.metric("\U0001f4a7 Humidity", f"{curr.humidity_pct:.0f}%" if curr.humidity_pct>=0 else "—")
                c4.metric("\U0001f4a8 Wind", f"{curr.wind_speed_ms:.1f} m/s" if curr.wind_speed_ms>=0 else "—")
                st.markdown(f"*Conditions:* {curr.description}")
            adj=weather_data.dynamic_risk_adjustment
            if adj>0:
                st.warning(f"\U0001f327\ufe0f **Dynamic risk adjustment: {adj:.2f}** — {weather_data.dynamic_risk_reason}")
            else:
                st.success("No significant rainfall — baseline hazard conditions apply.")
            if weather_data.forecast:
                st.subheader("Forecast (next 24h, 3h steps)")
                fc_rows=[{"Time":o.timestamp[:16],"Rain (mm)":f"{o.rainfall_mm:.1f}","Temp (°C)":f"{o.temperature_c:.1f}",
                          "Humidity":f"{o.humidity_pct:.0f}%","Conditions":o.description}
                         for o in weather_data.forecast[:8] if o.rainfall_mm>=0]
                if fc_rows: st.dataframe(pd.DataFrame(fc_rows),use_container_width=True,hide_index=True)
    elif result is not None:
        st.info("Enable **Live Weather** and re-run to see current conditions.")
    else: st.info("Run the analysis first.")

# ── Tab 6: Forecast ──────────────────────────────────────────────────────────
with tab_fc:
    if forecast_result is not None:
        st.subheader("\U0001f4c8 Short-term Flood-Risk Forecast")
        st.caption(f"FORECAST / ESTIMATE — {forecast_result.methodology}")
        st.caption(f"Source: {forecast_result.weather_source} | Generated: {forecast_result.forecast_timestamp}")
        ZCOLOR={"RED":"#c0392b","YELLOW":"#f39c12","GREEN":"#27ae60","WATER":"#2980b9"}
        cols=st.columns(len(forecast_result.horizons))
        for col,h in zip(cols,forecast_result.horizons):
            zc=ZCOLOR.get(h.spatial_zone,"#999")
            col.markdown(f"**{h.horizon_h}h**")
            col.markdown(f"<span style='color:{zc};font-weight:bold'>{h.spatial_zone}</span>",unsafe_allow_html=True)
            # Fix: avoid -0.0 display; show "N/A" when forecast rainfall is genuinely unavailable
            _rain_mm = h.forecast_rainfall_mm
            if _rain_mm < 0:
                _rain_str = "N/A"
            else:
                _rain_val = 0.0 if _rain_mm == 0.0 else _rain_mm   # normalise -0.0 → 0.0
                _rain_str = f"{_rain_val:.1f}"
            col.metric("Rain (mm)", _rain_str)
            # Fix: normalise -0.0 risk change to 0.0
            _rc = h.risk_change if h.risk_change != 0.0 else 0.0
            col.metric("\u0394 Risk", f"{_rc:+.1f}")
            col.caption(f"Conf: {h.confidence}")
        fig,ax=plt.subplots(figsize=(8,3))
        hs=[h.horizon_h for h in forecast_result.horizons]
        baselines=[h.baseline_risk_score for h in forecast_result.horizons]
        # Normalise any -0.0 to 0.0 to avoid misleading chart artefacts
        adjusteds=[0.0 if h.adjusted_risk_score == 0.0 else h.adjusted_risk_score for h in forecast_result.horizons]
        ax.plot(hs,baselines,"--",color="#3498db",label="Baseline")
        ax.plot(hs,adjusteds,"-o",color="#e74c3c",label="Forecast-adjusted")
        ax.fill_between(hs,baselines,adjusteds,alpha=0.2,color="#e74c3c")
        ax.set_xlabel("Horizon (hours)"); ax.set_ylabel("Area-mean Risk Score")
        ax.set_title("Forecast Risk vs Baseline (ESTIMATE)"); ax.legend()
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        st.info("⚠️ This is a FORECAST/ESTIMATE — not a deterministic flood prediction. "
                "It combines baseline susceptibility with precipitation forecast data. "
                "Consult local meteorological authorities for operational decisions.")
    elif result is not None:
        st.info("Enable **Forecast** and provide a weather API key to see short-term projections.")
    else: st.info("Run the analysis first.")

# ── Tab 7: Historical Validation ─────────────────────────────────────────────
with tab_val:
    if validation_result is not None:
        st.subheader("\U0001f4da Historical Flood Event Validation")
        st.caption("INDEPENDENT VALIDATION — distinct from ML cross-validation on WSI pseudo-labels.")
        if validation_result.data_status == "NO_EVENTS_AVAILABLE":
            st.info("No historical events in the bundled catalogue overlap with the current bounding box. "
                    "The catalogue currently covers Bangalore (Sep 2022) and Chennai (Nov–Dec 2015). "
                    "Select one of those regions to see validation results.")
        else:
            for ev,m in zip(validation_result.events, validation_result.metrics):
                with st.expander(f"📅 {ev.event_name} — {ev.event_date}"):
                    st.markdown(f"**Region:** {ev.region}  \n**Source:** {ev.source}")
                    if ev.source_url: st.markdown(f"**Reference:** {ev.source_url}")
                    st.markdown(f"**Notes:** {ev.notes}")
                    if m.f1_score >= 0:
                        c1,c2,c3,c4=st.columns(4)
                        c1.metric("Precision",f"{m.precision:.3f}"); c2.metric("Recall",f"{m.recall:.3f}")
                        c3.metric("F1 Score",f"{m.f1_score:.3f}"); c4.metric("IoU",f"{m.iou:.3f}")
                        st.caption(f"Overlap: {m.overlap_count} cells | Predicted high-risk: {m.predicted_high_count} | Observed flood: {m.observed_flood_count}")
                    else:
                        st.warning("Metrics unavailable — one or both cell sets are empty for this bbox.")
            st.caption(validation_result.overall_notes)
    elif result is not None:
        st.info("Enable **Historical Validation** in the sidebar and re-run to compare against documented flood events.")
    else: st.info("Run the analysis first.")

# ── Tab 8: Scenarios ─────────────────────────────────────────────────────────
with tab_scenarios:
    if result is not None:
        st.subheader("\U0001f9ea What-If Scenario Simulation")
        st.caption("SIMULATION — never modifies the baseline. Always labelled as ESTIMATE.")
        from flood_risk_zonation.scenarios.engine import build_preset_scenarios, run_scenario
        from flood_risk_zonation.models import ScenarioParameters
        tab_preset,tab_custom=st.tabs(["Preset Scenarios","Custom Scenario"])
        with tab_preset:
            presets=build_preset_scenarios()
            sel_scenario=st.selectbox("Choose scenario",[p.label for p in presets])
            sel_params=next(p for p in presets if p.label==sel_scenario)
            if st.button("▶️ Run Scenario"):
                with st.spinner(f"Running scenario: {sel_params.label}…"):
                    sr=run_scenario(result,sih_result,sel_params)
                    scenario_results[sel_params.scenario_id]=sr
                    st.session_state.scenario_results=scenario_results
            if sel_params.scenario_id in scenario_results:
                sr=scenario_results[sel_params.scenario_id]
                st.success(f"SIMULATION: {sr.narrative}")
                c1,c2=st.columns(2)
                with c1:
                    st.markdown("**Baseline**")
                    for z,cnt in sr.baseline_zone_counts.items():
                        st.markdown(f"{ZEMOJI.get(z,'')} {z}: {cnt}")
                with c2:
                    st.markdown(f"**Scenario: {sel_params.label}**")
                    for z,cnt in sr.scenario_zone_counts.items():
                        delta=sr.delta_zone_counts.get(z,0)
                        delta_str=f"({delta:+d})" if delta!=0 else ""
                        _dc="red" if (delta>0 and z=="RED") else ("green" if (delta<0 and z=="RED") else "inherit")
                        st.markdown(f"{ZEMOJI.get(z,'')} {z}: {cnt} <span style='color:{_dc}'>{delta_str}</span>",unsafe_allow_html=True)
                if sr.delta_critical!=0:
                    st.metric("CRITICAL habitation change",f"{sr.scenario_critical}",delta=f"{sr.delta_critical:+d}",delta_color="inverse")
                if sr.habitations_escalated:
                    st.warning(f"⚠️ {len(sr.habitations_escalated)} habitation(s) escalated to higher priority under this scenario.")
        with tab_custom:
            st.markdown("Define custom parameters:")
            c1,c2=st.columns(2)
            rain_mult=c1.slider("Rainfall multiplier",0.5,3.0,1.0,0.1)
            extra_mm=c2.number_input("Extra rainfall (mm)",0.0,200.0,0.0,5.0)
            drain_mult=c1.slider("Drainage capacity multiplier",0.1,1.0,1.0,0.05)
            pop_mult=c2.slider("Population multiplier",0.5,2.0,1.0,0.1)
            cust_label=f"Custom: rain×{rain_mult:.1f}+{extra_mm:.0f}mm drain×{drain_mult:.2f}"
            if st.button("▶️ Run Custom Scenario"):
                params=ScenarioParameters("sc_custom",cust_label,
                    rainfall_multiplier=rain_mult,extra_rainfall_mm=extra_mm,
                    drainage_capacity_multiplier=drain_mult,population_multiplier=pop_mult)
                with st.spinner("Running custom scenario…"):
                    sr=run_scenario(result,sih_result,params)
                    scenario_results["sc_custom"]=sr
                    st.session_state.scenario_results=scenario_results
            if "sc_custom" in scenario_results:
                sr=scenario_results["sc_custom"]
                st.success(f"SIMULATION: {sr.narrative}")
    else: st.info("Run the analysis first.")

# ── Tab 9: AI Decision Support ───────────────────────────────────────────────
with tab_ai:
    if full_result is not None and full_result.agent_decisions:
        decisions=full_result.agent_decisions
        st.subheader("\U0001f916 AI-Assisted Decision Support")
        if _has_llm: st.success(f"AI: {_llm_provider.title()}", icon="\U0001f916")
        else: st.info("Rule-based mode — set PRAVAAH_LLM_PROVIDER + key for AI explanations.", icon="ℹ️")
        st.caption("Decision-support recommendations only. Not official evacuation orders.")
        ov=[{"Name":dec.hab_name,"Priority":dec.priority_class,"Zone":dec.spatial_zone,
             "Score":round(dec.relocation_score,3),"Candidates":len(dec.candidate_areas),"AI":"✅" if dec.ai_assisted else "Rule-based"}
            for dec in decisions.values()]
        if ov:
            def _hd(row):
                bg={"CRITICAL":"background-color:#fdecea","HIGH":"background-color:#fef5e7","MEDIUM":"background-color:#fef9e7","LOW":"background-color:#eafaf1"}
                return [bg.get(row["Priority"],"")] * len(row)
            st.dataframe(pd.DataFrame(ov).style.apply(_hd,axis=1),use_container_width=True,hide_index=True)
        dlist=sorted(decisions.items(),key=lambda kv:{"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}.get(kv[1].priority_class,9))
        dnames=[f"{dec.priority_class} — {dec.hab_name}" for _,dec in dlist]
        sdi=st.selectbox("Select for detail",range(len(dlist)),format_func=lambda i:dnames[i])
        if sdi is not None:
            _,dec=dlist[sdi]
            st.markdown(f"### {PEMOJI.get(dec.priority_class,'')} {dec.hab_name} — {ZEMOJI.get(dec.spatial_zone,'')} {dec.spatial_zone}")
            st.progress(dec.relocation_score,text=f"Score: {dec.relocation_score:.3f}")
            if not dec.ai_assisted: st.caption(f"⚙️ {dec.fallback_reason}")
            st.info(dec.recommended_action)
            if dec.evidence:
                ev_tabs=st.tabs([f"🔍 {e.agent_name}" for e in dec.evidence])
                for etab,ev in zip(ev_tabs,dec.evidence):
                    with etab:
                        sc={"CRITICAL":"\U0001f534","HIGH":"\U0001f7e0","MEDIUM":"\U0001f7e1","LOW":"\U0001f7e2"}
                        st.markdown(f"**Severity:** {sc.get(ev.severity,'')} {ev.severity}"); st.markdown(ev.summary)
                        if ev.key_factors: st.markdown("**Key factors:**\n"+"".join(f"\n  • {f}" for f in ev.key_factors))
                        st.caption("AI-assisted" if ev.ai_assisted else "Rule-based fallback")
            if dec.candidate_areas:
                st.markdown("#### \U0001f7e9 Relocation Candidates")
                for rank,cand in enumerate(dec.candidate_areas[:5]):
                    with st.expander(f"Candidate {rank+1} — {cand.candidate_score:.3f} | {cand.distance_km:.1f} km | {cand.area_km2:.2f} km²",expanded=(rank==0)):
                        cc1,cc2,cc3,cc4=st.columns(4)
                        cc1.metric("Score",f"{cand.candidate_score:.3f}"); cc2.metric("Distance",f"{cand.distance_km:.1f} km")
                        cc3.metric("Area",f"{cand.area_km2:.2f} km²"); cc4.metric("Hazard",f"{cand.mean_hazard_score:.0f}/100")
                        st.markdown(cand.notes); st.caption(f"Data: {cand.data_provenance}")
    elif result is not None: st.info("Enable AI Decision Support in the sidebar.")
    else: st.info("Run the analysis first.")

# ── Tab 10: Explainability (SHAP) ────────────────────────────────────────────
with tab_xai:
    if result is not None:
        st.subheader("\U0001f4ca ML Hazard Model Explainability (SHAP)")
        st.caption("SHAP explains the ML hazard component only — not the full relocation decision.")
        grid=result.scored_grid; model=result.analysis_result.model
        high_cells=grid[grid["risk_class"]=="High"]
        if len(high_cells)==0: st.info("No High-risk cells in this analysis.")
        else:
            cell_options=high_cells["cell_id"].tolist()[:20] if "cell_id" in high_cells.columns else [f"cell_{i}" for i in range(min(20,len(high_cells)))]
            sel_cell=st.selectbox("Select a High-risk cell to explain",cell_options)
            if st.button("🔍 Explain this cell"):
                with st.spinner("Computing SHAP explanation…"):
                    try:
                        from flood_risk_zonation.explainability.shap_explainer import explain_cell, explain_global
                        if "cell_id" in grid.columns:
                            cell_row=grid[grid["cell_id"]==sel_cell].iloc[0]
                        else:
                            cell_row=high_cells.iloc[0]
                        feat_names=list(model.feature_names) if hasattr(model,"feature_names") else FEATURE_COLUMNS
                        row=cell_row[[c for c in feat_names if c in cell_row.index]]
                        bg=grid[[c for c in feat_names if c in grid.columns]].sample(min(100,len(grid)),random_state=42)
                        expl=explain_cell(str(sel_cell),row,model,background=bg)
                        st.markdown(f"**{expl.explanation_text}**")
                        st.caption(f"Provenance: {expl.provenance} | Predicted: {expl.predicted_value:.3f} | Base: {expl.base_value:.3f}")
                        if expl.shap_values:
                            sv=expl.shap_values; feat_s=sorted(sv.items(),key=lambda x:abs(x[1]),reverse=True)[:8]
                            labels=[f[0].replace("_"," ").title() for f,_ in feat_s]
                            vals=[v for _,v in feat_s]
                            colors=["#e74c3c" if v>0 else "#2ecc71" for v in vals]
                            fig,ax=plt.subplots(figsize=(7,3.5))
                            ax.barh(labels[::-1],vals[::-1],color=colors[::-1])
                            ax.axvline(0,color="#333",linewidth=0.8)
                            ax.set_xlabel("SHAP Value (positive = increases risk)")
                            ax.set_title(f"SHAP Explanation — {sel_cell}")
                            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                    except Exception as se: st.error(f"SHAP failed: {se}")
        # Global SHAP
        st.subheader("Global Feature Importance (SHAP)")
        if st.button("📊 Compute Global Importance (sample of 200 cells)"):
            with st.spinner("Computing global SHAP…"):
                try:
                    from flood_risk_zonation.explainability.shap_explainer import explain_global
                    gi=explain_global(model,grid,n_sample=200)
                    if gi:
                        gi_s=sorted(gi.items(),key=lambda x:x[1],reverse=True)[:10]
                        fig,ax=plt.subplots(figsize=(7,4))
                        ax.barh([f.replace("_"," ").title() for f,_ in gi_s[::-1]],[v for _,v in gi_s[::-1]],color="#3498db")
                        ax.set_xlabel("Mean |SHAP| (normalised)"); ax.set_title("Global Feature Importance")
                        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                except Exception as ge: st.error(f"Global SHAP failed: {ge}")
    else: st.info("Run the analysis first.")

# ── Tab 11: Data & Export ────────────────────────────────────────────────────
with tab_data:
    if result is not None:
        # Sentinel-1 Satellite Intelligence
        if result.data_provenance.get("sentinel1_status"):
            st.subheader("🛰️ Sentinel-1 Satellite Observation")
            s1_cols = st.columns(4)
            s1_status = result.data_provenance.get("sentinel1_status", "UNKNOWN")
            s1_conf = float(result.data_provenance.get("sentinel1_confidence", 0.0))
            s1_source = result.data_provenance.get("sentinel1_source", "unknown")
            s1_platform = result.data_provenance.get("sentinel1_platform", "Unknown")
            
            with s1_cols[0]:
                st.metric("Status", s1_status, delta=None)
            with s1_cols[1]:
                st.metric("Confidence", f"{s1_conf:.2f}")
            with s1_cols[2]:
                st.metric("Source", s1_source)
            with s1_cols[3]:
                st.metric("Platform", s1_platform)
            
            if s1_status == "OBSERVED":
                st.success("✅ Sentinel-1 flood observation is available for this area.")
            elif s1_status == "UNAVAILABLE":
                st.warning("⚠️ Sentinel-1 data provider unavailable.")
            else:
                st.info("ℹ️ Sentinel-1 observation data not available (UNKNOWN state).")
            
            with st.expander("📋 Sentinel-1 Provenance & Limitations"):
                prov_text = "**Sentinel-1 Observation Metadata:**\n\n"
                for key, value in result.data_provenance.items():
                    if key.startswith("sentinel1_"):
                        prov_text += f"- **{key.replace('sentinel1_', '').title()}**: {value}\n"
                st.markdown(prov_text)
                st.caption("Sentinel-1 satellite observations are provided as an optional intelligence layer. They do NOT modify hazard scores, risk zones, relocation priorities, or any existing Phase 0 analysis. This is informational metadata only.")
        
        st.subheader("Grid Data")
        disp=[c for c in ["cell_id","centroid_lat","centroid_lon"]+FEATURE_COLUMNS+["risk_score","risk_class"] if c in result.scored_grid.columns]
        df=result.scored_grid[disp].copy()
        rfd=st.multiselect("Filter",["Low","Medium","High","Water"],default=["Low","Medium","High","Water"])
        dff=df[df["risk_class"].isin(rfd)]; st.dataframe(dff,use_container_width=True)
        d1,d2,d3,d4=st.columns(4)
        with d1:
            cb2=io.StringIO(); dff.to_csv(cb2,index=False)
            st.download_button("⬇️ Hazard CSV",cb2.getvalue(),file_name="PRAVAAH-AI_Hazard_Grid.csv",mime="text/csv")
        with d2:
            st.download_button("⬇️ Hazard GeoJSON",result.scored_grid.to_json(),file_name="PRAVAAH-AI_Hazard_Map.geojson",mime="application/json")
        if sih_result and sih_result.relocation_results:
            with d3:
                re=[]
                for rel in sih_result.relocation_results:
                    exp=next((e for e in sih_result.exposure_results if e.hab_id==rel.hab_id),None)
                    vuln=next((v for v in sih_result.vulnerability_results if v.hab_id==rel.hab_id),None)
                    cap=next((c for c in sih_result.capacity_results if c.hab_id==rel.hab_id),None)
                    zone=full_result.habitation_zones.get(rel.hab_id,"") if full_result else ""
                    re.append({"hab_id":rel.hab_id,"name":rel.name,"spatial_zone":zone,
                        "hazard_score":rel.hazard_score,"hazard_class":exp.hazard_class if exp else "",
                        "vulnerability_score":rel.vulnerability_score,"vulnerability_class":vuln.vulnerability_class if vuln else "",
                        "capacity_score":rel.capacity_score,"capacity_status":cap.capacity_status if cap else "",
                        "safe_area_km2":cap.safe_area_km2 if cap else "",
                        "relocation_score":rel.relocation_score,"priority_class":rel.priority_class,
                        "population_exposed":rel.population_exposed or "UNKNOWN","population_source":rel.population_source,
                        "recommended_action":rel.recommended_action,"contributing_factors":" | ".join(rel.contributing_factors),
                        "num_candidates":len(full_result.relocation_candidates.get(rel.hab_id,[])) if full_result else 0})
                hc=io.StringIO(); pd.DataFrame(re).to_csv(hc,index=False)
                st.download_button("⬇️ Relocation CSV",hc.getvalue(),file_name="PRAVAAH-AI_Relocation_Priority.csv",mime="text/csv")
        with d4:
            if st.button("📄 Generate PDF"):
                with st.spinner("Generating report…"):
                    try:
                        import tempfile
                        from flood_risk_zonation.visualization.pdf_report import export_pdf_report
                        area_name=area_name_input.strip() or preset.get("area_name","") or f"Lat {result.bounding_box.min_lat:.3f}"
                        with tempfile.TemporaryDirectory() as td:
                            pp=export_pdf_report(result,Path(td)/"PRAVAAH_Risk_Assessment.pdf",area_name=area_name,data_tier=result.data_tier,sih_result=sih_result)
                            pb=Path(pp).read_bytes()
                        st.download_button("⬇️ Download PDF",pb,file_name="PRAVAAH-AI_Risk_Assessment.pdf",mime="application/pdf")
                        st.success("Report generated.")
                    except Exception as e: st.error(f"Report failed: {e}")
    else: st.info("Run the analysis first.")

# ── Tab 12: Methodology ──────────────────────────────────────────────────────
with tab_meth:
    st.markdown("""
## PRAVAAH-AI — Complete System Methodology

### Core Decision Chain
```
Geospatial Data → Hazard ML → RED/YELLOW/GREEN Zones
                                ↓
                    Habitation Overlay → Exposure → Vulnerability → Capacity
                                ↓
                        Relocation Priority → Candidates
                                ↓
            Live Weather → Dynamic Adjustment → Forecast
                                ↓
                    What-If Scenarios → Comparison
                                ↓
              Historical Validation → Independent Metrics
                                ↓
                 Agentic AI → Explainable Recommendations
```

### Hazard Model
10-factor Weighted Susceptibility Index + Random Forest Ensemble.
Scores are relative — not calibrated against real flood events (documented limitation).

### Spatial Zones
RED = ML High class | YELLOW = 8-neighbour adjacent to RED or Medium class | GREEN = lower-risk.
Underlying risk_class/risk_score never modified.

### Weather
OpenWeatherMap (LIVE → CACHED → UNAVAILABLE). Dynamic adjustment = min(1.0, rainfall / 50mm).
NEVER modifies the ML model.

### Forecast
Baseline susceptibility + forecast rainfall boost (max +15 points).
Always labelled FORECAST/ESTIMATE. Not a deterministic prediction.

### Historical Validation
Independent validation against documented flood extents.
Metrics: Precision, Recall, F1, IoU.
Distinct from ML cross-validation on WSI pseudo-labels.

### Scenarios
Isolated from baseline (never overwrites). SIMULATION label always present.
Supports: rainfall ×, drainage capacity ×, population ×.

### SHAP
TreeExplainer for RF/Ensemble. KernelExplainer fallback for WSI.
Explains ML hazard component only — not the full relocation decision.

### Agent Architecture
5 agents: HazardAnalyst, ExposureAnalyst, VulnerabilityAnalyst, CapacityAnalyst, RelocationPlanner.
New: WeatherAnalyst, ForecastAnalyst, ScenarioAnalyst, ValidationAnalyst.
LLM invoked for HIGH/CRITICAL only. Deterministic fallback always active.

### Scientific Honesty
- Population UNKNOWN when absent from OSM — never fabricated
- ML metrics are CV on pseudo-labels — not independent validation
- Historical validation uses independent observations
- Scenarios = SIMULATION, Forecasts = ESTIMATE, AI = RECOMMENDATION
- GREEN ≠ officially safe | Candidates = decision-support, not designated shelters
""")
