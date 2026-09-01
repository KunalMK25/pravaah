<div align="center">
  <img src="assets/pravaah-ai-logo.png" alt="PRAVAAH-AI — Predictive Risk & Vulnerability Assessment for At-Risk Habitations" width="600"/>
</div>

# PRAVAAH-AI

## Predictive Risk & Vulnerability Assessment for At-Risk Habitations

[![Tests](https://github.com/KunalMK25/pravaah/actions/workflows/test.yml/badge.svg)](https://github.com/KunalMK25/pravaah/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-808%20passing-brightgreen)

**PRAVAAH-AI** is an AI-powered geospatial decision-support system that identifies hazard-based red zones, evaluates vulnerable habitations, assesses exposure and carrying-capacity stress, and prioritises intervention or relocation using explainable spatial intelligence.

The system answers a single operational question:

> **"Which vulnerable habitations need attention, and why?"**

---

## Problem

Communities in flood-prone areas face a recurring challenge: hazard maps exist, but they do not answer which specific settlements are at risk, how many people are exposed, whether the area has the capacity to absorb displaced residents, or which habitations require immediate relocation rather than routine monitoring.

PRAVAAH-AI bridges that gap by layering habitation intelligence on top of a geospatial hazard foundation, producing ranked, explainable relocation priorities for every identified settlement in the study area.

---

## Solution

PRAVAAH-AI builds upon a geospatial risk-assessment foundation and extends it with habitation-level exposure, vulnerability, carrying-capacity, relocation intelligence, live weather integration, short-term forecasting, historical validation, and bounded agentic decision support.

```
Environmental / Geospatial Data
        ↓
Hazard Analysis Engine (ML + GIS)
  Grid · Features · WSI + Ensemble RF · Risk Score · RED/YELLOW/GREEN Zones
        ↓
Habitation Intelligence Layer
  Settlement Ingestion (OSM)
        ↓
  Exposure · Vulnerability · Carrying Capacity · Relocation Priority
        ↓
Dynamic Intelligence
  Live Weather → Dynamic Adjustment → 24–72h Forecast
        ↓
  Historical Flood Validation (independent metrics)
        ↓
  What-If Scenario Simulation
        ↓
  SHAP ML Explainability
        ↓
Agentic Decision Support (9 bounded agents)
        ↓
Explainable Recommendations → Authority Dashboard → PDF Report
```

---

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **Hazard mapping** | Grid-based susceptibility (WSI + Random Forest + Ensemble, 10 geospatial factors) |
| **RED/YELLOW/GREEN zones** | Operational spatial zones derived from ML output via 8-neighbour adjacency |
| **Settlement ingestion** | Live OpenStreetMap habitation nodes (city/town/village/hamlet/suburb) |
| **Exposure analysis** | Each settlement overlaid against the hazard grid; population from OSM tags |
| **Vulnerability scoring** | Transparent 7-component weighted index (all weights declared) |
| **Carrying capacity** | Safe area, road accessibility, healthcare proximity (OSM-sourced) |
| **Relocation priority** | Formula-driven priority with 4 documented guardrails |
| **Relocation candidates** | GREEN-zone discovery and scoring for HIGH/CRITICAL habitations |
| **Live weather** | OpenWeatherMap integration with dynamic risk adjustment |
| **Short-term forecast** | 24/48/72h rainfall-adjusted risk projection (ESTIMATE) |
| **Historical validation** | Independent P/R/F1/IoU against documented flood extents |
| **What-if scenarios** | Rainfall / drainage / population parameter simulation (SIMULATION) |
| **SHAP explainability** | TreeSHAP per-cell and global importance for the ML hazard model |
| **Agentic AI** | 9 bounded agents: Hazard, Exposure, Vulnerability, Capacity, Relocation, Weather, Forecast, Scenario, Validation |
| **Authority dashboard** | 12-tab Streamlit DSS for decision-makers |
| **PDF report** | Full risk assessment with habitation rankings and relocation priorities |

---

## Architecture

### Hazard Analysis

| Dataset | Source | Provenance |
|---------|--------|------------|
| Elevation | NASA SRTM GeoTIFF + OpenTopoData API | `real` / `api` / `synthetic` |
| Water bodies | OSM Overpass API | `osm_overpass` / `osm_cache` |
| Rainfall | Synthetic (GPM-structured) | `synthetic` |
| Habitations | OSM Overpass API — place nodes (city/town/village/hamlet/suburb/neighbourhood/locality/isolated_dwelling/farm/allotments), residential building footprints (building=house/residential/apartments/detached/semidetached_house/terrace/bungalow/dormitory/hut/cabin), and residential landuse polygons | `osm_overpass` / `osm_overpass_buildings` / `osm_cache` / `fallback` |
| Roads / Healthcare | OSM Overpass API | `osm_overpass` |
| Weather | OpenWeatherMap | `LIVE` / `CACHED` / `UNAVAILABLE` |
| Population | OSM `population` tag only | `osm_tag` / `UNKNOWN` |

### Water Preservation & Proximity Boost

- **Water Preservation:** All cells classified as WATER (permanent water bodies) in the baseline hazard grid are preserved exactly as WATER through all scenario simulations. Scenario rainfall and drainage adjustments do not reclassify permanent water; only LAND cells are subject to reclassification based on updated risk scores.
- **Proximity Boost:** Habitation cells receive a proximity-based boost to their risk scores during baseline grid generation, prioritising hazard identification near concentrations of known habitations. The boost is applied uniformly to the baseline; scenarios preserve the boosted baseline (via water preservation) and apply relative adjustments only to LAND cells.

### Spatial Zone System

| Zone | Definition | Source |
|------|-----------|--------|
| 🟥 **RED** | Primary hazard zone | ML `risk_class = "High"` |
| 🟨 **YELLOW** | Secondary attention zone | 8-neighbour adjacent to RED, or Medium class |
| 🟩 **GREEN** | Lower-risk / potential safe area | Not RED, YELLOW, or Water |
| 🔵 **WATER** | Permanent water body | `risk_class = "Water"` |

### Relocation Priority Formula

```
relocation_score =
    0.35 × hazard_score / 100
  + 0.30 × vulnerability_score
  + 0.20 × (1 − capacity_score)
  + 0.15 × exposure_component
```

Guardrails: capacity CRITICAL +0.10 bonus · coastal → HIGH escalates to CRITICAL · unknown population cap at HIGH.

### Agentic Architecture

9 bounded agents, each consuming structured PRAVAAH-AI pipeline outputs:

| Agent | Responsibility | Invoked for |
|-------|---------------|------------|
| HazardAnalyst | Interprets hazard metrics and zone | All priorities |
| ExposureAnalyst | Population exposure with honest provenance | MEDIUM+ |
| VulnerabilityAnalyst | Component breakdown | HIGH+ |
| CapacityAnalyst | Capacity constraints | HIGH+ |
| RelocationPlanner | Synthesises evidence, recommends candidates | HIGH+ |
| WeatherAnalyst | Live rainfall and dynamic adjustment | When weather available |
| ForecastAnalyst | 24–72h risk projections (ESTIMATE label) | When forecast available |
| ScenarioAnalyst | What-if deltas (SIMULATION label) | When scenario run |
| ValidationAnalyst | Historical validation metrics | When validation run |

LLM invoked for HIGH/CRITICAL only. Deterministic fallback always active. Set `PRAVAAH_LLM_PROVIDER=groq` + `GROQ_API_KEY` to enable AI explanations (recommended). OpenAI and Anthropic are also supported.

---

## Scientific Honesty

PRAVAAH-AI is explicit about data quality at every stage:

- Population is **UNKNOWN** when absent from OSM — never fabricated
- ML metrics are cross-validation on WSI pseudo-labels — **not validated against real flood events** (historical validation is independent)
- Forecasts are always labelled **ESTIMATE** — not deterministic predictions
- Scenarios are always labelled **SIMULATION** — baseline is never overwritten
- Relocation candidates are **decision-support recommendations** — not officially designated sites
- GREEN zones are **lower-risk areas** — not guaranteed safe
- Shelter capacity is **unavailable** — no curated national dataset integrated
- Road/healthcare distances are calculated using **routing-aware network distances where available** (networkx-based shortest-path on OSM road graph), with explicit **fallback to straight-line (Haversine) distance** when network data is unavailable or disconnected. Provenance is tracked as `"network_routing"` or `"straight_line_fallback"` in capacity notes.

---

## Preset Regions

PRAVAAH-AI includes 6 preset study areas with pre-cached offline data (5 regions) and live-only capability (Puri):

| Region | Geography | Bounding Box | Offline Data | Typical Use |
|--------|-----------|--------------|--------------|-------------|
| **Gottigere, Bangalore** | Inland urban, South India | 12.84°–12.91°N, 77.55°–77.62°E | ✓ Available | Urban flood risk, drainage adequacy |
| **Chennai Marina** | Coastal urban, Tamil Nadu | 12.98°–13.05°N, 80.24°–80.31°E | ✓ Available | Cyclone / storm surge exposure |
| **Dal Lake, Srinagar** | Alpine lake, Jammu & Kashmir | 34.07°–34.14°N, 74.83°–74.90°E | ✓ Available | High-altitude hydrology, overflow risk |
| **Puri, Odisha** | Cyclone coast, Odisha | 19.77°–19.84°N, 85.80°–85.87°E | ✗ Live-only | Tropical cyclone hazard zone |
| **Indian Hilly Region** | Foothills / mountainous, Himalayas | 27.45°–27.55°N, 88.45°–88.55°E | ✓ Available | Landslide, cloud-burst, steep terrain |
| **Indian Ocean** | Open water reference zone | 9.95°–10.05°N, 71.95°–72.05°E | ✓ Available | Water-only validation, reference hazard |

**Note:** Offline-available regions can be analysed without internet connectivity; Puri requires live OSM/weather data. Custom bounding boxes can be defined in the app.

---

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

### Configure environment (optional)

```bash
cp .env.example .env
# Edit .env — add OPENWEATHER_API_KEY, PRAVAAH_LLM_PROVIDER, etc.
```

### Run tests

```bash
python -m pytest tests/ -q --no-cov
# Expected: 808 passed, 0 failed
```

---

## Deployment

### Streamlit Community Cloud (recommended)

1. Push to `https://github.com/KunalMK25/pravaah`
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Repository: `KunalMK25/pravaah` · Branch: `main` · File: `app.py`
4. Add secrets (Settings → Secrets):
   ```toml
   OPENWEATHER_API_KEY = "your_openweather_key"   # optional — live weather

   # Recommended LLM provider (Groq — free tier available)
   PRAVAAH_LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_..."                        # required when provider=groq
   PRAVAAH_GROQ_MODEL = "llama3-8b-8192"           # optional — default shown

   # Alternative providers (only one block needed)
   # PRAVAAH_LLM_PROVIDER = "openai"
   # OPENAI_API_KEY = "sk-..."

   # PRAVAAH_LLM_PROVIDER = "anthropic"
   # ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Deploy — `packages.txt` and `requirements.txt` are read automatically

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment instructions.

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENWEATHER_API_KEY` | Optional | Live weather data (OpenWeatherMap) |
| `PRAVAAH_LLM_PROVIDER` | Optional | `groq` / `openai` / `anthropic` / `none` (default: `none`) |
| `GROQ_API_KEY` | If provider=groq | Groq API key (recommended provider) |
| `PRAVAAH_GROQ_MODEL` | Optional | Groq model override (default: `llama3-8b-8192`) |
| `OPENAI_API_KEY` | If provider=openai | OpenAI API key |
| `ANTHROPIC_API_KEY` | If provider=anthropic | Anthropic API key |

---

## Outputs

| File | Description |
|------|-------------|
| `PRAVAAH-AI_Hazard_Grid.csv` | Per-cell hazard scores and features |
| `PRAVAAH-AI_Hazard_Map.geojson` | Full hazard grid as GeoJSON |
| `PRAVAAH-AI_Relocation_Priority.csv` | Ranked habitation assessment |
| `PRAVAAH-AI_Risk_Assessment.pdf` | Full structured report |

---

## Demo Flow (3–5 minutes)

1. Open app → select **Gottigere, Bangalore** → click **Run Analysis**
2. **Hazard Map** tab → RED/YELLOW/GREEN zones + habitation markers
3. **Spatial Zones** tab → zone distribution + candidate areas
4. **Habitations** tab → select a CRITICAL habitation → full breakdown
5. **Relocation Priority** tab → ranked table + priority chart
6. **Weather** tab → live conditions + dynamic adjustment
7. **Forecast** tab → 24/48/72h projected zones
8. **Scenarios** tab → run "+30% Rainfall" → compare with baseline
9. **AI Support** tab → agent evidence + relocation candidates
10. **Explainability** tab → SHAP explanation for a high-risk cell
11. **Data & Export** → download CSV + generate PDF report

---

## Known Limitations

1. **Drainage infrastructure:** Municipal hydraulic capacity measurements are unavailable; the system uses mapped OSM drainage infrastructure (drain, canal, stream, and river linestrings) as a **spatial proxy**, not a hydraulic model. Dense nearby infrastructure → higher proxy score; absent infrastructure → synthetic fallback (population-inverse heuristic). Proximity scores are inverse-distance weighted; drain/canal density is clipped intersection length within 2× grid cell radius. Provenance is reported in `data_provenance["drainage"]`: `"osm_proxy"` or `"synthetic_fallback"`.
2. **Population data:** Population from OSM is sparse for many Indian settlements — many show UNKNOWN. No authoritative raster (WorldPop, Census) is currently integrated.
3. **Road and healthcare accessibility distances:** Routing-aware network distances are calculated where OSM road network data is available and the network is ≤500 nodes. The system builds an undirected road network graph from OSM Overpass queries and computes shortest-path distances between habitations and road/healthcare targets. For networks exceeding 500 nodes, or when network data is unavailable or disconnected, the system gracefully falls back to straight-line (Haversine) geodesic distance to maintain performance and reliability. Provenance is explicitly tracked in capacity assessment notes: `"network_routing"` or `"straight_line_fallback"`. Limitations: road network may be incomplete in remote areas, healthcare facilities are matched to nearest road node (not routed individually), and routing uses a simplified undirected graph (no one-way or turn restrictions).
4. **Weather data and forecasts:** Weather data is sourced from OpenWeatherMap; the flood-risk forecast is a rainfall-adjusted estimate — **not a physically-based hydrological simulation**. Forecasts are always labelled **ESTIMATE**.
5. **Historical validation:** Historical flood extent validation uses coarse approximate polygons (~1 km accuracy) and is independent of the ML training signal.
6. **Habitation detection:** Combines three OSM layers — named settlement place nodes, residential building footprints (strict allowlist: house/residential/apartments/detached/semidetached_house/terrace/bungalow/dormitory/hut/cabin), and residential landuse polygons — fetched in a single cached Overpass request. Building centroids within ~50 m of a place node are deduplicated. Non-residential buildings (industrial, warehouse, commercial, schools, hospitals, etc.) are excluded. Per-record provenance is `"osm_overpass"` / `"osm_building"` / `"osm_landuse"` / `"fallback"`. Coverage remains dependent on the completeness of OpenStreetMap mapping in the selected study area.
7. **Sentinel-1 satellite integration:** Architecture hook exists, not yet automated.
8. **Water preservation:** Permanent water cells (WATER class) are preserved exactly through all scenario simulations; only LAND cells are subject to reclassification.
9. **Scenario simulation:** All scenarios are labelled **SIMULATION** — baseline grid is never overwritten. Scenarios explore rainfall and drainage sensitivity only; other factors (elevation, habitations, etc.) remain constant.

---

## Future Scope

- Authoritative population raster integration (WorldPop / Census ward data)
- Road routing for realistic evacuation time estimates
- Sentinel-1 flood extent comparison layer
- Full hydrological forecast model (once temporal training data available)
- What-if scenarios integrated into agent decision workflow

---

## Project Structure

```
pravaah/
├── app.py                              # Streamlit entry point (12-tab DSS)
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System packages for cloud deployment
├── .env.example                        # Environment variable template
├── DEPLOYMENT.md                       # Deployment guide
├── assets/
│   └── pravaah_ai_logo.svg             # Product logo
├── flood_risk_zonation/                # Core analysis package
│   ├── pipeline.py                     # Phase 1 hazard engine
│   ├── sih_pipeline.py                 # Phase 2+3 habitation intelligence
│   ├── models.py                       # All typed dataclasses
│   ├── config.py                       # BoundingBox, PipelineConfig
│   ├── health.py                       # Application health check
│   ├── habitation/                     # OSM settlement ingestion
│   ├── exposure/                       # Spatial exposure analysis
│   ├── vulnerability/                  # Weighted vulnerability scorer
│   ├── capacity/                       # Carrying capacity assessment
│   ├── relocation/                     # Priority + candidate discovery
│   ├── spatial_zones/                  # RED/YELLOW/GREEN classifier
│   ├── weather/                        # Live weather client
│   ├── forecast/                       # 24–72h risk projection
│   ├── validation/                     # Historical flood validation
│   ├── scenarios/                      # What-if simulation engine
│   ├── explainability/                 # SHAP ML explainability
│   ├── agents/                         # Agentic decision support
│   ├── features/                       # Geospatial feature extraction
│   ├── scoring/                        # Susceptibility models
│   └── visualization/                  # Map, PDF, export
└── tests/
    ├── unit/                           # 40+ unit test files
    ├── integration/                    # Integration tests
    └── property/                       # Hypothesis property tests
```

---

*PRAVAAH-AI — Predictive Risk & Vulnerability Assessment for At-Risk Habitations*
