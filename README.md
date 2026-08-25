# PRAVAAH

## Predictive Risk & Vulnerability Assessment for At-Risk Habitations

PRAVAAH is an AI-powered geospatial decision-support system that identifies hazard-based
red zones, evaluates vulnerable habitations, assesses exposure and carrying-capacity
stress, and prioritises intervention or relocation using explainable spatial intelligence.

The system answers a single operational question:

> **"Which vulnerable habitations need attention, and why?"**

[![Tests](https://github.com/pravaah/pravaah/actions/workflows/test.yml/badge.svg)](https://github.com/pravaah/pravaah/actions)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem

Communities in flood-prone areas face a recurring challenge: hazard maps exist, but
they do not answer which specific settlements are at risk, how many people are exposed,
whether the area has the capacity to absorb displaced residents, or which habitations
require immediate relocation rather than routine monitoring.

PRAVAAH bridges that gap by layering habitation intelligence on top of a geospatial
hazard foundation, producing ranked, explainable relocation priorities for every
identified settlement in the study area.

---

## Solution

PRAVAAH builds upon an existing geospatial risk-assessment foundation and extends it
with habitation-level exposure, vulnerability, carrying-capacity, and relocation
intelligence.

```
Environmental / Geospatial Data
        ↓
Hazard Analysis Engine
  Grid Generation → Feature Engineering → Susceptibility Model → Risk Score → Red Zone
        ↓
Habitation Intelligence Layer
  Settlement Ingestion (OSM)
        ↓
  Exposure Analysis (spatial overlay)
        ↓
  Vulnerability Assessment (weighted indicators)
        ↓
  Carrying Capacity Assessment (safe area + road + healthcare)
        ↓
  Relocation Priority (transparent formula + guardrails)
        ↓
  Explainability + Authority Dashboard + PDF Report
```

---

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **Hazard mapping** | Grid-based susceptibility scoring using 10 geospatial factors |
| **Red zone identification** | High-risk cells highlighted as explicit red zones |
| **Settlement ingestion** | Habitations sourced live from OpenStreetMap |
| **Exposure analysis** | Each settlement spatially overlaid against the hazard grid |
| **Vulnerability scoring** | Transparent 7-component weighted index, all weights declared |
| **Carrying capacity** | Safe area, road accessibility, and healthcare proximity |
| **Relocation priority** | Formula-driven priority with 4 documented guardrails |
| **Explainability** | Per-habitation "WHY CRITICAL" narrative with component breakdown |
| **Authority dashboard** | 7-tab Streamlit interface for decision-makers |
| **PDF report** | Full risk assessment report with habitation rankings |
| **Data export** | CSV, GeoJSON, and PDF outputs |

---

## Architecture

### Hazard Analysis

| Component | Details |
|-----------|---------|
| Elevation | NASA SRTM GeoTIFF + OpenTopoData API + synthetic fallback |
| Water bodies | OSM Overpass API (cached, retry with exponential back-off, fallback) |
| Rainfall | Synthetic GPM-structured (real GeoTIFF if present) |
| Grid | Configurable cell size: 250 m / 500 m / 1 km |
| Features | 10 per-cell conditioning factors (elevation, TWI, slope, aspect, curvature, rainfall mean & max, distance to water, drainage capacity, population density) |
| Models | Weighted Susceptibility Index (WSI) · Random Forest · Ensemble (default) |
| Output | Risk score [0–100] and class (High / Medium / Low / Water) per grid cell |

### Habitation Intelligence

| Module | Purpose |
|--------|---------|
| `flood_risk_zonation/habitation/ingest.py` | OSM settlement node ingestion |
| `flood_risk_zonation/exposure/analysis.py` | Spatial overlay, per-habitation hazard class |
| `flood_risk_zonation/vulnerability/scorer.py` | Transparent weighted vulnerability index |
| `flood_risk_zonation/capacity/assessment.py` | Safe area, road & healthcare distances |
| `flood_risk_zonation/relocation/priority.py` | Relocation priority formula + guardrails |
| `flood_risk_zonation/sih_pipeline.py` | Orchestrates all habitation intelligence stages |

---

## Methodology

### Vulnerability Assessment

All weights are declared in source code and visible in the UI — no black-box AI.

| Component | Weight | Direction |
|-----------|--------|-----------|
| Hazard severity | 30% | Higher score → more vulnerable |
| Low elevation | 15% | Lower elevation → more vulnerable |
| Water proximity | 15% | Closer to water → more vulnerable |
| Poor drainage | 15% | Lower capacity → more vulnerable |
| Population exposure | 10% | Known exposed population → more vulnerable |
| Road accessibility | 10% | Farther road → more vulnerable |
| Healthcare access | 5% | Farther healthcare → more vulnerable |

Classes: **LOW** (< 0.25) · **MEDIUM** (< 0.50) · **HIGH** (< 0.75) · **CRITICAL** (≥ 0.75)

### Carrying Capacity

| Component | Weight | Measurement |
|-----------|--------|-------------|
| Safe area nearby | 45% | Low-risk land within 5 km |
| Road accessibility | 30% | Distance to nearest primary/secondary road (OSM) |
| Healthcare access | 25% | Distance to nearest hospital or clinic (OSM) |

Status: **ADEQUATE** (≥ 0.60) · **STRESSED** (≥ 0.35) · **CRITICAL** (< 0.35)

### Relocation Priority Formula

```
relocation_score =
    0.35 × hazard_score / 100
  + 0.30 × vulnerability_score
  + 0.20 × (1 − capacity_score)
  + 0.15 × exposure_component
```

**Guardrails (deterministic, documented):**
- Capacity CRITICAL status → +0.10 score bonus
- Coastal / tsunami flag → HIGH escalated to CRITICAL
- Unknown population + non-high hazard → capped at HIGH (precautionary principle)

| Score | Priority | Action |
|-------|----------|--------|
| < 0.25 | LOW | Routine monitoring |
| 0.25 – 0.50 | MEDIUM | Preparedness / alert readiness |
| 0.50 – 0.75 | HIGH | Priority intervention / evacuation planning |
| > 0.75 | CRITICAL | Immediate relocation consideration |

---

## Data Sources

| Dataset | Source | Provenance label |
|---------|--------|-----------------|
| Elevation | NASA SRTM (local GeoTIFF) | `real` |
| Elevation fallback | OpenTopoData API / synthetic | `api` / `synthetic` |
| Water bodies | OSM Overpass API | `osm_overpass` / `osm_cache` |
| Habitations | OSM Overpass API (place= nodes) | `osm_overpass` / `osm_cache` / `fallback` |
| Roads | OSM Overpass API | `osm_overpass` |
| Healthcare | OSM Overpass API | `osm_overpass` |
| Rainfall | Synthetic (GPM-structured) | `synthetic` |
| Population | OSM `population` tag only | `osm_tag` / `UNKNOWN` |
| Drainage | Synthetic | `synthetic` |

### Scientific Honesty

PRAVAAH is explicit about data quality at every stage:

- Population is labelled **UNKNOWN** when absent from OSM — never fabricated
- ML metrics are cross-validation on WSI pseudo-labels, not validated against real flood events
- Shelter capacity is **unavailable** — no curated national dataset is integrated
- Road and healthcare distances are **straight-line** (Euclidean), not routed — stated limitation
- All scoring weights are **declared in source code** and displayed in the UI
- Data tier is always shown: Tier 1 (real) · Tier 2 (partial) · Tier 3 (synthetic)

---

## Explainability

Every habitation receives a full narrative explanation. Example output:

```
WHY THIS HABITATION IS CRITICAL

  Hazard score:          82.4/100 → class High
  Vulnerability score:   0.763 → CRITICAL
  Capacity score:        0.182 → CRITICAL
  Population exposed:    UNKNOWN (not in OSM)
  Safe area nearby:      0.04 km²
  Nearest road:          4.2 km
  Nearest healthcare:    not found

Key factors:
  • High hazard score (82.4/100, class: High)
  • High vulnerability (CRITICAL, score 0.76)
  • Very limited nearby safe area (0.04 km² within 5km)
  • Remote road access (4.2km to major road)
  • No healthcare facility found in area
```

---

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

### Run tests

```bash
python -m pytest tests/ -q
```

Expected: **268+ tests passing**.

---

## Outputs

| File | Description |
|------|-------------|
| `PRAVAAH_Hazard_Grid.csv` | Per-cell hazard scores and features |
| `PRAVAAH_Hazard_Map.geojson` | Full hazard grid as GeoJSON |
| `PRAVAAH_Relocation_Priority.csv` | Ranked habitation assessment |
| `PRAVAAH_Risk_Assessment.pdf` | Full structured report |

---

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set `app.py` as the entry point
4. No API keys are required for core operation — OSM Overpass is free and unauthenticated
5. Optional: add `GOOGLE_MAPS_API_KEY` as a secret for Google Maps tile layer

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `GOOGLE_MAPS_API_KEY` | Google Maps tiles (optional) | No |

### Memory Notes

- 500 m cells on a 50 km × 50 km area: ~200–400 MB RAM
- Streamlit Cloud free tier (1 GB) handles typical study areas at 500 m resolution
- OSM results are cached locally — subsequent runs for the same area are instant

---

## Demo Flow

Suggested 3–5 minute walkthrough:

1. Open the app → select **Gottigere, Bangalore** → click **Run Analysis**
2. **Hazard Map** tab — identify red zones; note habitation markers on the map
3. Click a red marker — view hazard score, population, priority class
4. **Habitations** tab — ranked list; filter to CRITICAL
5. Select a CRITICAL habitation — read the full component breakdown
6. Expand **Full Explainability Report** — walk through the WHY narrative
7. **Relocation Priority** tab — priority distribution chart + ranked table
8. **Data & Export** tab — generate and download the PDF report

Key talking points:
- "PRAVAAH doesn't just show a risk map — it answers which habitations need to move and why"
- "Every score uses declared, auditable weights — visible in the Methodology tab"
- "Population is UNKNOWN when absent — we never fabricate data"
- "Judges can inspect the exact formula in the source code or Methodology tab"

---

## Known Limitations

1. Drainage capacity is synthetic — real municipal stormwater network data would improve accuracy
2. Population from OSM is sparse in India — many habitations show UNKNOWN
3. Road and healthcare distances are straight-line only — routing would be more accurate
4. No temporal forecasting — hazard assessment reflects current static conditions
5. Habitation completeness depends on OSM mapping quality for the study area
6. Hazard model metrics are cross-validation on WSI pseudo-labels, not calibrated against real events
7. Shelter capacity is unavailable — no curated national shelter dataset is integrated

---

## Future Scope

- Integration of authoritative population raster (WorldPop or Census ward data)
- Road routing for realistic evacuation time estimates
- Temporal forecasting using forecast rainfall and current susceptibility baseline
- Historical flood-event validation against satellite-derived inundation extents
- Sentinel-1 flood extent comparison layer
- What-if scenario simulation (rainfall +20 %, reduced drainage capacity, etc.)

---

## Project Structure

```
pravaah/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── flood_risk_zonation/            # Core analysis package
│   ├── config.py                   # BoundingBox, PipelineConfig
│   ├── pipeline.py                 # Hazard analysis pipeline
│   ├── sih_pipeline.py             # Habitation intelligence pipeline
│   ├── models.py                   # Typed data model dataclasses
│   ├── exceptions.py               # Exception hierarchy
│   ├── habitation/                 # Settlement ingestion
│   ├── exposure/                   # Exposure analysis
│   ├── vulnerability/              # Vulnerability scoring
│   ├── capacity/                   # Carrying capacity assessment
│   ├── relocation/                 # Relocation priority
│   ├── features/                   # Geospatial feature extraction
│   ├── grid/                       # Grid generation
│   ├── ingest/                     # Data ingestion (elevation, rainfall, water, population)
│   ├── scoring/                    # Susceptibility models (WSI, RF, Ensemble)
│   ├── utils/                      # Cache, CRS, validation utilities
│   └── visualization/              # Map builder, layers, PDF report, explainability
├── data/
│   ├── elevation/                  # SRTM GeoTIFF files
│   ├── water_bodies/               # Cached OSM water body GeoJSON
│   ├── drainage_lines/             # Drainage line GeoJSON
│   └── landmask/                   # Natural Earth land polygon
└── tests/
    ├── unit/                       # Unit tests
    ├── integration/                # Integration tests
    └── property/                   # Property-based tests (Hypothesis)
```

---

*PRAVAAH — Predictive Risk & Vulnerability Assessment for At-Risk Habitations*
