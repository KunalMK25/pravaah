# PRAVAAH — Deployment Guide

## Recommended Platform: Streamlit Community Cloud

Streamlit Community Cloud is the preferred deployment target because it:
- Supports all PRAVAAH geospatial dependencies (geopandas, rasterio, shapely)
- Has free tier with adequate memory for 500m-cell grid analysis
- Integrates natively with GitHub for continuous deployment
- Supports encrypted secrets management

### One-time Setup

1. **Push the repository to GitHub**
   The repository is at: `https://github.com/KunalMK25/pravaah`

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Click "New app"** and connect your GitHub account

4. **Configure the app:**
   - Repository: `KunalMK25/pravaah`
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: 3.11 (recommended)

5. **Configure secrets** (Settings → Secrets):
   ```toml
   # Required for live weather (optional — app works without it)
   OPENWEATHER_API_KEY = "your_key_here"

   # Optional LLM integration (app works in rule-based mode without it)
   PRAVAAH_LLM_PROVIDER = "openai"
   OPENAI_API_KEY = "sk-..."
   # OR
   PRAVAAH_LLM_PROVIDER = "anthropic"
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

6. **Click Deploy** — Streamlit Cloud will install from `requirements.txt` automatically.

### System Packages (packages.txt)

`packages.txt` is included with GDAL/spatial index packages for cloud deployment.
Streamlit Cloud reads this file automatically.

### Environment Variables Reference

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENWEATHER_API_KEY` | Optional | (none) | Live weather data |
| `PRAVAAH_LLM_PROVIDER` | Optional | `none` | `openai`/`anthropic`/`none` |
| `OPENAI_API_KEY` | If LLM=openai | (none) | OpenAI API key |
| `ANTHROPIC_API_KEY` | If LLM=anthropic | (none) | Anthropic API key |
| `PRAVAAH_OPENAI_MODEL` | Optional | `gpt-4o-mini` | OpenAI model |
| `PRAVAAH_ANTHROPIC_MODEL` | Optional | `claude-3-haiku-20240307` | Anthropic model |
| `PRAVAAH_WEATHER_CACHE_DIR` | Optional | `data/cache/weather` | Weather cache path |

### PRAVAAH works fully without any API keys:
- Weather: UNAVAILABLE (no dynamic adjustment — baseline analysis still complete)
- LLM: UNAVAILABLE (rule-based agent fallback — all decisions still produced)
- All 12 tabs functional
- All exports (CSV/GeoJSON/PDF) functional

---

## Alternative: Hugging Face Spaces

1. Create a new Space at `huggingface.co/spaces`
2. Choose **Streamlit** SDK
3. Push the repository contents to the Space repo
4. Add secrets in Settings → Repository Secrets (same variables as above)

---

## Local Development

```bash
# Clone
git clone https://github.com/KunalMK25/pravaah.git
cd pravaah

# Install
pip install -r requirements.txt

# Optional: configure environment
cp .env.example .env
# Edit .env with your API keys

# Run
streamlit run app.py

# Run tests
python -m pytest tests/ -q --no-cov
```

---

## Memory Notes

| Grid Size | Cell Size | Memory |
|-----------|-----------|--------|
| Typical study area (7km × 7km) | 500m | ~150–300 MB |
| Large area (50km × 50km) | 1000m | ~400–600 MB |

Streamlit Community Cloud free tier (1 GB RAM) handles typical 500m-cell analyses.
For large areas, use 1000m resolution.

---

## Health Check

PRAVAAH includes a built-in health status indicator in the sidebar showing:
- App core: AVAILABLE/UNAVAILABLE
- Weather API: AVAILABLE/DEGRADED/UNAVAILABLE
- OSM Overpass: AVAILABLE/DEGRADED/UNAVAILABLE
- LLM: AVAILABLE/UNAVAILABLE
- SHAP: AVAILABLE/UNAVAILABLE
