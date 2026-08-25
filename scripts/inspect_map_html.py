"""Inspect the generated map HTML to verify tooltip/popup wiring."""
import logging, re, tempfile, pathlib
logging.disable(logging.CRITICAL)

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.ingest.elevation import load_elevation
from flood_risk_zonation.ingest.rainfall import load_rainfall
from flood_risk_zonation.ingest.population import load_population
from flood_risk_zonation.ingest.drainage import generate_synthetic_drainage
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS, extract_features
from flood_risk_zonation.scoring.scorer import FloodRiskScorer
from flood_risk_zonation.scoring.susceptibility import WeightedSusceptibilityModel
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.models import AnalysisResult, FloodRiskResult
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
config = PipelineConfig(cell_size_meters=500, use_cache=False, allow_network=False, random_seed=42)

grid = generate_grid(bbox, 500)
elev = load_elevation(bbox, Path("data/elevation"))
rain = load_rainfall(bbox, Path("data/rainfall"))
pop = load_population(bbox, data_dir=str(config.cache_dir))
drain = generate_synthetic_drainage(grid, seed=42)

# Small lake so some cells are Water, most are land
lake = box(77.562, 12.858, 77.574, 12.872)
wb = gpd.GeoDataFrame(
    {"geometry": [lake], "water_type": ["water"], "name": ["Lake"]},
    crs="EPSG:4326"
)
wb.attrs["source"] = "test"

feat = extract_features(grid, elev, rain, wb, pop, drain)
X = feat[FEATURE_COLUMNS].copy()
model = WeightedSusceptibilityModel().fit(X)
scorer = FloodRiskScorer(); scorer.p_min = 0.0; scorer.p_max = 1.0
scored = scorer.score_grid(feat, model, FEATURE_COLUMNS, {"low_max": 33.0, "medium_max": 66.0})
pipeline = FloodRiskPipeline(config)
result_grid = pipeline._apply_water_mask_and_proximity_boost(
    scored, wb, config, elevation_source="real"
)

dist = result_grid["risk_class"].value_counts().to_dict()
print("Risk distribution:", dist)

analysis = AnalysisResult(
    model=model, feature_names=list(model.feature_names),
    feature_importances=model.feature_importances,
    method="weighted_susceptibility_index", validation_note="test"
)
fake_result = FloodRiskResult(
    scored_grid=result_grid, analysis_result=analysis,
    bounding_box=bbox, config=config,
    pipeline_duration_seconds=1.0, cell_count=len(result_grid), data_tier=1
)

builder = FloodRiskMapBuilder()
_model = fake_result.analysis_result.model
_bounds = {f: (_model.lower_[f], _model.upper_[f]) for f in _model.lower_}
m = builder.build_choropleth_map(
    fake_result.scored_grid, center=bbox.center, zoom_start=12,
    model_bounds=_bounds,
)

with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "test.html"
    m.save(str(p))
    html = p.read_text(encoding="utf-8")

print(f"\nHTML size: {round(len(html)/1024)}kb")

# Check for tooltip/popup presence
print(f"Has bindTooltip: {'bindTooltip' in html}")
print(f"Has bindPopup:   {'bindPopup' in html}")
print(f"Has L.tooltip:   {'L.tooltip' in html}")
print(f"Has L.popup:     {'L.popup' in html}")

# Check FeatureGroup names and order
fg_names = re.findall(r'name:\s*"([^"]+)"', html)
print(f"\nFeatureGroup order: {fg_names}")
print(f"Cell Info is last: {fg_names[-1] == 'Cell Info (hover/click)' if fg_names else 'N/A'}")

# Count GeoJson layers
geojson_layers = html.count("L.geoJson(")
print(f"\nTotal GeoJson layers: {geojson_layers}")

# Sample the tooltip HTML for one cell
tooltip_idx = html.find("bindTooltip")
if tooltip_idx >= 0:
    snippet = html[tooltip_idx:tooltip_idx+200].replace("\n", " ")
    print(f"\nFirst bindTooltip snippet: {snippet[:200]}")
else:
    print("\nNO bindTooltip found in HTML!")
    # Check what the explainability layer actually emitted
    cell_info_idx = html.find("Cell Info")
    if cell_info_idx >= 0:
        snippet = html[cell_info_idx:cell_info_idx+500].replace("\n", " ")
        print(f"Cell Info context: {snippet[:500]}")
