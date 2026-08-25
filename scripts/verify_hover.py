"""Verify hover/click HTML structure and water masking fix."""
import logging
logging.disable(logging.CRITICAL)
import tempfile, pathlib, re
import geopandas as gpd
from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.scoring.susceptibility import WeightedSusceptibilityModel
from flood_risk_zonation.scoring.scorer import FloodRiskScorer
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
from flood_risk_zonation.models import AnalysisResult, FloodRiskResult
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder
from flood_risk_zonation.ingest.elevation import generate_synthetic_elevation
from flood_risk_zonation.ingest.rainfall import generate_synthetic_rainfall
from flood_risk_zonation.ingest.population import load_population
from flood_risk_zonation.ingest.drainage import generate_synthetic_drainage
from flood_risk_zonation.features.extractor import extract_features
from flood_risk_zonation.pipeline import FloodRiskPipeline

bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
config = PipelineConfig(cell_size_meters=500, use_cache=False, allow_network=False, random_seed=42)
grid = generate_grid(bbox, 500)
elev = generate_synthetic_elevation(bbox, resolution_m=500, seed=42)
rain = generate_synthetic_rainfall(bbox, resolution_m=1000, seed=42)
pop = load_population(bbox, data_dir=str(config.cache_dir))
drain = generate_synthetic_drainage(grid, seed=42)
wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
wb.attrs["source"] = "test"
feat = extract_features(grid, elev, rain, wb, pop, drain)
X = feat[FEATURE_COLUMNS].copy()
model = WeightedSusceptibilityModel().fit(X)
scorer = FloodRiskScorer()
scorer.p_min = 0.0
scorer.p_max = 1.0
scored = scorer.score_grid(feat, model, FEATURE_COLUMNS, {"low_max": 33.0, "medium_max": 66.0})
pipeline = FloodRiskPipeline(config)
result_grid = pipeline._apply_water_mask_and_proximity_boost(
    scored, wb, config, elevation_source="synthetic"
)
bounds = {f: (model.lower_[f], model.upper_[f]) for f in model.lower_}

dist = result_grid["risk_class"].value_counts().to_dict()
total = len(result_grid)
pct = {k: round(v / total * 100, 1) for k, v in dist.items()}
print("=== Water masking (no real WB) ===")
print("Distribution:", pct)
water_pct = pct.get("Water", 0)
print("Water %:", water_pct, "-> should be 0% (no water bodies)")
assert water_pct == 0, f"FAIL: {water_pct}% Water with empty water bodies"
print("PASS: No false Water masking")

builder = FloodRiskMapBuilder()
m = builder.build_choropleth_map(
    result_grid, center=bbox.center, zoom_start=12, model_bounds=bounds
)
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "test.html"
    m.save(str(p))
    html = p.read_text(encoding="utf-8")

print()
print("=== Map HTML interactivity ===")
has_tooltip = "bindTooltip" in html
has_popup = "bindPopup" in html
has_cell_info = "Cell Info" in html
cell_info_pos = html.find("Cell Info")
risk_class_pos = html.find("Risk Classification")
cell_info_last = cell_info_pos > risk_class_pos
fg_names = re.findall(r'name: "([^"]+)"', html)
print("Has bindTooltip:", has_tooltip)
print("Has bindPopup:", has_popup)
print("Has Cell Info layer:", has_cell_info)
print("Cell Info comes after Risk Classification:", cell_info_last)
print("FeatureGroup order:", fg_names)
print("HTML size (kb):", round(len(html) / 1024))

assert has_tooltip, "FAIL: No bindTooltip in HTML"
assert has_popup, "FAIL: No bindPopup in HTML"
assert has_cell_info, "FAIL: Cell Info layer missing"
assert cell_info_last, "FAIL: Cell Info not after Risk Classification"
if fg_names:
    assert fg_names[-1] == "Cell Info (hover/click)", f"FAIL: Cell Info not last. Got: {fg_names}"
print("PASS: Hover/click HTML structure correct")
print()
print("All checks passed.")
