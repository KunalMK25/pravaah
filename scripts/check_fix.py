"""Quick check of the water masking fix — no network calls."""
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import unary_union
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
from pathlib import Path
import json

# ── Test 1: Bangalore masking with synthetic water bodies ──────────────────
print("=== Bangalore fix check ===")
bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
config = PipelineConfig(cell_size_meters=500, use_cache=False, allow_network=False, random_seed=42)

grid = generate_grid(bbox, 500)
elevation = load_elevation(bbox, Path("data/elevation"))
rainfall = load_rainfall(bbox, Path("data/rainfall"))
population = load_population(bbox, data_dir=str(config.cache_dir))
drainage = generate_synthetic_drainage(grid, seed=42)

# Simulate realistic water bodies: small lake + drain LineString
lake_poly = box(77.565, 12.855, 77.575, 12.865)   # small lake polygon
drain_line = box(77.58, 12.84, 77.5801, 12.91)     # thin drain sliver
wb = gpd.GeoDataFrame(
    {"geometry": [lake_poly, drain_line], "water_type": ["water", "drain"], "name": ["Lake", "Drain"]},
    crs="EPSG:4326"
)
wb.attrs["source"] = "test"

featured = extract_features(grid, elevation, rainfall, wb, population, drainage)
X = featured[FEATURE_COLUMNS].copy()
model = WeightedSusceptibilityModel().fit(X)
scorer = FloodRiskScorer()
scorer.p_min = 0.0; scorer.p_max = 1.0
scored = scorer.score_grid(featured, model, FEATURE_COLUMNS, {"low_max": 33.0, "medium_max": 66.0})

pipeline = FloodRiskPipeline(config)
result = pipeline._apply_water_mask_and_proximity_boost(scored, wb, config, elevation_source="data\\elevation\\gottigere_srtm.tif")
dist = result["risk_class"].value_counts().to_dict()
total = len(result)
pct = {k: round(v/total*100,1) for k,v in dist.items()}
print(f"  With lake+drain: {pct}")
water_pct = pct.get("Water", 0)
print(f"  Water %: {water_pct} (should be ~5-15%, not 48%)")
assert water_pct < 20, f"FAIL: Water% too high ({water_pct}%) — drain sliver still masking cells"
print("  PASS: Drain sliver does NOT cause over-masking")

# ── Test 2: Real lake polygon should still be masked ──────────────────────
print()
print("=== Lake masking still works ===")
lake_big = box(77.555, 12.845, 77.595, 12.885)  # large lake covering ~9 cells
wb2 = gpd.GeoDataFrame(
    {"geometry": [lake_big], "water_type": ["water"], "name": ["Big Lake"]},
    crs="EPSG:4326"
)
wb2.attrs["source"] = "test"
featured2 = extract_features(grid, elevation, rainfall, wb2, population, drainage)
X2 = featured2[FEATURE_COLUMNS].copy()
model2 = WeightedSusceptibilityModel().fit(X2)
scored2 = scorer.score_grid(featured2, model2, FEATURE_COLUMNS, {"low_max": 33.0, "medium_max": 66.0})
result2 = pipeline._apply_water_mask_and_proximity_boost(scored2, wb2, config, elevation_source="data\\elevation\\gottigere_srtm.tif")
dist2 = result2["risk_class"].value_counts().to_dict()
water_count = dist2.get("Water", 0)
print(f"  Distribution: {dist2}")
print(f"  Water cells: {water_count} (should be > 0 — lake centroids inside polygon)")
assert water_count > 0, "FAIL: Real lake polygon not masking cells"
print("  PASS: Real lake polygon correctly masks cells")

# ── Test 3: Map HTML has tooltips/popups ──────────────────────────────────
print()
print("=== Map HTML interactivity check ===")
import tempfile, pathlib, re
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder
from flood_risk_zonation.models import AnalysisResult, FloodRiskResult
import time

analysis = AnalysisResult(model=model2, feature_names=list(model2.feature_names),
    feature_importances=model2.feature_importances,
    method="weighted_susceptibility_index", validation_note="test")
fake_result = FloodRiskResult(scored_grid=result2, analysis_result=analysis,
    bounding_box=bbox, config=config, pipeline_duration_seconds=1.0,
    cell_count=len(result2), data_tier=3)

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

has_tooltip = "bindTooltip" in html or "tooltip" in html.lower()
has_popup = "bindPopup" in html or "popup" in html.lower()
has_cell_info = "Cell Info" in html
fg_names = re.findall(r'name: "([^"]+)"', html)
print(f"  Has tooltip: {has_tooltip}")
print(f"  Has popup: {has_popup}")
print(f"  Has Cell Info layer: {has_cell_info}")
print(f"  HTML size: {round(len(html)/1024)}kb")
print(f"  FeatureGroup order: {fg_names}")
assert has_tooltip, "FAIL: No tooltips in HTML"
assert has_popup, "FAIL: No popups in HTML"
assert has_cell_info, "FAIL: Cell Info layer missing"
# Cell Info must be LAST in the layer list (so it's on top in Leaflet z-order)
if fg_names:
    assert fg_names[-1] == "Cell Info (hover/click)", f"FAIL: Cell Info not last. Order: {fg_names}"
print("  PASS: Cell Info layer present and last (on top)")

print()
print("All checks passed.")
