"""Diagnose water masking and map interactivity."""
import logging
logging.disable(logging.CRITICAL)
import tempfile, pathlib, re
from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.visualization.map_builder import FloodRiskMapBuilder

config = PipelineConfig(cell_size_meters=500, use_cache=False, allow_network=False, random_seed=42)
pipeline = FloodRiskPipeline(config)
bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)
result = pipeline.run(bbox)
dist = result.risk_distribution
total = sum(dist.values())
pct = {k: round(v/total*100,1) for k,v in dist.items()}
print("Bangalore after fix:")
print("  Distribution:", pct)
print("  Provenance:", result.data_provenance)

builder = FloodRiskMapBuilder()
_model = result.analysis_result.model
_bounds = {f: (_model.lower_[f], _model.upper_[f]) for f in _model.lower_}
m = builder.build_choropleth_map(
    result.scored_grid, center=result.bounding_box.center, zoom_start=12,
    model_bounds=_bounds,
)
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td) / "test.html"
    m.save(str(p))
    html = p.read_text(encoding="utf-8")

has_tooltip = "tooltip" in html.lower()
has_popup = "popup" in html.lower()
has_cell_info = "Cell Info" in html
fg_names = re.findall(r'name: "([^"]+)"', html)
print()
print("Map HTML checks:")
print("  Has tooltip:", has_tooltip)
print("  Has popup:", has_popup)
print("  Has Cell Info layer:", has_cell_info)
print("  HTML size (kb):", round(len(html)/1024))
print("  FeatureGroup order:", fg_names)
# Count GeoJson elements
geojson_count = html.count("L.geoJson")
print("  GeoJson element count:", geojson_count)
