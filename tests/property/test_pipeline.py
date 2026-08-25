"""
Property 15: Pipeline Output Completeness
Property 16: Pipeline Determinism

Validates: Requirements 7.1, 7.2
"""
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline


def _make_bbox(min_lon, min_lat, lon_delta, lat_delta):
    from hypothesis import assume as hyp_assume
    max_lon = min_lon + lon_delta
    max_lat = min_lat + lat_delta
    hyp_assume(-180 <= min_lon and max_lon <= 180)
    hyp_assume(-90 <= min_lat and max_lat <= 90)
    return BoundingBox(min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


valid_bboxes = st.builds(
    _make_bbox,
    min_lon=st.floats(0, 0.5),
    min_lat=st.floats(0, 0.5),
    lon_delta=st.floats(0.5, 0.8),
    lat_delta=st.floats(0.5, 0.8),
)


@given(bbox=valid_bboxes)
@settings(max_examples=3, deadline=None)
def test_pipeline_output_completeness(bbox):
    """Property 15: Every cell has non-null risk_score in [0,100] and valid risk_class."""
    config = PipelineConfig(cell_size_meters=5000, rf_n_estimators=10, cv_folds=3, use_cache=False)
    pipeline = FloodRiskPipeline(config)
    result = pipeline.run(bbox)
    grid = result.scored_grid
    assert result.cell_count > 0
    assert not grid["risk_score"].isna().any()
    assert not grid["risk_class"].isna().any()
    assert grid["risk_score"].between(0.0, 100.0).all()
    assert set(grid["risk_class"].unique()).issubset({"Low", "Medium", "High", "Water"})


@given(bbox=valid_bboxes)
@settings(max_examples=3, deadline=None)
def test_pipeline_determinism(bbox):
    """Property 16: Same random_seed produces identical risk_score and risk_class."""
    config = PipelineConfig(cell_size_meters=5000, rf_n_estimators=10, cv_folds=3,
                            use_cache=False, random_seed=42)
    r1 = FloodRiskPipeline(config).run(bbox)
    r2 = FloodRiskPipeline(config).run(bbox)
    np.testing.assert_array_equal(
        r1.scored_grid["risk_score"].values,
        r2.scored_grid["risk_score"].values,
    )
    assert list(r1.scored_grid["risk_class"]) == list(r2.scored_grid["risk_class"])
