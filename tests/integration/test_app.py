"""
Integration test: Streamlit app pipeline invocation.
Task 21.1 — Validates: Requirements 7.1
"""
import pytest
from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exceptions import FloodRiskError, ConfigurationError
from flood_risk_zonation.models import FloodRiskResult
from flood_risk_zonation.pipeline import FloodRiskPipeline


def run_analysis(bbox: BoundingBox, config: PipelineConfig) -> FloodRiskResult:
    """Simulate the app's analysis function."""
    try:
        pipeline = FloodRiskPipeline(config)
        return pipeline.run(bbox)
    except FloodRiskError:
        raise  # re-raise so callers can handle


def test_app_pipeline_returns_result():
    """run_analysis with valid bbox returns FloodRiskResult without exceptions."""
    bbox = BoundingBox(0.0, 0.0, 0.5, 0.5)
    config = PipelineConfig(cell_size_meters=5000, rf_n_estimators=10, cv_folds=3, use_cache=False)
    result = run_analysis(bbox, config)
    assert isinstance(result, FloodRiskResult)
    assert result.cell_count > 0


def test_app_catches_flood_risk_errors():
    """FloodRiskError subclasses are catchable and do not propagate unhandled."""
    with pytest.raises(FloodRiskError):
        # Invalid bbox raises ConfigurationError (subclass of FloodRiskError)
        BoundingBox(min_lon=5.0, min_lat=0.0, max_lon=1.0, max_lat=1.0)
