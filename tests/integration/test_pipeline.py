"""
Integration test: full pipeline with synthetic data.
Task 17.1 — Validates: Requirements 7.1
"""
import pytest
from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.exceptions import FloodRiskError


def test_full_pipeline_synthetic_data():
    """FloodRiskPipeline.run on a small bbox returns a valid FloodRiskResult."""
    bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=0.5, max_lat=0.5)
    config = PipelineConfig(
        cell_size_meters=5000,
        rf_n_estimators=20,
        cv_folds=3,
        use_cache=False,
        random_seed=42,
    )
    pipeline = FloodRiskPipeline(config)
    result = pipeline.run(bbox)

    assert result.cell_count > 0
    assert "risk_score" in result.scored_grid.columns
    assert "risk_class" in result.scored_grid.columns
    assert result.scored_grid["risk_score"].between(0.0, 100.0).all()
    assert set(result.scored_grid["risk_class"].unique()).issubset({"Low", "Medium", "High", "Water"})
    # Pipeline uses EnsembleSusceptibilityModel by default.
    assert result.analysis_result.method == "ensemble"
    assert isinstance(result.analysis_result.feature_importances, dict)
    assert len(result.analysis_result.feature_importances) > 0


def test_pipeline_catches_flood_risk_errors():
    """FloodRiskError subclasses must not propagate as unhandled exceptions."""
    from flood_risk_zonation.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=10.0, min_lat=0.0, max_lon=5.0, max_lat=1.0)
