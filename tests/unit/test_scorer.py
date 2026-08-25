"""Unit tests for FloodRiskScorer."""
import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box

from flood_risk_zonation.scoring.scorer import FloodRiskScorer
from flood_risk_zonation.model.trainer import FloodRiskModelTrainer
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS


def _make_scored_grid():
    """Create a minimal scored GeoDataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    trainer = FloodRiskModelTrainer(n_estimators=10)
    result = trainer.train(X, y, cv_folds=3)

    geoms = [box(i * 0.01, 0, (i + 1) * 0.01, 0.01) for i in range(n)]
    grid = gpd.GeoDataFrame(X.copy(), geometry=geoms, crs="EPSG:4326")
    grid["cell_id"] = [f"0_{i}" for i in range(n)]
    grid["centroid_lat"] = 0.005
    grid["centroid_lon"] = [i * 0.01 + 0.005 for i in range(n)]

    scorer = FloodRiskScorer()
    scorer.p_min = trainer.p_min
    scorer.p_max = trainer.p_max
    scored = scorer.score_grid(grid, result.model, FEATURE_COLUMNS)
    return scored, scorer, result.model


def test_classify_known_values():
    """classify() with known values must produce correct labels."""
    scorer = FloodRiskScorer()
    scores = np.array([0.0, 33.0, 33.1, 66.0, 66.1, 100.0])
    labels = scorer.classify(scores)
    expected = ["Low", "Low", "Medium", "Medium", "High", "High"]
    assert list(labels) == expected


def test_normalize_scores_bounds():
    """normalize_scores() must map min→0 and max→100."""
    scorer = FloodRiskScorer()
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    scores = scorer.normalize_scores(probs, p_min=0.1, p_max=0.9)
    assert scores[0] == pytest.approx(0.0, abs=1e-6)
    assert scores[-1] == pytest.approx(100.0, abs=1e-6)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 100.0)


def test_score_grid_appends_columns():
    """score_grid() must append risk_score and risk_class with no NaN."""
    scored, _, _ = _make_scored_grid()
    assert "risk_score" in scored.columns
    assert "risk_class" in scored.columns
    assert not scored["risk_score"].isna().any()
    assert not scored["risk_class"].isna().any()
    assert set(scored["risk_class"].unique()).issubset({"Low", "Medium", "High"})
    assert scored["risk_score"].between(0.0, 100.0).all()
