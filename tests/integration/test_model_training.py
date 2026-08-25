"""
Integration test: model training AUC threshold.
Task 22 — Validates: Requirements 4.1, 4.3
"""
import numpy as np
import pandas as pd
import pytest

from flood_risk_zonation.model.trainer import FloodRiskModelTrainer
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS


def _make_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    # Create labels with some signal so AUC > 0.5
    signal = X["elevation_m"] + X["drainage_capacity"] - X["rainfall_mean_mm"]
    y = pd.Series((signal > signal.median()).astype(int))
    return X, y


def test_model_achieves_minimum_auc_on_synthetic_data():
    """Model must achieve mean_cv_auc >= 0.70 on 500 synthetic samples."""
    X, y = _make_data(n=500)
    trainer = FloodRiskModelTrainer(n_estimators=50, random_state=42)
    result = trainer.train(X, y, cv_folds=5)
    assert result.mean_cv_auc >= 0.70, (
        f"Expected mean_cv_auc >= 0.70, got {result.mean_cv_auc:.4f}"
    )


def test_model_feature_importances_sum_to_one():
    """Feature importance values must sum to approximately 1.0."""
    X, y = _make_data(n=200)
    trainer = FloodRiskModelTrainer(n_estimators=20, random_state=42)
    result = trainer.train(X, y, cv_folds=3)
    total = sum(result.feature_importances.values())
    assert abs(total - 1.0) < 0.01, f"Feature importances sum to {total:.4f}, expected ~1.0"
