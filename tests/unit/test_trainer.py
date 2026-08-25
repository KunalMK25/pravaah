"""Unit tests for FloodRiskModelTrainer."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flood_risk_zonation.model.trainer import FloodRiskModelTrainer
from flood_risk_zonation.exceptions import ModelTrainingError
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS


def make_synthetic_data(n_samples: int = 200, seed: int = 42):
    """Generate synthetic feature matrix and binary labels."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n_samples, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.Series((rng.random(n_samples) > 0.5).astype(int))
    return X, y


def test_train_raises_for_too_few_samples():
    """train() must raise ModelTrainingError with < 50 samples."""
    X, y = make_synthetic_data(n_samples=30)
    trainer = FloodRiskModelTrainer()
    with pytest.raises(ModelTrainingError, match="50"):
        trainer.train(X, y)


def test_train_raises_for_single_class():
    """train() must raise ModelTrainingError when all labels are the same class."""
    X, _ = make_synthetic_data(n_samples=100)
    y = pd.Series(np.zeros(100, dtype=int))
    trainer = FloodRiskModelTrainer()
    with pytest.raises(ModelTrainingError, match="2 classes"):
        trainer.train(X, y)


def test_train_returns_valid_training_result():
    """train() must return TrainingResult with mean_cv_auc in [0, 1]."""
    X, y = make_synthetic_data(n_samples=200)
    trainer = FloodRiskModelTrainer(n_estimators=10)
    result = trainer.train(X, y, cv_folds=3)
    assert 0.0 <= result.mean_cv_auc <= 1.0
    assert 0.0 <= result.mean_cv_f1 <= 1.0
    assert len(result.feature_importances) == len(FEATURE_COLUMNS)
    assert abs(sum(result.feature_importances.values()) - 1.0) < 0.01


def test_save_and_load_produce_identical_predictions():
    """save() + load() must produce bit-for-bit identical predictions."""
    X, y = make_synthetic_data(n_samples=200)
    trainer = FloodRiskModelTrainer(n_estimators=10)
    trainer.train(X, y, cv_folds=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        trainer.save(path)
        loaded = FloodRiskModelTrainer.load(path)

    X_test, _ = make_synthetic_data(n_samples=20, seed=99)
    orig_preds = trainer._model.predict_proba(X_test)
    loaded_preds = loaded._model.predict_proba(X_test)
    # Use allclose with tight tolerance: serialization preserves predictions
    # within floating-point machine epsilon (~2e-16)
    np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-14, rtol=0)
