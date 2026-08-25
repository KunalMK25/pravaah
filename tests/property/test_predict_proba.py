"""Property 10: Predicted Probabilities Are Valid — Validates: Requirements 4.4"""
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from flood_risk_zonation.model.trainer import FloodRiskModelTrainer
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS

_SHARED_TRAINER_P10 = None

def _get_trainer():
    global _SHARED_TRAINER_P10
    if _SHARED_TRAINER_P10 is None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.random((200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        y = pd.Series((rng.random(200) > 0.5).astype(int))
        _SHARED_TRAINER_P10 = FloodRiskModelTrainer(n_estimators=10)
        _SHARED_TRAINER_P10.train(X, y, cv_folds=3)
    return _SHARED_TRAINER_P10


@given(n_samples=st.integers(1, 50))
@settings(max_examples=20, deadline=None)
def test_predict_proba_values_in_0_1(n_samples):
    """Property 10: All predict_proba values are in [0, 1] and rows sum to 1."""
    trainer = _get_trainer()
    rng = np.random.default_rng(n_samples + 100)
    X = pd.DataFrame(rng.random((n_samples, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    proba = trainer._model.predict_proba(X)
    assert np.all(proba >= 0.0), f"Negative probabilities: {proba[proba < 0]}"
    assert np.all(proba <= 1.0), f"Probabilities > 1: {proba[proba > 1]}"
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)
