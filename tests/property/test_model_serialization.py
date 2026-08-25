"""Property 9: Model Serialization Round-Trip — Validates: Requirements 4.2"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from flood_risk_zonation.model.trainer import FloodRiskModelTrainer
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS

# Pre-train a single model to reuse across examples (avoids re-training per example)
_SHARED_TRAINER = None

def _get_shared_trainer():
    global _SHARED_TRAINER
    if _SHARED_TRAINER is None:
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.random((200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        y = pd.Series((rng.random(200) > 0.5).astype(int))
        _SHARED_TRAINER = FloodRiskModelTrainer(n_estimators=10)
        _SHARED_TRAINER.train(X, y, cv_folds=3)
    return _SHARED_TRAINER


@given(n_samples=st.integers(1, 50))
@settings(max_examples=20, deadline=None)
def test_model_serialization_round_trip(n_samples):
    """Property 9: Serialized + deserialized model produces identical predictions."""
    trainer = _get_shared_trainer()
    rng = np.random.default_rng(n_samples)
    X_test = pd.DataFrame(rng.random((n_samples, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.joblib"
        trainer.save(path)
        loaded = FloodRiskModelTrainer.load(path)

    orig = trainer._model.predict_proba(X_test)
    loaded_preds = loaded._model.predict_proba(X_test)
    # Use allclose with tight tolerance: serialization preserves predictions
    # within floating-point machine epsilon (~2e-16)
    np.testing.assert_allclose(orig, loaded_preds, atol=1e-14, rtol=0)
