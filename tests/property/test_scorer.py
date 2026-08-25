"""
Property 11: Risk Score Normalization Bounds
Property 12: Risk Classification Threshold Correctness

Validates: Requirements 5.1, 5.2, 5.3
"""
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from flood_risk_zonation.scoring.scorer import FloodRiskScorer


@given(
    probs=arrays(
        dtype=float,
        shape=st.integers(1, 50),
        elements=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=20)
def test_normalize_scores_always_in_0_100(probs):
    """Property 11: All normalized scores are in [0, 100]."""
    scorer = FloodRiskScorer()
    p_min = float(probs.min())
    p_max = float(probs.max())
    scores = scorer.normalize_scores(probs, p_min=p_min, p_max=p_max)
    assert np.all(scores >= 0.0), f"Scores below 0: {scores[scores < 0]}"
    assert np.all(scores <= 100.0), f"Scores above 100: {scores[scores > 100]}"


@given(
    score=st.floats(0.0, 100.0, allow_nan=False),
    low_max=st.floats(1.0, 49.0, allow_nan=False),
    medium_max=st.floats(51.0, 99.0, allow_nan=False),
)
@settings(max_examples=20)
def test_classify_threshold_correctness(score, low_max, medium_max):
    """Property 12: Each score maps to exactly one label matching threshold rules."""
    assume(low_max < medium_max)
    scorer = FloodRiskScorer()
    labels = scorer.classify(np.array([score]), thresholds={"low_max": low_max, "medium_max": medium_max})
    assert len(labels) == 1
    label = labels[0]
    assert label in {"Low", "Medium", "High"}
    if score <= low_max:
        assert label == "Low", f"score={score} <= low_max={low_max} but got {label}"
    elif score <= medium_max:
        assert label == "Medium", f"score={score} in ({low_max},{medium_max}] but got {label}"
    else:
        assert label == "High", f"score={score} > medium_max={medium_max} but got {label}"
