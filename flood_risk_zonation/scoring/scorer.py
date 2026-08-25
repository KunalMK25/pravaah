"""
Risk scorer — normalization and classification for the Flood Risk Zonation System.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from flood_risk_zonation.exceptions import ScoringError

logger = logging.getLogger(__name__)


class FloodRiskScorer:
    """
    Normalizes raw model probabilities to [0, 100] and classifies into
    Low / Medium / High risk categories.
    """

    DEFAULT_THRESHOLDS = {"low_max": 33.0, "medium_max": 66.0}

    def __init__(self) -> None:
        self.p_min: float = 0.0
        self.p_max: float = 1.0

    def calibrate(self, raw_probabilities: np.ndarray) -> None:
        """Set p_min/p_max from the 1st and 99th percentile of training probs."""
        self.p_min = float(np.percentile(raw_probabilities, 1))
        self.p_max = float(np.percentile(raw_probabilities, 99))
        if self.p_max <= self.p_min:
            self.p_min = float(raw_probabilities.min())
            self.p_max = float(raw_probabilities.max())

    def normalize_scores(
        self,
        raw_probabilities: np.ndarray,
        p_min: float | None = None,
        p_max: float | None = None,
    ) -> np.ndarray:
        """
        Map raw probabilities to [0, 100] using min-max scaling.

        score = (p - p_min) / (p_max - p_min) * 100, clipped to [0, 100].
        """
        lo = p_min if p_min is not None else self.p_min
        hi = p_max if p_max is not None else self.p_max
        denom = hi - lo
        if denom <= 0:
            # Degenerate case: all probabilities identical → return 50
            return np.full(len(raw_probabilities), 50.0, dtype=np.float64)
        scores = (raw_probabilities - lo) / denom * 100.0
        return np.clip(scores, 0.0, 100.0)

    def classify(
        self,
        scores: np.ndarray,
        thresholds: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Apply threshold classification to produce categorical labels.

        score <= low_max          → "Low"
        low_max < score <= medium_max → "Medium"
        score > medium_max        → "High"
        """
        t = thresholds if thresholds is not None else self.DEFAULT_THRESHOLDS
        low_max = t.get("low_max", 33.0)
        medium_max = t.get("medium_max", 66.0)

        labels = np.where(
            scores <= low_max,
            "Low",
            np.where(scores <= medium_max, "Medium", "High"),
        )
        return labels

    def score_grid(
        self,
        grid: gpd.GeoDataFrame,
        model,
        feature_columns: list[str],
        thresholds: dict[str, float] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Run end-to-end scoring: predict → normalize → classify.
        Appends 'risk_score' and 'risk_class' columns to the grid.
        """
        X = grid[feature_columns].values
        X_df = pd.DataFrame(X, columns=feature_columns)

        raw_probs = model.predict_proba(X_df)[:, -1]

        # Calibrate from this batch if not already calibrated
        if self.p_max <= self.p_min:
            self.calibrate(raw_probs)

        scores = self.normalize_scores(raw_probs)
        labels = self.classify(scores, thresholds)

        result = grid.copy()
        result["risk_score"] = scores.astype(np.float32)
        result["risk_class"] = labels
        return result
