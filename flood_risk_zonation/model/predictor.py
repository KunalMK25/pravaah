"""
ML model predictor for the Flood Risk Zonation System.

Wraps a trained sklearn/LightGBM estimator to produce flood risk
probabilities and feature importance rankings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FloodRiskPredictor:
    """
    Inference wrapper around a trained flood risk classifier.

    Parameters
    ----------
    model : Any
        A fitted sklearn or LightGBM estimator with predict_proba().
    """

    def __init__(self, model=None) -> None:
        self.model = model

    def predict(self, X: pd.DataFrame, model=None) -> np.ndarray:
        """
        Return the predicted probability of the high-risk class for each sample.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (n_samples, n_features). No NaN or inf.
        model : optional
            If provided, use this model instead of self.model.

        Returns
        -------
        np.ndarray
            1D array of shape (n_samples,) with probabilities in [0, 1].
        """
        clf = model if model is not None else self.model
        proba = clf.predict_proba(X)
        # Return probability of the positive (high-risk) class — last column
        return proba[:, -1]

    def get_feature_importance(
        self,
        model=None,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Return feature importances sorted by importance descending.

        Parameters
        ----------
        model : optional
            If provided, use this model instead of self.model.
        feature_names : list[str] | None
            Feature names. If None, uses generic names f0, f1, ...

        Returns
        -------
        dict[str, float]
            Mapping of feature name → importance score, sorted descending.
        """
        clf = model if model is not None else self.model
        importances = clf.feature_importances_

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]

        return dict(
            sorted(
                zip(feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )
