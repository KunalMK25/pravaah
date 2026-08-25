"""
ML model trainer for the Flood Risk Zonation System.

Supports Random Forest (primary) and LightGBM (optional).
Uses stratified k-fold cross-validation and reports AUC-ROC and F1.
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from flood_risk_zonation.exceptions import ModelTrainingError
from flood_risk_zonation.models import TrainingResult

logger = logging.getLogger(__name__)

MIN_SAMPLES = 50


class FloodRiskModelTrainer:
    """
    Trains a flood risk classification model with stratified k-fold CV.

    Parameters
    ----------
    model_type : str
        "random_forest" (default) or "lightgbm".
    n_estimators : int
        Number of trees (Random Forest) or boosting rounds (LightGBM).
    min_samples_leaf : int
        Minimum samples per leaf (Random Forest only).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model_type: Literal["random_forest", "lightgbm"] = "random_forest",
        n_estimators: int = 200,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model = None
        self.p_min: float = 0.0
        self.p_max: float = 1.0

    def _build_model(self):
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to Random Forest.")
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
    ) -> TrainingResult:
        """
        Train the model with stratified k-fold cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (n_samples, n_features). Must have no NaN or inf.
        y : pd.Series
            Binary labels (0 = low/medium risk, 1 = high risk).
        cv_folds : int
            Number of CV folds.

        Returns
        -------
        TrainingResult
            Fitted model, CV scores, feature importances, timestamps.

        Raises
        ------
        ModelTrainingError
            If fewer than MIN_SAMPLES samples or all labels are the same class.
        """
        n_samples = len(X)
        if n_samples < MIN_SAMPLES:
            raise ModelTrainingError(
                f"Training requires at least {MIN_SAMPLES} samples, got {n_samples}."
            )

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ModelTrainingError(
                f"Training requires at least 2 classes, got {unique_classes}."
            )

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        auc_scores = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = self._build_model()
            fold_model.fit(X_train, y_train)

            y_prob = fold_model.predict_proba(X_val)
            # Handle case where fold has only one class (returns 1 column)
            if y_prob.shape[1] == 1:
                y_prob_pos = y_prob[:, 0]
            else:
                y_prob_pos = y_prob[:, 1]

            try:
                auc = roc_auc_score(y_val, y_prob_pos)
            except ValueError:
                auc = 0.5  # only one class in val fold
            f1 = f1_score(y_val, fold_model.predict(X_val), zero_division=0)

            auc_scores.append(float(auc))
            f1_scores.append(float(f1))
            logger.debug("Fold %d: AUC=%.4f, F1=%.4f", fold + 1, auc, f1)

        # Train final model on all data
        self._model = self._build_model()
        self._model.fit(X, y)

        # Calibrate score normalization range from training probabilities
        all_probs = self._model.predict_proba(X)[:, 1]
        self.p_min = float(np.percentile(all_probs, 1))
        self.p_max = float(np.percentile(all_probs, 99))
        if self.p_max <= self.p_min:
            self.p_min = float(all_probs.min())
            self.p_max = float(all_probs.max())

        # Feature importances
        feature_names = list(X.columns)
        importances = self._model.feature_importances_
        feature_importances = dict(
            sorted(zip(feature_names, importances.tolist()), key=lambda x: x[1], reverse=True)
        )

        return TrainingResult(
            model=self._model,
            feature_names=feature_names,
            feature_importances=feature_importances,
            cv_scores={"auc": auc_scores, "f1": f1_scores},
            mean_cv_auc=float(np.mean(auc_scores)),
            mean_cv_f1=float(np.mean(f1_scores)),
            training_timestamp=datetime.now(),
        )

    def save(self, path: Path | str) -> None:
        """Serialize the trainer (model + calibration params) to a .joblib file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "FloodRiskModelTrainer":
        """Deserialize a previously saved trainer from a .joblib file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        trainer = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return trainer
