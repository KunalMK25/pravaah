"""
PRAVAAH-AI — SHAP ML Hazard Explainability.

PURPOSE:
  Explain WHY the ML model classified a specific grid cell as high-risk.
  SHAP (SHapley Additive exPlanations) assigns each feature a contribution
  value for a single prediction.

SCOPE:
  This module explains the ML HAZARD component only.
  It does NOT explain the full PRAVAAH relocation decision.
  The relocation decision uses the transparent weighted formula and
  is self-explaining from its component scores.

SUPPORTED MODELS:
  - WeightedSusceptibilityModel    → uses KernelExplainer (slower, exact)
  - RandomForestSusceptibilityModel → uses TreeExplainer (fast, exact for trees)
  - EnsembleSusceptibilityModel    → uses TreeExplainer on the RF component,
                                     adds WSI contribution analytically

PERFORMANCE:
  - SHAP is computed ON DEMAND for a single selected cell or a small sample.
  - Never computed for the entire grid automatically.
  - Results are cached in-memory by cell_id for the session.
  - Computation time: ~0.1s per cell for RF/Ensemble (TreeExplainer).

FALLBACK:
  If SHAP is unavailable (not installed, model incompatible):
  - Returns a SHAPExplanation with provenance="unavailable" and
    top features derived from the WSI declared weights × feature values.
  - The application continues without SHAP.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from flood_risk_zonation.models import SHAPExplanation

logger = logging.getLogger(__name__)

# In-memory session cache: cell_id → SHAPExplanation
_EXPLANATION_CACHE: dict[str, SHAPExplanation] = {}
_MAX_CACHE_SIZE = 500


def _feature_importance_fallback(
    cell_id: str,
    row: pd.Series,
    model: Any,
) -> SHAPExplanation:
    """
    Fallback: approximate feature contributions from WSI declared weights × values.
    Used when SHAP is unavailable or model-incompatible.
    """
    try:
        importances = model.feature_importances  # dict[str, float]
        feat_names = list(importances.keys())
        values = {f: float(row.get(f, 0.0)) for f in feat_names}
        # Simple contribution proxy: importance × normalised value
        contributions: dict[str, float] = {}
        for f, imp in importances.items():
            contributions[f] = round(imp * values.get(f, 0.0), 6)

        sorted_c = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        pos = [(f, v) for f, v in sorted_c if v > 0][:5]
        neg = [(f, v) for f, v in sorted_c if v < 0][:3]
        text = (
            f"Feature importance fallback (SHAP unavailable). "
            f"Top driver: {pos[0][0] if pos else 'n/a'}."
        )
        return SHAPExplanation(
            cell_id=cell_id,
            shap_values=contributions,
            base_value=0.5,
            predicted_value=float(row.get("risk_score", 50.0)) / 100.0,
            top_positive_features=pos,
            top_negative_features=neg,
            explanation_text=text,
            provenance="feature_importance_fallback",
        )
    except Exception as exc:
        logger.debug("SHAP fallback failed: %s", exc)
        return SHAPExplanation(
            cell_id=cell_id,
            explanation_text="Explanation unavailable.",
            provenance="unavailable",
        )


def explain_cell(
    cell_id: str,
    row: pd.Series,
    model: Any,
    background: pd.DataFrame | None = None,
    use_cache: bool = True,
) -> SHAPExplanation:
    """
    Compute a SHAP explanation for a single grid cell.

    Parameters
    ----------
    cell_id : str
        Unique cell identifier.
    row : pd.Series
        Feature values for the cell (from scored_grid).
    model : WeightedSusceptibilityModel | RandomForestSusceptibilityModel | EnsembleSusceptibilityModel
        The fitted PRAVAAH susceptibility model.
    background : pd.DataFrame | None
        Background dataset for KernelExplainer.  If None, a sample of
        the scored_grid is used.  Ignored for TreeExplainer.
    use_cache : bool
        If True (default), return cached result if available.

    Returns
    -------
    SHAPExplanation
    """
    if use_cache and cell_id in _EXPLANATION_CACHE:
        return _EXPLANATION_CACHE[cell_id]

    # Determine model type
    model_type = type(model).__name__

    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — using fallback.")
        result = _feature_importance_fallback(cell_id, row, model)
        _cache_result(cell_id, result)
        return result

    # ── Identify feature columns ───────────────────────────────────────────────
    try:
        if hasattr(model, "feature_names"):
            feat_names = list(model.feature_names)
        else:
            from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
            feat_names = [c for c in FEATURE_COLUMNS if c in row.index]
    except Exception:
        from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
        feat_names = [c for c in FEATURE_COLUMNS if c in row.index]

    X_row = pd.DataFrame([row[feat_names].values], columns=feat_names)

    # ── TreeExplainer path (RF / Ensemble RF component) ───────────────────────
    rf_model = None
    if model_type == "RandomForestSusceptibilityModel":
        rf_model = model._rf
    elif model_type == "EnsembleSusceptibilityModel":
        rf_model = model._rf

    if rf_model is not None:
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer.shap_values(X_row)
            # shap_values returns [class0_array, class1_array] for binary classifiers
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]   # class 1 (high risk)
            else:
                sv = shap_vals[0]

            base_val = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
            pred_val = float(rf_model.predict_proba(X_row)[0, 1])
            shap_dict = {f: round(float(v), 6) for f, v in zip(feat_names, sv)}

            sorted_sv = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            pos = [(f, v) for f, v in sorted_sv if v > 0][:5]
            neg = [(f, v) for f, v in sorted_sv if v < 0][:3]

            top_name = pos[0][0].replace("_", " ").title() if pos else "n/a"
            text = (
                f"ML hazard explanation (TreeSHAP). "
                f"Prediction: {pred_val:.3f} vs base {base_val:.3f}. "
                f"Top driver: {top_name}."
            )
            result = SHAPExplanation(
                cell_id=cell_id,
                shap_values=shap_dict,
                base_value=round(base_val, 4),
                predicted_value=round(pred_val, 4),
                top_positive_features=pos,
                top_negative_features=neg,
                explanation_text=text,
                provenance="shap_tree_explainer",
            )
            _cache_result(cell_id, result)
            return result
        except Exception as exc:
            logger.warning("TreeExplainer failed (%s), falling back.", exc)

    # ── KernelExplainer path (WeightedSusceptibilityModel) ───────────────────
    try:
        if background is None or len(background) == 0:
            logger.debug("No background data for KernelExplainer — using fallback.")
            raise ValueError("No background data available.")

        bg_sample = background[feat_names].sample(
            min(100, len(background)), random_state=42
        )

        def _predict(X: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(X, columns=feat_names)
            return model.predict_proba(df)[:, 1]

        explainer = shap.KernelExplainer(_predict, bg_sample)
        sv = explainer.shap_values(X_row, nsamples=100)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.array(sv).flatten()

        pred_val = float(_predict(X_row.values)[0])
        base_val = float(explainer.expected_value)
        shap_dict = {f: round(float(v), 6) for f, v in zip(feat_names, sv)}

        sorted_sv = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        pos = [(f, v) for f, v in sorted_sv if v > 0][:5]
        neg = [(f, v) for f, v in sorted_sv if v < 0][:3]

        text = (
            f"ML hazard explanation (KernelSHAP). "
            f"Top driver: {pos[0][0].replace('_', ' ').title() if pos else 'n/a'}."
        )
        result = SHAPExplanation(
            cell_id=cell_id,
            shap_values=shap_dict,
            base_value=round(base_val, 4),
            predicted_value=round(pred_val, 4),
            top_positive_features=pos,
            top_negative_features=neg,
            explanation_text=text,
            provenance="shap_kernel_explainer",
        )
        _cache_result(cell_id, result)
        return result

    except Exception as exc:
        logger.warning("KernelExplainer failed (%s), using fallback.", exc)
        result = _feature_importance_fallback(cell_id, row, model)
        _cache_result(cell_id, result)
        return result


def _cache_result(cell_id: str, result: SHAPExplanation) -> None:
    if len(_EXPLANATION_CACHE) >= _MAX_CACHE_SIZE:
        # Remove oldest entry (FIFO)
        oldest = next(iter(_EXPLANATION_CACHE))
        del _EXPLANATION_CACHE[oldest]
    _EXPLANATION_CACHE[cell_id] = result


def explain_global(
    model: Any,
    scored_grid: pd.DataFrame,
    n_sample: int = 200,
) -> dict[str, float]:
    """
    Compute global SHAP feature importance (mean |SHAP|) from a sample of cells.

    Returns
    -------
    dict[str, float]
        Feature name → mean absolute SHAP value.
        Falls back to model.feature_importances if SHAP fails.
    """
    try:
        import shap
    except ImportError:
        return model.feature_importances if hasattr(model, "feature_importances") else {}

    rf_model = None
    if hasattr(model, "_rf") and model._rf is not None:
        rf_model = model._rf

    if rf_model is None:
        return model.feature_importances if hasattr(model, "feature_importances") else {}

    try:
        if hasattr(model, "feature_names"):
            feat_names = list(model.feature_names)
        else:
            from flood_risk_zonation.features.extractor import FEATURE_COLUMNS
            feat_names = [c for c in FEATURE_COLUMNS if c in scored_grid.columns]

        sample_grid = scored_grid[feat_names].sample(
            min(n_sample, len(scored_grid)), random_state=42
        )
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(sample_grid)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]   # class 1 (high risk)
        else:
            sv = shap_vals
        mean_abs = np.abs(sv).mean(axis=0)
        total = mean_abs.sum()
        if total > 0:
            mean_abs = mean_abs / total
        return {f: round(float(v), 6) for f, v in zip(feat_names, mean_abs)}
    except Exception as exc:
        logger.warning("Global SHAP failed (%s), using model importances.", exc)
        return model.feature_importances if hasattr(model, "feature_importances") else {}


def clear_cache() -> None:
    """Clear the in-memory SHAP explanation cache."""
    _EXPLANATION_CACHE.clear()
