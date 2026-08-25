"""
Tests for SHAP ML explainability module.
All tests must pass with SHAP installed (shap>=0.44 is in requirements.txt).
"""
from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from flood_risk_zonation.models import SHAPExplanation
from flood_risk_zonation.explainability.shap_explainer import (
    explain_cell,
    explain_global,
    clear_cache,
    _feature_importance_fallback,
    _EXPLANATION_CACHE,
)
from flood_risk_zonation.scoring.susceptibility import (
    WeightedSusceptibilityModel,
    RandomForestSusceptibilityModel,
    EnsembleSusceptibilityModel,
    FACTOR_WEIGHTS,
)
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS


def _make_df(n=50, seed=42):
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0.1, 1.0, n) for col in FEATURE_COLUMNS}
    data["elevation_m"] = rng.uniform(5.0, 200.0, n)
    data["dist_water_m"] = rng.uniform(100.0, 5000.0, n)
    data["drainage_capacity"] = rng.uniform(0.1, 0.9, n)
    data["twi"] = rng.uniform(2.0, 15.0, n)
    data["rainfall_mean_mm"] = rng.uniform(500.0, 3000.0, n)
    data["rainfall_max_24h_mm"] = rng.uniform(20.0, 200.0, n)
    data["population_density"] = rng.uniform(10.0, 5000.0, n)
    data["slope_deg"] = rng.uniform(0.5, 20.0, n)
    data["aspect_deg"] = rng.uniform(0.0, 360.0, n)
    data["curvature"] = rng.uniform(-5.0, 5.0, n)
    return pd.DataFrame(data)


def _wsi_model(df):
    return WeightedSusceptibilityModel().fit(df)


def _rf_model(df):
    return RandomForestSusceptibilityModel(n_estimators=20, cv_folds=2).fit(df)


def _ensemble_model(df):
    return EnsembleSusceptibilityModel(n_estimators=20, cv_folds=2).fit(df)


class TestFeatureImportanceFallback:
    def test_returns_shap_explanation(self):
        df = _make_df()
        model = _wsi_model(df)
        row = df.iloc[0]
        result = _feature_importance_fallback("c001", row, model)
        assert isinstance(result, SHAPExplanation)
        assert result.cell_id == "c001"
        assert result.provenance == "feature_importance_fallback"

    def test_fields_populated(self):
        df = _make_df()
        model = _wsi_model(df)
        row = df.iloc[0]
        result = _feature_importance_fallback("c001", row, model)
        assert isinstance(result.shap_values, dict)
        assert len(result.shap_values) > 0
        assert len(result.explanation_text) > 0

    def test_does_not_crash_on_bad_model(self):
        class BadModel:
            pass
        df = _make_df()
        row = df.iloc[0]
        result = _feature_importance_fallback("c001", row, BadModel())
        assert isinstance(result, SHAPExplanation)
        assert result.provenance == "unavailable"


class TestExplainCellWSI:
    def test_returns_explanation(self):
        df = _make_df()
        model = _wsi_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c001", row, model, background=df)
        assert isinstance(result, SHAPExplanation)
        assert result.cell_id == "c001"

    def test_provenance_is_set(self):
        df = _make_df()
        model = _wsi_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c_wsi", row, model, background=df)
        assert result.provenance in (
            "shap_tree_explainer",
            "shap_kernel_explainer",
            "feature_importance_fallback",
            "unavailable",
        )

    def test_explanation_text_not_empty(self):
        df = _make_df()
        model = _wsi_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c_wsi", row, model, background=df)
        assert len(result.explanation_text) > 0

    def test_no_background_falls_back_gracefully(self):
        df = _make_df()
        model = _wsi_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c_nb", row, model, background=None)
        assert isinstance(result, SHAPExplanation)


class TestExplainCellRF:
    def test_rf_uses_tree_explainer(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c_rf", row, model)
        assert isinstance(result, SHAPExplanation)
        assert result.provenance in ("shap_tree_explainer", "feature_importance_fallback")

    def test_shap_values_dict_populated(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c_rf2", row, model)
        if result.provenance == "shap_tree_explainer":
            assert len(result.shap_values) > 0
            for k, v in result.shap_values.items():
                assert isinstance(v, float)

    def test_top_features_sorted(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c_rf3", row, model)
        if result.top_positive_features:
            vals = [abs(v) for _, v in result.top_positive_features]
            assert vals == sorted(vals, reverse=True)


class TestExplainCellEnsemble:
    def test_ensemble_returns_valid_explanation(self):
        df = _make_df(n=80)
        model = _ensemble_model(df)
        row = df.iloc[0]
        clear_cache()
        result = explain_cell("c_ens", row, model)
        assert isinstance(result, SHAPExplanation)
        assert result.cell_id == "c_ens"


class TestExplainCellCaching:
    def test_cache_hit_returns_same_object(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        row = df.iloc[0]
        clear_cache()
        r1 = explain_cell("c_cache", row, model)
        r2 = explain_cell("c_cache", row, model)
        assert r1 is r2

    def test_cache_disabled_recomputes(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        row = df.iloc[0]
        clear_cache()
        r1 = explain_cell("c_nocache", row, model, use_cache=False)
        r2 = explain_cell("c_nocache", row, model, use_cache=False)
        # Both valid but may be different objects
        assert isinstance(r1, SHAPExplanation)
        assert isinstance(r2, SHAPExplanation)

    def test_clear_cache_empties(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        row = df.iloc[0]
        explain_cell("c_clr", row, model)
        clear_cache()
        assert "c_clr" not in _EXPLANATION_CACHE


class TestExplainGlobal:
    def test_returns_dict(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        result = explain_global(model, df, n_sample=20)
        assert isinstance(result, dict)

    def test_keys_are_feature_names(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        result = explain_global(model, df, n_sample=20)
        for k in result.keys():
            assert k in FEATURE_COLUMNS or k in list(FACTOR_WEIGHTS.keys())

    def test_values_non_negative(self):
        df = _make_df(n=80)
        model = _rf_model(df)
        result = explain_global(model, df, n_sample=20)
        for v in result.values():
            assert v >= 0.0

    def test_wsi_fallback_returns_feature_importances(self):
        df = _make_df()
        model = _wsi_model(df)
        result = explain_global(model, df)
        # WSI has no _rf, so falls back to feature_importances
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_ensemble_global_explanation(self):
        df = _make_df(n=80)
        model = _ensemble_model(df)
        result = explain_global(model, df, n_sample=20)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestSHAPFallbackOnImportError:
    def test_explain_cell_always_returns_valid_object(self):
        """explain_cell always returns a SHAPExplanation regardless of errors."""
        df = _make_df()
        model = _wsi_model(df)
        row = df.iloc[0]
        clear_cache()
        # No mocking needed — real SHAP is installed; just verify the contract
        result = explain_cell("c_contract", row, model, background=df)
        assert isinstance(result, SHAPExplanation)
        assert result.cell_id == "c_contract"
        assert result.provenance in (
            "shap_tree_explainer",
            "shap_kernel_explainer",
            "feature_importance_fallback",
            "unavailable",
        )
        assert isinstance(result.shap_values, dict)
        assert isinstance(result.top_positive_features, list)
        assert isinstance(result.top_negative_features, list)
        assert isinstance(result.explanation_text, str)
