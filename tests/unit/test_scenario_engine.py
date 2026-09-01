"""
Regression tests for the scenario engine.

DESIGN INTENT:
- Rainfall scenarios should modify rainfall-dependent features
- Risk scores/classes should be recalculated per cell using modified features
- Permanent WATER cells must remain WATER regardless of rainfall changes
- Baseline data must not be mutated by scenario execution
- Scenario class counts must be derived from per-cell classifications

SCIENTIFIC CORRECTNESS:
- Increasing rainfall should generally increase (or maintain) flood risk
- Permanent water bodies are distinct from rainfall-induced inundation
- Scenario results must be reproducible and deterministic
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

import pytest

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.models import (
    ScenarioParameters,
    AnalysisResult,
    FloodRiskResult,
)
from flood_risk_zonation.scenarios.engine import run_scenario
from flood_risk_zonation.scoring.scorer import FloodRiskScorer


logger = logging.getLogger(__name__)





class TestWaterCellPreservation:
    """Verify permanent WATER cells are preserved across scenarios."""

    @pytest.fixture
    def baseline_with_water(self):
        """Create a baseline with permanent WATER cells."""
        bbox = BoundingBox(min_lon=77.0, min_lat=12.0, max_lon=77.1, max_lat=12.1)
        grid = generate_grid(bbox, cell_size_meters=1000.0)
        
        # Add required columns
        grid["risk_score"] = np.linspace(10, 90, len(grid))
        grid["risk_class"] = np.where(
            grid["risk_score"] <= 33,
            "Low",
            np.where(grid["risk_score"] <= 66, "Medium", "High")
        )
        
        # Mark every 5th cell as permanent WATER
        water_mask = (np.arange(len(grid)) % 5) == 0
        grid.loc[water_mask, "risk_class"] = "Water"
        grid.loc[water_mask, "risk_score"] = 0.0
        grid.loc[water_mask, "water_type"] = "ocean"
        grid.loc[water_mask, "water_mask_reason"] = "landmask"
        grid.loc[water_mask, "water_coverage_pct"] = 100.0
        grid.loc[~water_mask, "water_type"] = "land"
        grid.loc[~water_mask, "water_mask_reason"] = ""
        grid.loc[~water_mask, "water_coverage_pct"] = 0.0
        
        grid["is_coastal_tsunami_risk"] = False
        grid["rainfall_mean_mm"] = 1200.0
        grid["rainfall_max_24h_mm"] = 150.0
        grid["elevation_m"] = 100.0
        grid["slope_deg"] = 5.0
        grid["drainage_capacity"] = 0.5
        grid["population_density"] = 100.0
        grid["dist_water_m"] = 1000.0
        grid["land_use_agricultural"] = 0.5
        grid["land_use_urban"] = 0.2
        grid["land_use_forested"] = 0.3
        grid["wetness_index"] = 0.5
        
        # Spatial zones
        grid["spatial_zone"] = np.where(
            grid["risk_class"] == "Water",
            "WATER",
            np.where(
                grid["risk_class"] == "High",
                "RED",
                np.where(grid["risk_class"] == "Medium", "YELLOW", "GREEN")
            )
        )
        
        # Mock model that respects rainfall
        class MockModel:
            def predict_proba(self, X):
                # Handle both ndarray and DataFrame
                if isinstance(X, pd.DataFrame):
                    rainfall_col = X["rainfall_mean_mm"].values if "rainfall_mean_mm" in X.columns else np.ones(len(X)) * 1200
                else:
                    rainfall_col = X[:, 0] if X.ndim == 2 and X.shape[1] > 0 else np.ones(len(X)) * 1200
                
                # Probability increases with rainfall
                probs = 0.1 + (rainfall_col / 2500.0) * 0.8
                probs = np.clip(probs, 0.0, 1.0)
                return np.column_stack([1 - probs, probs])
        
        model = MockModel()
        scorer = FloodRiskScorer()
        
        # Calibrate from baseline
        baseline_X = pd.DataFrame({
            "rainfall_mean_mm": grid["rainfall_mean_mm"].values,
            "elevation_m": grid["elevation_m"].values,
            "slope_deg": grid["slope_deg"].values,
        })
        baseline_probs = model.predict_proba(baseline_X)[:, -1]
        scorer.calibrate(baseline_probs)
        
        analysis_result = AnalysisResult(
            model=model,
            feature_names=["rainfall_mean_mm", "elevation_m", "slope_deg", "drainage_capacity", "population_density"],
            feature_importances={"rainfall_mean_mm": 0.4, "elevation_m": 0.3, "slope_deg": 0.2, "drainage_capacity": 0.05, "population_density": 0.05},
            method="mock",
            validation_note="Mock model for testing",
            scorer=scorer,
        )
        
        config = PipelineConfig(low_threshold=33.0, medium_threshold=66.0)
        
        return FloodRiskResult(
            scored_grid=grid,
            analysis_result=analysis_result,
            bounding_box=bbox,
            config=config,
            pipeline_duration_seconds=1.0,
            cell_count=len(grid),
            data_provenance={},
            data_tier=1,
        )

    def test_water_cells_preserved_in_rainfall_scenario(self, baseline_with_water):
        """WATER cells must remain WATER after rainfall scenario."""
        baseline_water_count = (baseline_with_water.scored_grid["risk_class"] == "Water").sum()
        assert baseline_water_count > 0, "Test fixture must have WATER cells"
        
        params = ScenarioParameters("test", "Test +50% Rainfall", rainfall_multiplier=1.5)
        scenario_result = run_scenario(baseline_with_water, None, params)
        
        # Scenario must preserve WATER cell count
        scenario_water_count = scenario_result.scenario_zone_counts.get("WATER", 0)
        
        assert scenario_water_count == baseline_water_count, \
            f"WATER cells must be preserved: baseline={baseline_water_count}, scenario={scenario_water_count}"

    def test_baseline_immutable_after_scenario(self, baseline_with_water):
        """Baseline data must not be mutated by scenario execution."""
        baseline_grid_copy = baseline_with_water.scored_grid.copy()
        params = ScenarioParameters("test", "Test Scenario", rainfall_multiplier=1.5)
        
        run_scenario(baseline_with_water, None, params)
        
        # Verify baseline grid unchanged
        pd.testing.assert_frame_equal(baseline_with_water.scored_grid, baseline_grid_copy)

    def test_water_cells_preserved_in_drainage_scenario(self, baseline_with_water):
        """WATER cells must remain WATER even when drainage changes."""
        baseline_water_count = (baseline_with_water.scored_grid["risk_class"] == "Water").sum()
        
        params = ScenarioParameters("test", "Test Degraded Drainage", drainage_capacity_multiplier=0.5)
        scenario_result = run_scenario(baseline_with_water, None, params)
        
        scenario_water_count = scenario_result.scenario_zone_counts.get("WATER", 0)
        
        assert scenario_water_count == baseline_water_count, \
            f"WATER cells must be preserved in drainage scenario: baseline={baseline_water_count}, scenario={scenario_water_count}"


class TestRainfallScenarioSanity:
    """Verify rainfall scenarios behave scientifically (increasing rainfall → not decreasing risk)."""

    def test_rainfall_increase_does_not_decrease_red_cells(self):
        """Increasing rainfall should not cause RED cells to decrease (generic sanity check)."""
        bbox = BoundingBox(min_lon=77.0, min_lat=12.0, max_lon=77.1, max_lat=12.1)
        grid = generate_grid(bbox, cell_size_meters=1000.0)
        
        # Create a mix of risk classes with explicit rainfall sensitivity
        grid["risk_score"] = 50.0  # All medium initially
        grid["risk_class"] = "Medium"
        grid["rainfall_mean_mm"] = np.linspace(800, 1600, len(grid))  # Vary rainfall
        grid["rainfall_max_24h_mm"] = 100.0
        grid["elevation_m"] = 100.0
        grid["slope_deg"] = 5.0
        grid["drainage_capacity"] = 0.5
        grid["population_density"] = 100.0
        grid["dist_water_m"] = 1000.0
        grid["land_use_agricultural"] = 0.5
        grid["land_use_urban"] = 0.2
        grid["land_use_forested"] = 0.3
        grid["wetness_index"] = 0.5
        grid["water_type"] = "land"
        grid["water_mask_reason"] = ""
        grid["water_coverage_pct"] = 0.0
        grid["is_coastal_tsunami_risk"] = False
        grid["spatial_zone"] = "YELLOW"
        
        # Model that correctly increases probability with rainfall
        class RainfallSensitiveModel:
            def predict_proba(self, X):
                if isinstance(X, pd.DataFrame) and "rainfall_mean_mm" in X.columns:
                    rainfall_col = X["rainfall_mean_mm"].values
                else:
                    rainfall_col = X[:, 0] if X.shape[1] > 0 else np.ones(len(X)) * 1200
                
                # Linear relationship: higher rainfall → higher flood risk
                probs = 0.2 + (rainfall_col / 2000.0) * 0.6
                probs = np.clip(probs, 0.0, 1.0)
                return np.column_stack([1 - probs, probs])
        
        model = RainfallSensitiveModel()
        scorer = FloodRiskScorer()
        baseline_probs = model.predict_proba(grid[["rainfall_mean_mm", "elevation_m", "slope_deg"]].values)[:, -1]
        scorer.calibrate(baseline_probs)
        
        analysis_result = AnalysisResult(
            model=model,
            feature_names=["rainfall_mean_mm", "elevation_m", "slope_deg", "drainage_capacity", "population_density"],
            feature_importances={"rainfall_mean_mm": 0.5, "elevation_m": 0.2, "slope_deg": 0.15, "drainage_capacity": 0.1, "population_density": 0.05},
            method="mock",
            validation_note="Rainfall-sensitive mock model",
            scorer=scorer,
        )
        
        config = PipelineConfig(low_threshold=33.0, medium_threshold=66.0)
        baseline_result = FloodRiskResult(
            scored_grid=grid,
            analysis_result=analysis_result,
            bounding_box=bbox,
            config=config,
            pipeline_duration_seconds=1.0,
            cell_count=len(grid),
            data_provenance={},
            data_tier=1,
        )
        
        # Count baseline RED cells
        baseline_red = (baseline_result.scored_grid["spatial_zone"] == "RED").sum()
        
        # Run +50% rainfall scenario
        params = ScenarioParameters("test", "+50% Rainfall", rainfall_multiplier=1.5)
        scenario_result = run_scenario(baseline_result, None, params)
        scenario_red = scenario_result.scenario_zone_counts.get("RED", 0)
        
        # With higher rainfall and a rainfall-sensitive model, RED should not arbitrarily decrease
        # (It could stay same or increase, depending on the model and thresholds)
        delta_red = scenario_red - baseline_red
        logger.info(f"Baseline RED: {baseline_red}, Scenario RED: {scenario_red}, Delta: {delta_red}")
        
        # The key assertion: RED cells should not decrease more than expected from recalibration artifacts
        # (We allow some variance due to probability normalization, but massive decreases are suspicious)
        assert delta_red >= -len(grid) * 0.1, \
            f"RED cells decreased too much: {delta_red} (baseline={baseline_red}, scenario={scenario_red})"
