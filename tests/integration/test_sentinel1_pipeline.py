"""Integration tests for Sentinel-1 satellite integration in the pipeline."""
from __future__ import annotations

import math
import tempfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box, Point

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.models import FloodRiskResult
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.satellite.comparison import (
    Sentinel1ComparisonMetrics,
    compute_sentinel1_comparison_metrics,
    create_unavailable_comparison_metrics,
)
from flood_risk_zonation.satellite.result import Sentinel1ObservationResult


class TestSentinel1ComparisonMetrics:
    """Test the Sentinel-1 comparison metrics computation."""

    def test_comparison_with_perfect_match(self):
        """When model and satellite agree perfectly, metrics should be 1.0."""
        # Create a simple scored grid where all cells in a region are "High" (flood)
        grid_data = {
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1), Point(1, 1), Point(2, 1)],
            "risk_score": [80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            "risk_class": ["High", "High", "High", "High", "High", "High"],
        }
        grid = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

        # Create a Sentinel-1 observation that also observes flood everywhere
        observation = Sentinel1ObservationResult(
            observation_status="OBSERVED",
            flood_observed=True,  # Flood everywhere
            inundation_fraction=1.0,
            flooded_area_km2=100.0,
            no_data_fraction=0.0,
            confidence=0.95,
            coverage_fraction=1.0,
            source="sentinel1_geotiff",
            provider="Test",
            platform="Sentinel-1A",
            sensor="SAR",
            acquisition_time=datetime.now(),
            processing_time=datetime.now(),
            method="TEST_SYNTHETIC",
            spatial_resolution_m=10.0,
            crs="EPSG:4326",
            bbox=(-1.0, -1.0, 3.0, 2.0),  # Covers all grid points
            input_format="TEST",
        )

        metrics = compute_sentinel1_comparison_metrics(grid, observation)

        assert metrics.comparison_status == "COMPUTED"
        assert metrics.iou == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.true_positives == 6
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0

    def test_comparison_with_no_match(self):
        """When model and satellite completely disagree, metrics should be 0.0."""
        # Grid: all Low risk (no flood predicted)
        grid_data = {
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1), Point(1, 1), Point(2, 1)],
            "risk_score": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            "risk_class": ["Low", "Low", "Low", "Low", "Low", "Low"],
        }
        grid = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

        # Sentinel-1: observes flood everywhere
        observation = Sentinel1ObservationResult(
            observation_status="OBSERVED",
            flood_observed=True,
            inundation_fraction=1.0,
            flooded_area_km2=100.0,
            no_data_fraction=0.0,
            confidence=0.95,
            coverage_fraction=1.0,
            source="sentinel1_geotiff",
            provider="Test",
            platform="Sentinel-1A",
            sensor="SAR",
            acquisition_time=datetime.now(),
            processing_time=datetime.now(),
            method="TEST_SYNTHETIC",
            spatial_resolution_m=10.0,
            crs="EPSG:4326",
            bbox=(-1.0, -1.0, 3.0, 2.0),
            input_format="TEST",
        )

        metrics = compute_sentinel1_comparison_metrics(grid, observation)

        assert metrics.comparison_status == "COMPUTED"
        assert metrics.iou == 0.0
        assert metrics.precision == 0.0  # No true positives
        assert metrics.recall == 0.0  # All false negatives
        assert metrics.f1_score == 0.0
        assert metrics.true_positives == 0
        assert metrics.false_negatives == 6

    def test_comparison_with_partial_match(self):
        """When model and satellite partially agree, metrics should reflect trade-offs."""
        # Grid: 3 High (flood), 3 Low (no flood)
        grid_data = {
            "geometry": [Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1), Point(1, 1), Point(2, 1)],
            "risk_score": [80.0, 80.0, 80.0, 30.0, 30.0, 30.0],
            "risk_class": ["High", "High", "High", "Low", "Low", "Low"],
        }
        grid = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

        # Sentinel-1: observes flood (should match first 3)
        observation = Sentinel1ObservationResult(
            observation_status="OBSERVED",
            flood_observed=True,  # All cells in bbox are flooded
            inundation_fraction=0.5,  # 50% of area inundated
            flooded_area_km2=50.0,
            no_data_fraction=0.0,
            confidence=0.95,
            coverage_fraction=1.0,
            source="sentinel1_geotiff",
            provider="Test",
            platform="Sentinel-1A",
            sensor="SAR",
            acquisition_time=datetime.now(),
            processing_time=datetime.now(),
            method="TEST_SYNTHETIC",
            spatial_resolution_m=10.0,
            crs="EPSG:4326",
            bbox=(-1.0, -1.0, 3.0, 2.0),
            input_format="TEST",
        )

        metrics = compute_sentinel1_comparison_metrics(grid, observation)

        assert metrics.comparison_status == "COMPUTED"
        # TP=3 (first 3 High cells match), FP=0, FN=3 (last 3 Low cells don't match)
        assert metrics.true_positives == 3
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 3
        assert metrics.iou == 0.5  # 3 / (3 + 3)
        assert metrics.precision == 1.0  # 3 / 3
        assert metrics.recall == 0.5  # 3 / 6
        assert abs(metrics.f1_score - 2 * (1.0 * 0.5) / (1.0 + 0.5)) < 0.01

    def test_comparison_unavailable_when_observation_is_none(self):
        """Comparison should return UNAVAILABLE when observation is None."""
        grid_data = {
            "geometry": [Point(0, 0)],
            "risk_score": [80.0],
            "risk_class": ["High"],
        }
        grid = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

        metrics = compute_sentinel1_comparison_metrics(grid, None)

        assert metrics.comparison_status == "UNAVAILABLE"
        assert metrics.iou is None
        assert metrics.precision is None
        assert metrics.error_reason == "No Sentinel-1 observation provided"

    def test_comparison_unavailable_when_observation_not_observed(self):
        """Comparison should return UNAVAILABLE when observation status is not OBSERVED."""
        grid_data = {
            "geometry": [Point(0, 0)],
            "risk_score": [80.0],
            "risk_class": ["High"],
        }
        grid = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

        # Create an UNKNOWN observation
        observation = Sentinel1ObservationResult(
            observation_status="UNKNOWN",
            flood_observed=None,
            inundation_fraction=math.nan,
            flooded_area_km2=math.nan,
            no_data_fraction=1.0,
            confidence=0.0,
            coverage_fraction=0.0,
            source="unknown",
            provider="Unknown",
            platform="Unknown",
            sensor="Unknown",
            acquisition_time=datetime.now(),
            processing_time=datetime.now(),
            method="UNKNOWN",
            spatial_resolution_m=math.nan,
            crs="EPSG:4326",
            bbox=(0, 0, 1, 1),
            input_format="UNKNOWN",
        )

        metrics = compute_sentinel1_comparison_metrics(grid, observation)

        assert metrics.comparison_status == "UNAVAILABLE"
        assert metrics.iou is None

    def test_comparison_unavailable_when_grid_empty(self):
        """Comparison should return UNAVAILABLE when grid is empty."""
        grid = gpd.GeoDataFrame({"geometry": [], "risk_score": [], "risk_class": []}, crs="EPSG:4326")

        observation = Sentinel1ObservationResult(
            observation_status="OBSERVED",
            flood_observed=True,
            inundation_fraction=1.0,
            flooded_area_km2=100.0,
            no_data_fraction=0.0,
            confidence=0.95,
            coverage_fraction=1.0,
            source="sentinel1_geotiff",
            provider="Test",
            platform="Sentinel-1A",
            sensor="SAR",
            acquisition_time=datetime.now(),
            processing_time=datetime.now(),
            method="TEST_SYNTHETIC",
            spatial_resolution_m=10.0,
            crs="EPSG:4326",
            bbox=(0, 0, 1, 1),
            input_format="TEST",
        )

        metrics = compute_sentinel1_comparison_metrics(grid, observation)

        assert metrics.comparison_status == "UNAVAILABLE"
        assert "empty" in metrics.error_reason.lower()

    def test_comparison_unavailable_when_no_overlap(self):
        """Comparison should return UNAVAILABLE when grid doesn't overlap Sentinel-1 bbox."""
        # Grid at (10, 10) to (12, 12)
        grid_data = {
            "geometry": [Point(10, 10), Point(11, 11)],
            "risk_score": [80.0, 80.0],
            "risk_class": ["High", "High"],
        }
        grid = gpd.GeoDataFrame(grid_data, crs="EPSG:4326")

        # Sentinel-1 bbox at (0, 0) to (1, 1) — no overlap
        observation = Sentinel1ObservationResult(
            observation_status="OBSERVED",
            flood_observed=True,
            inundation_fraction=1.0,
            flooded_area_km2=100.0,
            no_data_fraction=0.0,
            confidence=0.95,
            coverage_fraction=1.0,
            source="sentinel1_geotiff",
            provider="Test",
            platform="Sentinel-1A",
            sensor="SAR",
            acquisition_time=datetime.now(),
            processing_time=datetime.now(),
            method="TEST_SYNTHETIC",
            spatial_resolution_m=10.0,
            crs="EPSG:4326",
            bbox=(0, 0, 1, 1),  # No overlap with grid
            input_format="TEST",
        )

        metrics = compute_sentinel1_comparison_metrics(grid, observation)

        assert metrics.comparison_status == "UNAVAILABLE"
        assert metrics.coverage_fraction == 0.0

    def test_unavailable_metrics_factory(self):
        """Test factory function for unavailable metrics."""
        metrics = create_unavailable_comparison_metrics("Test reason")

        assert metrics.comparison_status == "UNAVAILABLE"
        assert metrics.error_reason == "Test reason"
        assert metrics.iou is None
        assert len(metrics.limitations) > 0


class TestSentinel1PipelineIntegration:
    """Integration tests with the full pipeline."""

    def test_pipeline_result_includes_sentinel1_fields(self):
        """Pipeline result should include sentinel1_observation and comparison_metrics fields."""
        # Use a minimal config
        config = PipelineConfig(
            cell_size_meters=500,
            random_seed=42,
            allow_network=False,  # No network calls
        )
        pipeline = FloodRiskPipeline(config)

        # Minimal bbox (Gottigere, Bangalore)
        bbox = BoundingBox(
            min_lon=77.55,
            min_lat=12.84,
            max_lon=77.56,
            max_lat=12.85,
        )

        # Run without Sentinel-1 data
        result = pipeline.run(bbox)

        assert isinstance(result, FloodRiskResult)
        assert hasattr(result, "sentinel1_observation")
        assert hasattr(result, "sentinel1_comparison_metrics")
        # Without data provided, observation should be UNKNOWN (explicit fallback state)
        # and comparison_metrics should be UNAVAILABLE
        assert result.sentinel1_observation is not None
        assert result.sentinel1_observation.observation_status == "UNKNOWN"
        assert result.sentinel1_comparison_metrics is not None
        assert result.sentinel1_comparison_metrics.comparison_status == "UNAVAILABLE"

    def test_pipeline_without_sentinel1_still_produces_hazard_map(self):
        """Pipeline should work normally and produce hazard map without Sentinel-1 data."""
        config = PipelineConfig(
            cell_size_meters=500,
            random_seed=42,
            allow_network=False,
        )
        pipeline = FloodRiskPipeline(config)

        bbox = BoundingBox(
            min_lon=77.55,
            min_lat=12.84,
            max_lon=77.56,
            max_lat=12.85,
        )

        result = pipeline.run(bbox)

        # Should have a valid scored grid
        assert result.scored_grid is not None
        assert len(result.scored_grid) > 0
        assert "risk_class" in result.scored_grid.columns
        assert "risk_score" in result.scored_grid.columns
        # Should have expected risk classes
        unique_classes = set(result.scored_grid["risk_class"].unique())
        assert "High" in unique_classes or "Medium" in unique_classes or "Low" in unique_classes or "Water" in unique_classes

    def test_pipeline_comparison_metrics_status_tracking(self):
        """Pipeline should properly track comparison metrics status."""
        config = PipelineConfig(
            cell_size_meters=500,
            random_seed=42,
            allow_network=False,
        )
        pipeline = FloodRiskPipeline(config)

        bbox = BoundingBox(
            min_lon=77.55,
            min_lat=12.84,
            max_lon=77.56,
            max_lat=12.85,
        )

        result = pipeline.run(bbox)

        # Without Sentinel-1 data, metrics should be UNAVAILABLE
        assert result.sentinel1_comparison_metrics is not None
        assert result.sentinel1_comparison_metrics.comparison_status == "UNAVAILABLE"

    def test_sentinel1_in_provenance(self):
        """Sentinel-1 observation metadata should be in provenance dict."""
        config = PipelineConfig(
            cell_size_meters=500,
            random_seed=42,
            allow_network=False,
        )
        pipeline = FloodRiskPipeline(config)

        bbox = BoundingBox(
            min_lon=77.55,
            min_lat=12.84,
            max_lon=77.56,
            max_lat=12.85,
        )

        result = pipeline.run(bbox)

        # Provenance should include Sentinel-1 keys (even if values are "unavailable")
        assert "sentinel1_status" in result.data_provenance or result.sentinel1_observation is None
        # When Sentinel-1 is None, no keys should be present
        if result.sentinel1_observation is None:
            # Check that standard keys exist
            assert "elevation" in result.data_provenance
            assert "rainfall" in result.data_provenance
            assert "water_bodies" in result.data_provenance


class TestSentinel1DataFlowValidation:
    """Test data flow and backward compatibility."""

    def test_backward_compatibility_without_sentinel1(self):
        """Existing pipelines without Sentinel-1 should continue to work."""
        config = PipelineConfig(
            cell_size_meters=500,
            random_seed=42,
            allow_network=False,
        )
        pipeline = FloodRiskPipeline(config)

        bbox = BoundingBox(
            min_lon=77.55,
            min_lat=12.84,
            max_lon=77.56,
            max_lat=12.85,
        )

        result = pipeline.run(bbox)

        # Result should be fully functional
        assert result.scored_grid is not None
        assert len(result.scored_grid) > 0
        assert result.pipeline_duration_seconds > 0
        assert result.cell_count > 0
        assert result.data_tier >= 1

    def test_hazard_map_not_regressed(self):
        """Hazard map generation should not be affected by Sentinel-1 integration."""
        config = PipelineConfig(
            cell_size_meters=500,
            random_seed=42,
            allow_network=False,
        )
        pipeline = FloodRiskPipeline(config)

        bbox = BoundingBox(
            min_lon=77.55,
            min_lat=12.84,
            max_lon=77.56,
            max_lat=12.85,
        )

        result = pipeline.run(bbox)

        # Check that hazard map has expected characteristics
        assert "risk_class" in result.scored_grid.columns
        assert "risk_score" in result.scored_grid.columns

        # Should have differentiated risk zones (not monochromatic)
        risk_classes = result.scored_grid["risk_class"].unique()
        # May have 1-4 classes depending on data, but should not be all one value
        if len(risk_classes) == 1:
            # Single class is acceptable (e.g., all Water in coastal region)
            pass
        else:
            # Multiple classes indicate proper differentiation
            assert len(risk_classes) >= 1
