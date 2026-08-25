"""
Tests for bounding-box size validation (Item 1) and pipeline edge cases (Item 2).
"""
from __future__ import annotations

import pytest

from flood_risk_zonation.config import (
    BoundingBox,
    PipelineConfig,
    validate_bbox_size,
    BBOX_MIN_SIDE_KM,
    BBOX_MAX_SIDE_KM,
)
from flood_risk_zonation.pipeline import FloodRiskPipeline


# ── Item 1: bbox size limits ──────────────────────────────────────────────────

class TestBboxSizeValidation:

    def test_normal_bbox_passes(self):
        """A typical 7 km × 8 km bbox must return None (no error)."""
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.91)  # Bangalore ~7×8 km
        assert validate_bbox_size(bbox) is None

    def test_too_small_width_rejected(self):
        """A bbox narrower than BBOX_MIN_SIDE_KM must return an error string."""
        # ~0.009° lon ≈ ~1 km at 13°N — well below the 2 km min
        bbox = BoundingBox(77.55, 12.84, 77.559, 12.91)
        err = validate_bbox_size(bbox)
        assert err is not None
        assert "too small" in err.lower()

    def test_too_small_height_rejected(self):
        """A bbox shorter than BBOX_MIN_SIDE_KM must return an error string."""
        bbox = BoundingBox(77.55, 12.84, 77.62, 12.849)
        err = validate_bbox_size(bbox)
        assert err is not None
        assert "too small" in err.lower()

    def test_too_large_width_rejected(self):
        """A bbox wider than BBOX_MAX_SIDE_KM must return an error string."""
        # ~0.9° lon ≈ ~100 km at 13°N — well above the 50 km max
        bbox = BoundingBox(77.0, 12.84, 77.9, 12.91)
        err = validate_bbox_size(bbox)
        assert err is not None
        assert "too large" in err.lower()

    def test_too_large_height_rejected(self):
        """A bbox taller than BBOX_MAX_SIDE_KM must return an error string."""
        bbox = BoundingBox(77.55, 12.0, 77.62, 12.9)  # ~100 km tall
        err = validate_bbox_size(bbox)
        assert err is not None
        assert "too large" in err.lower()

    def test_minimum_boundary_accepted(self):
        """A bbox exactly at the minimum size (2 km per side) must pass."""
        # 2 km / 111.32 ≈ 0.01797° — use 0.018° to be safely above
        deg = 2.0 / 111.32
        bbox = BoundingBox(77.55, 12.84, 77.55 + deg + 0.001, 12.84 + deg + 0.001)
        assert validate_bbox_size(bbox) is None

    def test_maximum_boundary_accepted(self):
        """A bbox just under the max size (50 km per side) must pass."""
        deg_lat = 49.0 / 111.32           # just under 50 km tall
        deg_lon = 49.0 / (111.32 * 0.97)  # conservative: cos(14°) ≈ 0.97
        bbox = BoundingBox(77.0, 12.0, 77.0 + deg_lon, 12.0 + deg_lat)
        assert validate_bbox_size(bbox) is None

    def test_error_message_includes_dimensions(self):
        """Error message must mention the actual dimensions."""
        bbox = BoundingBox(77.55, 12.84, 77.559, 12.91)
        err = validate_bbox_size(bbox)
        assert err is not None
        # Should include km values
        assert "km" in err

    def test_polar_bbox_uses_latitude_aware_conversion(self):
        """
        The same degree span covers less km near the poles.
        A 0.5° × 0.5° bbox near the equator is ~55 km — too large.
        A 0.5° × 0.5° bbox at 80° latitude is ~10 km wide — should be accepted.
        """
        # Near equator: 0.5° lon ≈ 55 km — too large
        bbox_eq = BoundingBox(0.0, 0.0, 0.5, 0.5)
        err_eq = validate_bbox_size(bbox_eq)
        assert err_eq is not None and "too large" in err_eq.lower()

        # Near pole (80°): cos(80°) ≈ 0.174 → 0.5° lon ≈ 9.7 km — should be OK
        bbox_polar = BoundingBox(0.0, 80.0, 0.5, 80.09)  # ~0.09° lat ≈ 10 km tall
        err_polar = validate_bbox_size(bbox_polar)
        assert err_polar is None, (
            f"Polar bbox should pass but got: {err_polar}"
        )


# ── Item 2: pipeline edge cases ───────────────────────────────────────────────

class TestPipelineEdgeCases:

    _config = PipelineConfig(
        cell_size_meters=5000,
        use_cache=False,
        allow_network=False,
        random_seed=42,
    )

    def test_ocean_bbox_completes_without_crash(self):
        """
        A bbox entirely over open ocean should complete and return a valid result.
        All cells should be Water (elevation-based mask fires for SRTM 0 values)
        or at least the pipeline must not raise.
        """
        # Mid-Indian Ocean — no land, no SRTM coverage expected
        bbox = BoundingBox(min_lon=72.0, min_lat=-5.0, max_lon=72.45, max_lat=-4.55)
        pipeline = FloodRiskPipeline(self._config)
        result = pipeline.run(bbox)
        assert result is not None
        assert result.cell_count > 0
        # Should not raise — result can have any valid distribution
        assert set(result.scored_grid["risk_class"].unique()).issubset(
            {"Low", "Medium", "High", "Water"}
        )

    def test_no_srtm_coverage_falls_back_to_synthetic(self):
        """
        A bbox with no local SRTM GeoTIFF coverage must fall back to synthetic
        elevation gracefully (no DataIngestionError propagated to caller).
        """
        # Arbitrary location with no local GeoTIFF — will always fall back
        bbox = BoundingBox(min_lon=10.0, min_lat=50.0, max_lon=10.45, max_lat=50.45)
        pipeline = FloodRiskPipeline(self._config)
        result = pipeline.run(bbox)
        assert result is not None
        assert result.data_provenance.get("elevation") == "synthetic"

    def test_no_population_data_falls_back_to_synthetic(self):
        """
        Population data fallback must be transparent — result must still have
        population_density column populated with sensible values.
        """
        bbox = BoundingBox(min_lon=77.55, min_lat=12.84, max_lon=77.62, max_lat=12.91)
        pipeline = FloodRiskPipeline(self._config)
        result = pipeline.run(bbox)
        assert "population_density" in result.scored_grid.columns
        assert result.scored_grid["population_density"].isna().sum() == 0

    def test_repeated_runs_are_identical(self):
        """
        Running the same bbox twice must produce bit-for-bit identical
        risk_class and risk_score columns. No randomness allowed.
        """
        bbox = BoundingBox(min_lon=77.55, min_lat=12.84, max_lon=77.62, max_lat=12.91)
        pipeline = FloodRiskPipeline(self._config)

        result1 = pipeline.run(bbox)
        result2 = pipeline.run(bbox)

        scores1 = result1.scored_grid["risk_score"].values
        scores2 = result2.scored_grid["risk_score"].values
        classes1 = result1.scored_grid["risk_class"].values
        classes2 = result2.scored_grid["risk_class"].values

        assert (scores1 == scores2).all(), "risk_score differs between runs"
        assert (classes1 == classes2).all(), "risk_class differs between runs"

    def test_result_has_is_coastal_tsunami_risk_column(self):
        """is_coastal_tsunami_risk must always be present, even with no water bodies."""
        bbox = BoundingBox(min_lon=77.55, min_lat=12.84, max_lon=77.62, max_lat=12.91)
        pipeline = FloodRiskPipeline(self._config)
        result = pipeline.run(bbox)
        assert "is_coastal_tsunami_risk" in result.scored_grid.columns
