"""
Unit tests for BoundingBox and PipelineConfig validation.

Covers:
- BoundingBox raises ConfigurationError for invalid longitude order
- BoundingBox raises ConfigurationError for invalid latitude order
- BoundingBox raises ConfigurationError for out-of-range coordinates
- BoundingBox.center returns the correct midpoint
- BoundingBox.area_km2 returns a positive, approximately correct value
- PipelineConfig raises ConfigurationError for non-positive cell_size_meters
- PipelineConfig raises ConfigurationError when low_threshold >= medium_threshold
- PipelineConfig default construction succeeds without error
"""

import math

import pytest

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# BoundingBox — longitude order
# ---------------------------------------------------------------------------


def test_bounding_box_raises_for_invalid_lon_order():
    """min_lon >= max_lon must raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=10.0, min_lat=0.0, max_lon=10.0, max_lat=1.0)

    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=20.0, min_lat=0.0, max_lon=10.0, max_lat=1.0)


# ---------------------------------------------------------------------------
# BoundingBox — latitude order
# ---------------------------------------------------------------------------


def test_bounding_box_raises_for_invalid_lat_order():
    """min_lat >= max_lat must raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=0.0, min_lat=5.0, max_lon=1.0, max_lat=5.0)

    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=0.0, min_lat=10.0, max_lon=1.0, max_lat=5.0)


# ---------------------------------------------------------------------------
# BoundingBox — out-of-range coordinates
# ---------------------------------------------------------------------------


def test_bounding_box_raises_for_out_of_range_lon():
    """min_lon < -180 or max_lon > 180 must raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=-181.0, min_lat=0.0, max_lon=0.0, max_lat=1.0)

    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=181.0, max_lat=1.0)


def test_bounding_box_raises_for_out_of_range_lat():
    """min_lat < -90 or max_lat > 90 must raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=0.0, min_lat=-91.0, max_lon=1.0, max_lat=0.0)

    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=1.0, max_lat=91.0)


# ---------------------------------------------------------------------------
# BoundingBox — center property
# ---------------------------------------------------------------------------


def test_bounding_box_center():
    """Center of BoundingBox(0, 0, 2, 4) should be (2.0, 1.0)."""
    bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=2.0, max_lat=4.0)
    center_lat, center_lon = bbox.center
    assert center_lat == pytest.approx(2.0)
    assert center_lon == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BoundingBox — area_km2 property
# ---------------------------------------------------------------------------


def test_bounding_box_area_km2():
    """
    area_km2 should be positive and approximately correct for a known bbox.

    For BoundingBox(0, 0, 1, 1) at the equator:
      height_km = 1 * 111.32 = 111.32
      width_km  = 1 * 111.32 * cos(0.5°) ≈ 111.32 * 0.99996 ≈ 111.315
      area_km2  ≈ 111.32 * 111.315 ≈ 12392.2
    We allow ±1% tolerance.
    """
    bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=1.0, max_lat=1.0)
    area = bbox.area_km2
    assert area > 0, "area_km2 must be positive"

    center_lat = 0.5
    expected_height = 111.32
    expected_width = 111.32 * math.cos(math.radians(center_lat))
    expected_area = expected_height * expected_width
    assert area == pytest.approx(expected_area, rel=0.01)


# ---------------------------------------------------------------------------
# PipelineConfig — cell_size_meters validation
# ---------------------------------------------------------------------------


def test_pipeline_config_raises_for_zero_cell_size():
    """cell_size_meters=0 must raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        PipelineConfig(cell_size_meters=0)


def test_pipeline_config_raises_for_negative_cell_size():
    """cell_size_meters=-100 must raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        PipelineConfig(cell_size_meters=-100)


# ---------------------------------------------------------------------------
# PipelineConfig — threshold validation
# ---------------------------------------------------------------------------


def test_pipeline_config_raises_for_invalid_thresholds():
    """low_threshold >= medium_threshold must raise ConfigurationError."""
    # Equal thresholds
    with pytest.raises(ConfigurationError):
        PipelineConfig(low_threshold=50.0, medium_threshold=50.0)

    # low > medium
    with pytest.raises(ConfigurationError):
        PipelineConfig(low_threshold=70.0, medium_threshold=40.0)


# ---------------------------------------------------------------------------
# PipelineConfig — valid defaults
# ---------------------------------------------------------------------------


def test_pipeline_config_valid_defaults():
    """Default PipelineConfig() must construct without raising any error."""
    config = PipelineConfig()
    assert config.cell_size_meters == 500.0
    assert config.model_type == "ensemble"
    assert config.low_threshold < config.medium_threshold
