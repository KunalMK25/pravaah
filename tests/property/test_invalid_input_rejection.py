"""
Property 17: Invalid Input Rejection

**Validates: Requirements 7.3, 8.1, 8.2**
"""
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import pytest
from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exceptions import ConfigurationError


@given(
    lon=st.floats(-180, 180),
    min_lat=st.floats(-89, 89),
    max_lat=st.floats(-89, 90),
    delta=st.floats(0, 10)
)
@settings(max_examples=20)
def test_invalid_bbox_lon_order_raises(lon, min_lat, max_lat, delta):
    assume(min_lat < max_lat)
    max_lon = lon - delta
    assume(-180 <= max_lon <= 180)
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


@given(
    min_lon=st.floats(-179, 179),
    lat=st.floats(-90, 90),
    max_lon_delta=st.floats(0.001, 10),
    delta=st.floats(0, 10)
)
@settings(max_examples=20)
def test_invalid_bbox_lat_order_raises(min_lon, lat, max_lon_delta, delta):
    max_lon = min_lon + max_lon_delta
    assume(max_lon <= 180)
    max_lat = lat - delta
    assume(-90 <= max_lat <= 90)
    assume(-90 <= lat <= 90)
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=min_lon, min_lat=lat, max_lon=max_lon, max_lat=max_lat)


@given(
    min_lon=st.floats(-1000, -180.001),
    min_lat=st.floats(-89, 0),
    max_lat=st.floats(0.001, 89)
)
@settings(max_examples=20)
def test_invalid_bbox_min_lon_out_of_range(min_lon, min_lat, max_lat):
    assume(min_lat < max_lat)
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=min_lon, min_lat=min_lat, max_lon=0.0, max_lat=max_lat)


@given(
    max_lon=st.floats(180.001, 1000),
    min_lat=st.floats(-89, 0),
    max_lat=st.floats(0.001, 89)
)
@settings(max_examples=20)
def test_invalid_bbox_max_lon_out_of_range(max_lon, min_lat, max_lat):
    assume(min_lat < max_lat)
    with pytest.raises(ConfigurationError):
        BoundingBox(min_lon=0.0, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat)


@given(cell_size=st.floats(-10000, 0))
@settings(max_examples=20)
def test_invalid_config_cell_size_raises(cell_size):
    with pytest.raises(ConfigurationError):
        PipelineConfig(cell_size_meters=cell_size)


@given(
    low=st.floats(0.1, 99.9),
    delta=st.floats(0, 50)
)
@settings(max_examples=20)
def test_invalid_config_threshold_order_raises(low, delta):
    medium = low - delta
    assume(medium >= 0)
    with pytest.raises(ConfigurationError):
        PipelineConfig(low_threshold=low, medium_threshold=medium)
