"""
Property 1: Raster Reprojection Preserves Target CRS

Validates: Requirements 1.1
"""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from rasterio.crs import CRS
from rasterio.transform import from_origin

from flood_risk_zonation.models import RasterDataset
from flood_risk_zonation.utils.crs import reproject_raster


def make_synthetic_raster_utm(lat: float, size: int = 10) -> RasterDataset:
    zone = 32
    epsg = 32632 if lat >= 0 else 32732
    crs = CRS.from_epsg(epsg)
    rng = np.random.default_rng(42)
    array = rng.uniform(0, 500, (size, size)).astype(np.float32)
    easting = 500_000.0
    northing = abs(lat) * 111_320.0 if lat >= 0 else (90 - abs(lat)) * 111_320.0
    cell_size = 30.0
    transform = from_origin(easting, northing + size * cell_size, cell_size, cell_size)
    return RasterDataset(array=array, transform=transform, crs=crs, nodata=None, source="synthetic_utm")


@given(lat=st.floats(-60, 60))
@settings(max_examples=20, deadline=None)
def test_reproject_preserves_target_crs(lat):
    """Property 1: After reprojection to WGS84, CRS equals EPSG:4326 and values are finite."""
    raster = make_synthetic_raster_utm(lat)
    reprojected = reproject_raster(raster, "EPSG:4326")
    assert reprojected.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {reprojected.crs}"
    finite_values = reprojected.array[~np.isnan(reprojected.array)]
    assert np.all(np.isfinite(finite_values))
