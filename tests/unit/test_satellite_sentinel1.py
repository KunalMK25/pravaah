"""Unit tests for Sentinel-1 satellite flood observation integration."""
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
from shapely.geometry import box

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.satellite.observations import (
    RasterFloodMaskProvider,
    VectorFloodPolygonProvider,
)
from flood_risk_zonation.satellite.provider import UnknownSentinel1Provider
from flood_risk_zonation.satellite.result import (
    Sentinel1ObservationResult,
    create_unavailable_sentinel1_result,
    create_unknown_sentinel1_result,
)
from flood_risk_zonation.satellite.sentinel1 import load_sentinel1_observation


class TestSentinel1ResultModel:
    """Test Sentinel1ObservationResult invariants and validation."""

    def test_result_valid_observed(self):
        """Valid OBSERVED result should not raise."""
        result = Sentinel1ObservationResult(
            observation_status="OBSERVED",
            flood_observed=True,
            inundation_fraction=0.35,
            flooded_area_km2=125.5,
            no_data_fraction=0.05,
            confidence=0.85,
            coverage_fraction=0.95,
            source="sentinel1_geotiff",
            provider="Local",
            platform="Sentinel-1A",
            sensor="SAR",
            acquisition_time=datetime.now(),
            processing_time=datetime.now(),
            method="DERIVED_FLOOD_MASK",
            spatial_resolution_m=10.0,
            crs="EPSG:4326",
            bbox=(77.5, 12.8, 77.7, 13.0),
            input_format="GeoTIFF",
        )
        assert result.observation_status == "OBSERVED"
        assert result.flood_observed is True

    def test_result_invalid_confidence(self):
        """Invalid confidence (outside [0,1]) should raise."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            Sentinel1ObservationResult(
                observation_status="OBSERVED",
                flood_observed=True,
                inundation_fraction=0.5,
                flooded_area_km2=100.0,
                no_data_fraction=0.0,
                confidence=1.5,  # Invalid
                coverage_fraction=1.0,
                source="test",
                provider="Test",
                platform="Test",
                sensor="Test",
                acquisition_time=datetime.now(),
                processing_time=datetime.now(),
                method="TEST",
                spatial_resolution_m=10.0,
                crs="EPSG:4326",
                bbox=(0, 0, 1, 1),
                input_format="TEST",
            )

    def test_result_invalid_inundation(self):
        """Invalid inundation fraction should raise."""
        with pytest.raises(ValueError, match="Inundation fraction"):
            Sentinel1ObservationResult(
                observation_status="OBSERVED",
                flood_observed=True,
                inundation_fraction=1.5,  # Invalid
                flooded_area_km2=100.0,
                no_data_fraction=0.0,
                confidence=0.85,
                coverage_fraction=1.0,
                source="test",
                provider="Test",
                platform="Test",
                sensor="Test",
                acquisition_time=datetime.now(),
                processing_time=datetime.now(),
                method="TEST",
                spatial_resolution_m=10.0,
                crs="EPSG:4326",
                bbox=(0, 0, 1, 1),
                input_format="TEST",
            )

    def test_result_scientific_integrity_unknown_not_false(self):
        """UNKNOWN observation cannot have flood_observed=False."""
        with pytest.raises(ValueError, match="Cannot mark UNKNOWN"):
            Sentinel1ObservationResult(
                observation_status="UNKNOWN",
                flood_observed=False,  # Invalid: UNKNOWN should have None
                inundation_fraction=math.nan,
                flooded_area_km2=math.nan,
                no_data_fraction=1.0,
                confidence=0.0,
                coverage_fraction=0.0,
                source="test",
                provider="Test",
                platform="Test",
                sensor="Test",
                acquisition_time=datetime.now(),
                processing_time=datetime.now(),
                method="TEST",
                spatial_resolution_m=math.nan,
                crs="EPSG:4326",
                bbox=(0, 0, 1, 1),
                input_format="TEST",
            )

    def test_result_unknown_state(self):
        """UNKNOWN state should be valid."""
        result = create_unknown_sentinel1_result((77.5, 12.8, 77.7, 13.0))
        assert result.observation_status == "UNKNOWN"
        assert result.flood_observed is None
        assert result.confidence == 0.0
        assert result.coverage_fraction == 0.0

    def test_result_unavailable_state(self):
        """UNAVAILABLE state should be valid."""
        result = create_unavailable_sentinel1_result((77.5, 12.8, 77.7, 13.0))
        assert result.observation_status == "UNAVAILABLE"
        assert result.flood_observed is None
        assert result.confidence == 0.0


class TestRasterFloodMaskProvider:
    """Test GeoTIFF raster flood mask loading."""

    def _create_test_geotiff(self, tmp_path: Path, binary: bool = True) -> Path:
        """Create a test GeoTIFF file."""
        output_file = tmp_path / "test_flood.tif"
        
        # Create binary flood mask (0 = no flood, 1 = flood)
        if binary:
            data = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
        else:
            # Continuous inundation fraction
            data = np.array([[0.0, 0.2, 0.8], [0.1, 0.5, 0.9], [0.7, 0.9, 1.0]], dtype=np.float32)
        
        bbox = (77.5, 12.8, 77.7, 13.0)
        transform = from_bounds(*bbox, data.shape[1], data.shape[0])
        
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=255 if binary else -9999,
        ) as dst:
            dst.write(data, 1)
        
        return output_file

    def test_load_binary_geotiff(self, tmp_path):
        """Load binary flood mask from GeoTIFF."""
        geotiff_path = self._create_test_geotiff(tmp_path, binary=True)
        provider = RasterFloodMaskProvider(geotiff_path)
        bbox = (77.5, 12.8, 77.7, 13.0)
        
        result = provider.load_observation(bbox)
        
        assert result.observation_status == "OBSERVED"
        assert result.source == "sentinel1_geotiff"
        assert result.method == "DERIVED_FLOOD_MASK"
        assert isinstance(result.inundation_fraction, float)
        assert 0.0 <= result.inundation_fraction <= 1.0
        assert result.confidence >= 0.7

    def test_load_continuous_geotiff(self, tmp_path):
        """Load continuous inundation fraction from GeoTIFF."""
        geotiff_path = self._create_test_geotiff(tmp_path, binary=False)
        provider = RasterFloodMaskProvider(geotiff_path)
        bbox = (77.5, 12.8, 77.7, 13.0)
        
        result = provider.load_observation(bbox)
        
        assert result.observation_status == "OBSERVED"
        assert isinstance(result.inundation_fraction, float)
        assert 0.0 <= result.inundation_fraction <= 1.0

    def test_geotiff_not_found(self):
        """Nonexistent GeoTIFF should return UNKNOWN."""
        provider = RasterFloodMaskProvider("/nonexistent/path.tif")
        bbox = (77.5, 12.8, 77.7, 13.0)
        
        result = provider.load_observation(bbox)
        
        assert result.observation_status == "UNKNOWN"
        assert result.flood_observed is None


class TestVectorFloodPolygonProvider:
    """Test GeoJSON flood polygon loading."""

    def _create_test_geojson(self, tmp_path: Path) -> Path:
        """Create a test GeoJSON file with flood polygons."""
        output_file = tmp_path / "test_floods.geojson"
        
        # Create flood polygons
        geometries = [
            box(77.52, 12.82, 77.55, 12.85),
            box(77.60, 12.90, 77.65, 12.95),
        ]
        
        gdf = gpd.GeoDataFrame(
            {"flood_type": ["observed", "observed"], "confidence": [0.85, 0.90]},
            geometry=geometries,
            crs="EPSG:4326",
        )
        
        gdf.to_file(output_file, driver="GeoJSON")
        return output_file

    def test_load_geojson_polygons(self, tmp_path):
        """Load flood polygons from GeoJSON."""
        geojson_path = self._create_test_geojson(tmp_path)
        provider = VectorFloodPolygonProvider(geojson_path)
        bbox = (77.5, 12.8, 77.7, 13.0)
        
        result = provider.load_observation(bbox)
        
        assert result.observation_status == "OBSERVED"
        assert result.source == "sentinel1_geojson"
        assert result.method == "VECTOR_POLYGONS"
        assert result.inundation_fraction > 0.0
        assert result.flooded_area_km2 > 0.0
        assert result.coverage_fraction == 1.0

    def test_geojson_not_found(self):
        """Nonexistent GeoJSON should return UNKNOWN."""
        provider = VectorFloodPolygonProvider("/nonexistent/path.geojson")
        bbox = (77.5, 12.8, 77.7, 13.0)
        
        result = provider.load_observation(bbox)
        
        assert result.observation_status == "UNKNOWN"


class TestUnknownSentinel1Provider:
    """Test terminal UNKNOWN provider."""

    def test_always_returns_unknown(self):
        """UnknownProvider should always return UNKNOWN."""
        provider = UnknownSentinel1Provider()
        bbox = (77.5, 12.8, 77.7, 13.0)
        
        result = provider.load_observation(bbox)
        
        assert result.observation_status == "UNKNOWN"
        assert result.flood_observed is None
        assert result.confidence == 0.0


class TestSentinel1Integration:
    """Test end-to-end Sentinel-1 integration."""

    def _create_test_geotiff(self, tmp_path: Path) -> Path:
        """Create test GeoTIFF."""
        output_file = tmp_path / "test_flood.tif"
        data = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        bbox = (77.5, 12.8, 77.7, 13.0)
        transform = from_bounds(*bbox, data.shape[1], data.shape[0])
        
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=np.uint8,
            crs=CRS.from_epsg(4326),
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        
        return output_file

    def test_load_observation_geotiff(self, tmp_path):
        """Load Sentinel-1 observation from GeoTIFF."""
        geotiff_path = self._create_test_geotiff(tmp_path)
        bbox = BoundingBox(77.5, 12.8, 77.7, 13.0)
        
        result = load_sentinel1_observation(bbox, geotiff_path=geotiff_path)
        
        assert result.observation_status == "OBSERVED"
        assert result.source == "sentinel1_geotiff"

    def test_load_observation_no_file(self):
        """Load with no files should return UNKNOWN."""
        bbox = BoundingBox(77.5, 12.8, 77.7, 13.0)
        
        result = load_sentinel1_observation(bbox)
        
        assert result.observation_status == "UNKNOWN"
        assert result.flood_observed is None


class TestSentinel1ScientificIntegrity:
    """Test scientific integrity guarantees."""

    def test_no_fabrication_unknown_status(self):
        """UNKNOWN status must not contain fabricated flood measurements."""
        result = create_unknown_sentinel1_result((77.5, 12.8, 77.7, 13.0))
        
        assert result.flood_observed is None
        assert math.isnan(result.inundation_fraction)
        assert math.isnan(result.flooded_area_km2)
        assert result.confidence == 0.0

    def test_no_fabrication_unavailable_status(self):
        """UNAVAILABLE status must not contain fabricated flood measurements."""
        result = create_unavailable_sentinel1_result((77.5, 12.8, 77.7, 13.0))
        
        assert result.flood_observed is None
        assert math.isnan(result.inundation_fraction)
        assert result.confidence == 0.0

    def test_provenance_preserved(self):
        """Provenance must be completely preserved."""
        result = Sentinel1ObservationResult(
            observation_status="OBSERVED",
            flood_observed=True,
            inundation_fraction=0.5,
            flooded_area_km2=100.0,
            no_data_fraction=0.01,
            confidence=0.85,
            coverage_fraction=0.99,
            source="sentinel1_geotiff",
            provider="Local",
            platform="Sentinel-1A",
            sensor="SAR",
            acquisition_time=datetime(2024, 8, 26, 12, 30, 0),
            processing_time=datetime(2024, 8, 27, 10, 0, 0),
            method="DERIVED_FLOOD_MASK",
            spatial_resolution_m=10.0,
            crs="EPSG:4326",
            bbox=(77.5, 12.8, 77.7, 13.0),
            input_format="GeoTIFF",
            limitations=["Test observation"],
        )
        
        assert result.source == "sentinel1_geotiff"
        assert result.platform == "Sentinel-1A"
        assert result.method == "DERIVED_FLOOD_MASK"
        assert result.acquisition_time == datetime(2024, 8, 26, 12, 30, 0)
        assert len(result.limitations) > 0
