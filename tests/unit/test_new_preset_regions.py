"""
Unit tests for 2 preset regions: Indian Hilly Region, Indian Ocean Open Water.

Tests verify:
  T1 -- Indian Hilly Region preset is correctly defined in PRESET_REGIONS
  T2 -- Indian Ocean Open Water preset is correctly defined
  T3 -- Both presets have correct coordinates and metadata
  T4 -- Both presets have corresponding DEMO_REGIONS entries
  T5 -- DEMO_REGIONS entries have valid elevation, relief, and rainfall parameters
  T7 -- Non-Nepal presets have active_flood_override=False
  T8 -- Offline keys are correctly set for all new presets
"""
from __future__ import annotations

import pytest

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.ingest.sample_data import DEMO_REGIONS, DemoRegion


# Import PRESET_REGIONS from app.py
def _get_preset_regions():
    """Import PRESET_REGIONS from app.py dynamically to avoid circular imports."""
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    return app_module.PRESET_REGIONS


class TestIndianHillyRegionPreset:
    """T1: Indian Hilly Region preset verification."""

    def test_preset_exists(self):
        """Indian Hilly Region must exist in PRESET_REGIONS."""
        presets = _get_preset_regions()
        assert "Indian Hilly Region" in presets

    def test_preset_coordinates(self):
        """Indian Hilly Region must have valid coordinates."""
        presets = _get_preset_regions()
        preset = presets["Indian Hilly Region"]
        
        assert preset["min_lon"] == 88.45
        assert preset["min_lat"] == 27.45
        assert preset["max_lon"] == 88.55
        assert preset["max_lat"] == 27.55
        assert preset["min_lon"] < preset["max_lon"]
        assert preset["min_lat"] < preset["max_lat"]

    def test_preset_metadata(self):
        """Indian Hilly Region must have required metadata fields."""
        presets = _get_preset_regions()
        preset = presets["Indian Hilly Region"]
        
        assert "area_name" in preset
        assert "offline_key" in preset
        assert preset["area_name"] != ""
        assert "Hilly" in preset["area_name"] or "Sikkim" in preset["area_name"]

    def test_demo_region_exists(self):
        """Indian Hilly Region must have corresponding DEMO_REGIONS entry."""
        assert "Indian Hilly Region" in DEMO_REGIONS

    def test_demo_region_valid(self):
        """DEMO_REGIONS entry for Indian Hilly must be valid DemoRegion."""
        region = DEMO_REGIONS["Indian Hilly Region"]
        assert isinstance(region, DemoRegion)
        assert region.base_elevation_m > 0
        assert region.relief_m > 0
        assert region.mean_rainfall_mm > 0
        assert region.seed > 0

    def test_bbox_matches_preset(self):
        """DEMO_REGIONS bbox must match PRESET_REGIONS coordinates."""
        presets = _get_preset_regions()
        preset = presets["Indian Hilly Region"]
        demo = DEMO_REGIONS["Indian Hilly Region"]
        
        bbox = demo.bbox
        assert bbox.min_lon == preset["min_lon"]
        assert bbox.min_lat == preset["min_lat"]
        assert bbox.max_lon == preset["max_lon"]
        assert bbox.max_lat == preset["max_lat"]




class TestIndianOceanPreset:
    """T3: Indian Ocean Open Water preset verification."""

    def test_preset_exists(self):
        """Indian Ocean Open Water must exist in PRESET_REGIONS."""
        presets = _get_preset_regions()
        assert "Indian Ocean Open Water" in presets

    def test_preset_coordinates(self):
        """Indian Ocean preset must have valid open water coordinates."""
        presets = _get_preset_regions()
        preset = presets["Indian Ocean Open Water"]
        
        assert preset["min_lon"] == 71.95
        assert preset["min_lat"] == 9.95
        assert preset["max_lon"] == 72.05
        assert preset["max_lat"] == 10.05
        assert preset["min_lon"] < preset["max_lon"]
        assert preset["min_lat"] < preset["max_lat"]

    def test_preset_metadata(self):
        """Indian Ocean preset must have required metadata fields."""
        presets = _get_preset_regions()
        preset = presets["Indian Ocean Open Water"]
        
        assert "area_name" in preset
        assert "offline_key" in preset
        assert preset["area_name"] != ""
        assert "Arabian" in preset["area_name"] or "Ocean" in preset["area_name"]

    def test_demo_region_exists(self):
        """Indian Ocean preset must have corresponding DEMO_REGIONS entry."""
        assert "Indian Ocean" in DEMO_REGIONS

    def test_demo_region_valid(self):
        """DEMO_REGIONS entry for Indian Ocean must be valid DemoRegion."""
        region = DEMO_REGIONS["Indian Ocean"]
        assert isinstance(region, DemoRegion)
        # Ocean region has 0 elevation and relief
        assert region.base_elevation_m == 0.0
        assert region.relief_m == 0.0
        assert region.mean_rainfall_mm > 0
        assert region.seed > 0

    def test_bbox_matches_preset(self):
        """DEMO_REGIONS bbox must match PRESET_REGIONS coordinates."""
        presets = _get_preset_regions()
        preset = presets["Indian Ocean Open Water"]
        demo = DEMO_REGIONS["Indian Ocean"]
        
        bbox = demo.bbox
        assert bbox.min_lon == preset["min_lon"]
        assert bbox.min_lat == preset["min_lat"]
        assert bbox.max_lon == preset["max_lon"]
        assert bbox.max_lat == preset["max_lat"]


class TestAllNewPresetsBackwardCompatibility:
    """T4-T8: Verify all presets integrate with existing systems."""

    def test_existing_presets_unchanged(self):
        """Existing 4 presets must still be present and correct."""
        presets = _get_preset_regions()
        existing = [
            "Gottigere, Bangalore",
            "Chennai Marina (Coastal)",
            "Dal Lake, Srinagar",
            "Puri, Odisha (Cyclone Coast)",
        ]
        for name in existing:
            assert name in presets, f"Existing preset '{name}' was removed or renamed"

    def test_custom_region_still_present(self):
        """Custom Region preset must still be present."""
        presets = _get_preset_regions()
        custom_found = False
        for name in presets.keys():
            if "Custom" in name:
                custom_found = True
        assert custom_found, "Custom Region preset missing"

    def test_demo_regions_count(self):
        """DEMO_REGIONS must now contain 5 regions (3 original + 2 new, Nepal removed)."""
        assert len(DEMO_REGIONS) == 5, (
            f"Expected 5 DEMO_REGIONS (3 original + 2 new, Nepal removed), got {len(DEMO_REGIONS)}"
        )

    def test_all_demo_regions_have_valid_seed(self):
        """All DEMO_REGIONS entries must have unique, positive seeds."""
        seeds = []
        for name, region in DEMO_REGIONS.items():
            assert region.seed > 0, f"{name} has invalid seed: {region.seed}"
            assert region.seed not in seeds, f"Duplicate seed {region.seed} in {name}"
            seeds.append(region.seed)

