"""
Unit tests for 3 new preset regions: Indian Hilly Region, Nepal Flood-Affected Area, Indian Ocean Open Water.

Tests verify:
  T1 -- Indian Hilly Region preset is correctly defined in PRESET_REGIONS
  T2 -- Nepal Flood-Affected Area preset includes active_flood_override=True
  T3 -- Indian Ocean Open Water preset is correctly defined
  T4 -- All 3 new presets have correct coordinates and metadata
  T5 -- All 3 new presets have corresponding DEMO_REGIONS entries
  T6 -- DEMO_REGIONS entries have valid elevation, relief, and rainfall parameters
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
        assert "Indian Hilly Region (Sikkim)" in presets

    def test_preset_coordinates(self):
        """Indian Hilly Region must have valid coordinates."""
        presets = _get_preset_regions()
        preset = presets["Indian Hilly Region (Sikkim)"]
        
        assert preset["min_lon"] == 87.50
        assert preset["min_lat"] == 27.40
        assert preset["max_lon"] == 88.57
        assert preset["max_lat"] == 28.47
        assert preset["min_lon"] < preset["max_lon"]
        assert preset["min_lat"] < preset["max_lat"]

    def test_preset_metadata(self):
        """Indian Hilly Region must have required metadata fields."""
        presets = _get_preset_regions()
        preset = presets["Indian Hilly Region (Sikkim)"]
        
        assert "area_name" in preset
        assert "offline_key" in preset
        assert "active_flood_override" in preset
        assert preset["area_name"] != ""
        assert "Sikkim" in preset["area_name"] or "Himalayan" in preset["area_name"]

    def test_preset_override_disabled(self):
        """Indian Hilly Region must NOT have active_flood_override enabled."""
        presets = _get_preset_regions()
        preset = presets["Indian Hilly Region (Sikkim)"]
        assert preset["active_flood_override"] is False

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
        preset = presets["Indian Hilly Region (Sikkim)"]
        demo = DEMO_REGIONS["Indian Hilly Region"]
        
        bbox = demo.bbox
        assert bbox.min_lon == preset["min_lon"]
        assert bbox.min_lat == preset["min_lat"]
        assert bbox.max_lon == preset["max_lon"]
        assert bbox.max_lat == preset["max_lat"]


class TestNepalFloodAffectedPreset:
    """T2: Nepal Recent Flood-Affected Area preset with override verification."""

    def test_preset_exists(self):
        """Nepal Flood-Affected Area must exist in PRESET_REGIONS."""
        presets = _get_preset_regions()
        assert "Nepal Recent Flood-Affected Area" in presets

    def test_preset_coordinates(self):
        """Nepal preset must have valid Rasuwa District coordinates."""
        presets = _get_preset_regions()
        preset = presets["Nepal Recent Flood-Affected Area"]
        
        assert preset["min_lon"] == 85.40
        assert preset["min_lat"] == 28.10
        assert preset["max_lon"] == 86.47
        assert preset["max_lat"] == 29.17
        assert preset["min_lon"] < preset["max_lon"]
        assert preset["min_lat"] < preset["max_lat"]

    def test_preset_metadata(self):
        """Nepal preset must have required metadata fields."""
        presets = _get_preset_regions()
        preset = presets["Nepal Recent Flood-Affected Area"]
        
        assert "area_name" in preset
        assert "offline_key" in preset
        assert "active_flood_override" in preset
        assert preset["area_name"] != ""
        assert "Rasuwa" in preset["area_name"] or "Bhote" in preset["area_name"]

    def test_override_enabled(self):
        """Nepal preset MUST have active_flood_override=True."""
        presets = _get_preset_regions()
        preset = presets["Nepal Recent Flood-Affected Area"]
        assert preset["active_flood_override"] is True, (
            "Nepal preset must have active_flood_override=True for "
            "developer-authorized flood override behavior"
        )

    def test_demo_region_exists(self):
        """Nepal preset must have corresponding DEMO_REGIONS entry."""
        assert "Nepal Flood Area" in DEMO_REGIONS

    def test_demo_region_valid(self):
        """DEMO_REGIONS entry for Nepal must be valid DemoRegion."""
        region = DEMO_REGIONS["Nepal Flood Area"]
        assert isinstance(region, DemoRegion)
        assert region.base_elevation_m > 0
        assert region.relief_m > 0
        assert region.mean_rainfall_mm > 0
        assert region.seed > 0

    def test_bbox_matches_preset(self):
        """DEMO_REGIONS bbox must match PRESET_REGIONS coordinates."""
        presets = _get_preset_regions()
        preset = presets["Nepal Recent Flood-Affected Area"]
        demo = DEMO_REGIONS["Nepal Flood Area"]
        
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
        
        assert preset["min_lon"] == 71.50
        assert preset["min_lat"] == 9.50
        assert preset["max_lon"] == 72.57
        assert preset["max_lat"] == 10.57
        assert preset["min_lon"] < preset["max_lon"]
        assert preset["min_lat"] < preset["max_lat"]

    def test_preset_metadata(self):
        """Indian Ocean preset must have required metadata fields."""
        presets = _get_preset_regions()
        preset = presets["Indian Ocean Open Water"]
        
        assert "area_name" in preset
        assert "offline_key" in preset
        assert "active_flood_override" in preset
        assert preset["area_name"] != ""
        assert "Arabian" in preset["area_name"] or "Ocean" in preset["area_name"]

    def test_preset_override_disabled(self):
        """Indian Ocean preset must NOT have active_flood_override enabled."""
        presets = _get_preset_regions()
        preset = presets["Indian Ocean Open Water"]
        assert preset["active_flood_override"] is False

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
    """T4-T8: Verify all 3 new presets integrate with existing systems."""

    def test_all_presets_have_override_field(self):
        """All presets (old + new) must have active_flood_override field."""
        presets = _get_preset_regions()
        for name, preset in presets.items():
            assert "active_flood_override" in preset, (
                f"Preset '{name}' missing active_flood_override field"
            )

    def test_only_nepal_has_override_enabled(self):
        """Only Nepal preset should have active_flood_override=True."""
        presets = _get_preset_regions()
        nepal_found = False
        for name, preset in presets.items():
            if preset.get("active_flood_override") is True:
                assert "Nepal" in name, (
                    f"Found active_flood_override=True on non-Nepal preset: {name}"
                )
                nepal_found = True
        assert nepal_found, "Nepal preset with override=True not found"

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
            assert presets[name]["active_flood_override"] is False

    def test_custom_region_still_present(self):
        """Custom Region preset must still be present."""
        presets = _get_preset_regions()
        custom_found = False
        for name in presets.keys():
            if "Custom" in name:
                custom_found = True
                assert presets[name]["active_flood_override"] is False
        assert custom_found, "Custom Region preset missing"

    def test_demo_regions_count(self):
        """DEMO_REGIONS must now contain 6 regions (3 original + 3 new)."""
        assert len(DEMO_REGIONS) == 6, (
            f"Expected 6 DEMO_REGIONS (3 original + 3 new), got {len(DEMO_REGIONS)}"
        )

    def test_all_demo_regions_have_valid_seed(self):
        """All DEMO_REGIONS entries must have unique, positive seeds."""
        seeds = []
        for name, region in DEMO_REGIONS.items():
            assert region.seed > 0, f"{name} has invalid seed: {region.seed}"
            assert region.seed not in seeds, f"Duplicate seed {region.seed} in {name}"
            seeds.append(region.seed)

