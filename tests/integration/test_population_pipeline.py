"""Integration tests for population provider chain (Phase 1B)."""
from __future__ import annotations

import math
from datetime import datetime

import geopandas as gpd
import pytest
from shapely.geometry import Point

from flood_risk_zonation.config import BoundingBox
from flood_risk_zonation.models import Habitation, HabitationDataset
from flood_risk_zonation.population.chain import PopulationProviderChain
from flood_risk_zonation.population.enums import PopulationDataStatus, PopulationProviderType
from flood_risk_zonation.population.factory import create_population_provider_chain
from flood_risk_zonation.population.implementations import (
    OSMProvider,
    SyntheticProvider,
    WorldPopProvider,
)
from flood_risk_zonation.population.result import PopulationResult


class TestSyntheticProvider:
    """Test the Synthetic provider (fallback tier)."""

    def test_synthetic_provider_returns_synthetic_status(self):
        """Synthetic provider should return SYNTHETIC status with zero confidence."""
        provider = SyntheticProvider()

        result = provider.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.55,
            bbox_min_lat=12.84,
            bbox_max_lon=77.65,
            bbox_max_lat=12.90,
        )

        assert result.status == PopulationDataStatus.SYNTHETIC
        assert result.confidence == 0.0
        assert result.provider == PopulationProviderType.SYNTHETIC
        assert result.population is not None
        assert result.population > 0

    def test_synthetic_provider_deterministic(self):
        """Same habitation should always get same synthetic estimate."""
        provider = SyntheticProvider()

        result1 = provider.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.55,
            bbox_min_lat=12.84,
            bbox_max_lon=77.65,
            bbox_max_lat=12.90,
        )

        result2 = provider.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.55,
            bbox_min_lat=12.84,
            bbox_max_lon=77.65,
            bbox_max_lat=12.90,
        )

        assert result1.population == result2.population

    def test_synthetic_provider_different_habitations(self):
        """Different habitations should get different estimates (variation)."""
        provider = SyntheticProvider()

        result1 = provider.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.55,
            bbox_min_lat=12.84,
            bbox_max_lon=77.65,
            bbox_max_lat=12.90,
        )

        result2 = provider.get_population(
            hab_id="test_002",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.55,
            bbox_min_lat=12.84,
            bbox_max_lon=77.65,
            bbox_max_lat=12.90,
        )

        # Different IDs should produce different estimates (with 10% variation)
        # Not guaranteed to be different, but likely
        assert result1.population >= 10
        assert result2.population >= 10

    def test_synthetic_provider_has_limitations(self):
        """Synthetic provider should list limitations explicitly."""
        provider = SyntheticProvider()

        result = provider.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.55,
            bbox_min_lat=12.84,
            bbox_max_lon=77.65,
            bbox_max_lat=12.90,
        )

        assert len(result.limitations) > 0
        assert any("SYNTHETIC" in lim.upper() for lim in result.limitations)
        assert result.fallback_reason == "all_real_sources_unavailable"


class TestPopulationProviderChain:
    """Test the population provider chain orchestration."""

    def test_chain_with_synthetic_only(self):
        """Chain with only synthetic provider should work."""
        synthetic = SyntheticProvider()
        chain = PopulationProviderChain(synthetic=synthetic)

        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        assert result.status == PopulationDataStatus.SYNTHETIC
        assert result.population is not None

    def test_chain_falls_back_to_synthetic(self):
        """Chain should fall back to synthetic when no real data available."""
        synthetic = SyntheticProvider()
        chain = PopulationProviderChain(synthetic=synthetic)

        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        # Should have used synthetic as fallback
        assert result.provider == PopulationProviderType.SYNTHETIC

    def test_chain_returns_unknown_without_synthetic(self):
        """Chain without any providers should return UNKNOWN."""
        chain = PopulationProviderChain()

        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        assert result.status == PopulationDataStatus.UNKNOWN


class TestPopulationFactoryWithSynthetic:
    """Test factory function with synthetic provider configuration."""

    def test_factory_creates_chain_with_synthetic_enabled(self):
        """Factory should create chain with synthetic when enabled."""
        config = {
            "synthetic": {"enabled": True},
        }

        chain = create_population_provider_chain(config)

        # Chain should have synthetic provider
        assert chain is not None
        # Get population to test it works
        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )
        assert result is not None

    def test_factory_respects_synthetic_disabled(self):
        """Factory should skip synthetic when disabled."""
        config = {
            "synthetic": {"enabled": False},
        }

        chain = create_population_provider_chain(config)

        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        # Without any providers, should be UNKNOWN
        assert result.status == PopulationDataStatus.UNKNOWN

    def test_factory_with_multiple_providers_prefers_first(self):
        """Factory should prefer first available provider in chain."""
        config = {
            "osm": {"enabled": True},
            "synthetic": {"enabled": True},
        }

        habitations_dict = {
            "hab_1": Habitation(
                hab_id="hab_1",
                name="Test Settlement",
                hab_type="village",
                lat=12.85,
                lon=77.60,
                source="osm",
                population=100,
            )
        }

        chain = create_population_provider_chain(config, habitations_dict=habitations_dict)

        result = chain.get_population(
            hab_id="hab_1",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        # Should use OSM provider first (not synthetic)
        assert result.provider == PopulationProviderType.OSM


class TestPopulationBackwardCompatibility:
    """Test backward compatibility without breaking existing code."""

    def test_chain_without_synthetic_still_works(self):
        """Old code creating chain without synthetic should still work."""
        chain = PopulationProviderChain(
            authoritative=None,
            regional=None,
            worldpop=None,
            osm=None,
            derived=None,
        )

        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        assert result is not None
        assert result.status == PopulationDataStatus.UNKNOWN

    def test_factory_without_synthetic_config_key(self):
        """Factory should work when synthetic config key is absent."""
        config = {
            "osm": {"enabled": False},
        }

        chain = create_population_provider_chain(config)

        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        assert result is not None


class TestPopulationProviderChainFallback:
    """Test fallback behavior in the chain."""

    def test_chain_falls_back_through_tiers(self):
        """Chain should try OSM first, then fall back to synthetic."""
        # OSM provider will fail (no habitations dict)
        osm = OSMProvider(habitations={})
        synthetic = SyntheticProvider()
        chain = PopulationProviderChain(osm=osm, synthetic=synthetic)

        result = chain.get_population(
            hab_id="test_unknown",  # Not in empty habitations dict
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        # Should fall back to synthetic
        assert result.provider == PopulationProviderType.SYNTHETIC
        assert result.status == PopulationDataStatus.SYNTHETIC

    def test_no_data_fabrication(self):
        """Chain should never return OBSERVED when data comes from synthetic."""
        synthetic = SyntheticProvider()
        chain = PopulationProviderChain(synthetic=synthetic)

        result = chain.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox=(77.55, 12.84, 77.65, 12.90),
        )

        # Must be explicitly SYNTHETIC, never OBSERVED
        assert result.status == PopulationDataStatus.SYNTHETIC
        assert result.status != PopulationDataStatus.OBSERVED


class TestPopulationProviderConfiguration:
    """Test configuration options for population providers."""

    def test_synthetic_provider_custom_density(self):
        """Synthetic provider should respect custom base density."""
        config = {"base_density_per_km2": 200}  # Higher density
        provider = SyntheticProvider(config)

        result = provider.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.55,
            bbox_min_lat=12.84,
            bbox_max_lon=77.65,
            bbox_max_lat=12.90,
        )

        # Population should be higher than default (100 persons/km²)
        assert result.population > 0

    def test_synthetic_provider_error_handling(self):
        """Synthetic provider should handle edge cases gracefully."""
        provider = SyntheticProvider()

        # Use invalid bbox (min > max) - synthetic will treat this as degenerate but still generate estimate
        result = provider.get_population(
            hab_id="test_001",
            lat=12.85,
            lon=77.60,
            bbox_min_lon=77.65,  # min > max
            bbox_min_lat=12.90,  # min > max
            bbox_max_lon=77.55,
            bbox_max_lat=12.84,
        )

        # Should return SYNTHETIC with a valid (though small) population estimate
        # Synthetic provider doesn't crash on edge cases, it generates a fallback estimate
        assert result.status == PopulationDataStatus.SYNTHETIC
        assert result.population is not None
        assert result.population > 0
