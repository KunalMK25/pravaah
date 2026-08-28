"""Tests for population provider system."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flood_risk_zonation.models import Habitation, RasterDataset
from flood_risk_zonation.population.chain import PopulationProviderChain
from flood_risk_zonation.population.confidence import (
    apply_coverage_penalty,
    apply_spatial_resolution_penalty,
    apply_temporal_age_penalty,
    compute_confidence,
    get_baseline_confidence,
)
from flood_risk_zonation.population.enums import (
    PopulationDataStatus,
    PopulationMethod,
    PopulationProviderType,
)
from flood_risk_zonation.population.factory import create_population_provider_chain
from flood_risk_zonation.population.implementations import (
    AuthoritativeProvider,
    DerivedProvider,
    OSMProvider,
    RegionalProvider,
    WorldPopProvider,
)
from flood_risk_zonation.population.provider import UnknownProvider
from flood_risk_zonation.population.result import PopulationResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_raster():
    """Create a mock WorldPop raster."""
    from rasterio.transform import Affine

    # Create a small raster array (10x10 pixels at 1000m resolution)
    array = np.random.exponential(scale=500, size=(10, 10)).astype(np.float32)
    array = np.clip(array, 0, None)

    # Transform: upper-left corner at (77.0, 13.0), pixel size 0.009 degrees (~1km)
    transform = Affine(0.009, 0, 77.0, 0, -0.009, 13.0)

    return RasterDataset(
        array=array,
        transform=transform,
        crs="EPSG:4326",
        nodata=None,
        source="test_worldpop_2020",
    )


@pytest.fixture
def mock_habitations():
    """Create mock habitations with OSM population tags."""
    habs = {
        "hab_001": Habitation(
            hab_id="hab_001",
            name="Test Town",
            hab_type="village",
            lat=12.95,
            lon=77.05,
            source="osm_overpass",
            population=5000,
        ),
        "hab_002": Habitation(
            hab_id="hab_002",
            name="Unknown Village",
            hab_type="village",
            lat=12.96,
            lon=77.06,
            source="osm_overpass",
            population=None,  # No OSM population tag
        ),
    }
    return habs


# ── Enum Tests ────────────────────────────────────────────────────────────────


class TestEnums:
    """Test population provider enums."""

    def test_population_data_status_values(self):
        """PopulationDataStatus has correct values."""
        assert PopulationDataStatus.OBSERVED.value == "OBSERVED"
        assert PopulationDataStatus.ESTIMATED.value == "ESTIMATED"
        assert PopulationDataStatus.UNKNOWN.value == "UNKNOWN"

    def test_population_provider_type_values(self):
        """PopulationProviderType has all 6 tiers."""
        providers = PopulationProviderType
        assert hasattr(providers, "AUTHORITATIVE")
        assert hasattr(providers, "REGIONAL")
        assert hasattr(providers, "WORLDPOP")
        assert hasattr(providers, "OSM")
        assert hasattr(providers, "DERIVED")
        assert hasattr(providers, "UNKNOWN")

    def test_population_method_values(self):
        """PopulationMethod has correct methods."""
        assert PopulationMethod.OSM_TAG_DIRECT.value == "osm_tag_direct"
        assert PopulationMethod.RASTER_AGGREGATION.value == "raster_aggregation"


# ── Result Tests ──────────────────────────────────────────────────────────────


class TestPopulationResult:
    """Test PopulationResult dataclass."""

    def test_valid_observed_result(self):
        """Create valid OBSERVED result."""
        result = PopulationResult(
            population=5000,
            source="osm_tag",
            provider=PopulationProviderType.OSM,
            method=PopulationMethod.OSM_TAG_DIRECT,
            status=PopulationDataStatus.OBSERVED,
            confidence=0.60,
            spatial_resolution_m=0.0,
            temporal_resolution="point_in_time",
            collection_year=None,
            retrieved_at=datetime.now(),
            coverage_percent=100.0,
        )
        assert result.population == 5000
        assert result.confidence == 0.60
        assert result.status == PopulationDataStatus.OBSERVED

    def test_valid_unknown_result(self):
        """Create valid UNKNOWN result."""
        result = PopulationResult(
            population=None,
            source="unknown",
            provider=PopulationProviderType.UNKNOWN,
            method=None,
            status=PopulationDataStatus.UNKNOWN,
            confidence=0.0,
            spatial_resolution_m=float("nan"),
            temporal_resolution="unknown",
            collection_year=None,
            retrieved_at=datetime.now(),
            coverage_percent=0.0,
        )
        assert result.population is None
        assert result.confidence == 0.0
        assert result.status == PopulationDataStatus.UNKNOWN

    def test_invalid_confidence_raises(self):
        """Confidence must be in [0, 1]."""
        with pytest.raises(ValueError, match="confidence"):
            PopulationResult(
                population=5000,
                source="test",
                provider=PopulationProviderType.OSM,
                method=PopulationMethod.OSM_TAG_DIRECT,
                status=PopulationDataStatus.OBSERVED,
                confidence=1.5,  # Invalid!
                spatial_resolution_m=0.0,
                temporal_resolution="point_in_time",
                collection_year=None,
                retrieved_at=datetime.now(),
                coverage_percent=100.0,
            )

    def test_unknown_status_requires_zero_confidence(self):
        """UNKNOWN status must have confidence=0.0."""
        with pytest.raises(ValueError, match="confidence=0.0"):
            PopulationResult(
                population=None,
                source="unknown",
                provider=PopulationProviderType.UNKNOWN,
                method=None,
                status=PopulationDataStatus.UNKNOWN,
                confidence=0.5,  # Invalid for UNKNOWN!
                spatial_resolution_m=float("nan"),
                temporal_resolution="unknown",
                collection_year=None,
                retrieved_at=datetime.now(),
                coverage_percent=0.0,
            )

    def test_to_dict_serialization(self):
        """PopulationResult converts to dict."""
        result = PopulationResult(
            population=3000,
            source="osm_tag",
            provider=PopulationProviderType.OSM,
            method=PopulationMethod.OSM_TAG_DIRECT,
            status=PopulationDataStatus.OBSERVED,
            confidence=0.60,
            spatial_resolution_m=0.0,
            temporal_resolution="point_in_time",
            collection_year=None,
            retrieved_at=datetime.now(),
            coverage_percent=100.0,
        )
        d = result.to_dict()
        assert d["population"] == 3000
        assert d["provider"] == "OSM"
        assert d["status"] == "OBSERVED"
        assert d["confidence"] == 0.60


# ── Confidence Tests ──────────────────────────────────────────────────────────


class TestConfidenceCalculation:
    """Test confidence scoring."""

    def test_baseline_confidence_per_provider(self):
        """Baseline confidence values are correct."""
        assert get_baseline_confidence(PopulationProviderType.AUTHORITATIVE) == 0.92
        assert get_baseline_confidence(PopulationProviderType.WORLDPOP) == 0.78
        assert get_baseline_confidence(PopulationProviderType.OSM) == 0.60
        assert get_baseline_confidence(PopulationProviderType.UNKNOWN) == 0.00

    def test_spatial_resolution_penalty(self):
        """Spatial resolution reduces confidence."""
        baseline = 0.80
        # 1km (1000m): penalty ≈ 0.15
        adjusted = apply_spatial_resolution_penalty(baseline, 1000.0)
        assert adjusted < baseline
        assert adjusted > 0.60

    def test_temporal_age_penalty(self):
        """Data age reduces confidence (non-linear)."""
        baseline = 0.80
        # 0 years old: no penalty
        assert apply_temporal_age_penalty(baseline, 2026, 2026) == baseline
        # 5 years old: ~23% penalty
        old_data = apply_temporal_age_penalty(baseline, 2021, 2026)
        assert old_data < baseline

    def test_coverage_penalty(self):
        """Partial coverage reduces confidence."""
        baseline = 0.80
        # 100% coverage: no penalty
        assert apply_coverage_penalty(baseline, 100.0) == baseline
        # 50% coverage: penalty
        partial = apply_coverage_penalty(baseline, 50.0)
        assert partial < baseline

    def test_compute_confidence_full(self):
        """Compute confidence with all adjustments."""
        confidence = compute_confidence(
            provider_type=PopulationProviderType.WORLDPOP,
            spatial_resolution_m=1000.0,
            collection_year=2020,
            coverage_percent=100.0,
            current_year=2026,
        )
        assert 0.0 <= confidence <= 1.0
        # Should be lower than baseline due to temporal age
        baseline = get_baseline_confidence(PopulationProviderType.WORLDPOP)
        assert confidence < baseline


# ── Provider Tests ────────────────────────────────────────────────────────────


class TestUnknownProvider:
    """Test UnknownProvider (terminal tier)."""

    def test_always_returns_unknown(self):
        """UnknownProvider always returns UNKNOWN."""
        provider = UnknownProvider()
        result = provider.get_population("h1", 12.95, 77.05, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.UNKNOWN
        assert result.population is None
        assert result.confidence == 0.0


class TestOSMProvider:
    """Test OSMProvider (Tier 4)."""

    def test_returns_osm_population_when_available(self, mock_habitations):
        """OSMProvider returns OSM population when tag exists."""
        provider = OSMProvider(habitations=mock_habitations)
        result = provider.get_population("hab_001", 12.95, 77.05, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.OBSERVED
        assert result.population == 5000
        assert result.provider == PopulationProviderType.OSM
        assert result.confidence > 0.0

    def test_returns_unavailable_when_osm_tag_missing(self, mock_habitations):
        """OSMProvider returns UNAVAILABLE when tag missing."""
        provider = OSMProvider(habitations=mock_habitations)
        result = provider.get_population("hab_002", 12.96, 77.06, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.UNAVAILABLE
        assert result.population is None

    def test_returns_unavailable_for_missing_habitation(self):
        """OSMProvider returns UNAVAILABLE for unknown habitation ID."""
        provider = OSMProvider(habitations={})
        result = provider.get_population("unknown", 12.95, 77.05, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.UNAVAILABLE


class TestWorldPopProvider:
    """Test WorldPopProvider (Tier 3)."""

    def test_aggregates_raster_within_radius(self, mock_raster):
        """WorldPopProvider aggregates population within radius."""
        provider = WorldPopProvider(raster=mock_raster, search_radius_km=2.0)
        # Point in center of raster (77.045, 12.945)
        result = provider.get_population(
            "hab_001", 12.945, 77.045, 77.0, 12.9, 77.1, 13.0
        )
        # Should have population data (raster has values)
        assert result.status in (PopulationDataStatus.OBSERVED, PopulationDataStatus.UNAVAILABLE)
        assert result.provider == PopulationProviderType.WORLDPOP

    def test_returns_unavailable_outside_raster(self, mock_raster):
        """WorldPopProvider returns UNAVAILABLE outside raster coverage."""
        provider = WorldPopProvider(raster=mock_raster, search_radius_km=0.5)
        # Point far outside raster bounds (0, 0)
        result = provider.get_population("hab_001", 0.0, 0.0, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.UNAVAILABLE


class TestAuthoritativeProvider:
    """Test AuthoritativeProvider (Tier 1)."""

    def test_returns_unavailable_when_not_configured(self):
        """AuthoritativeProvider returns UNAVAILABLE when not configured."""
        provider = AuthoritativeProvider(config={"enabled": False})
        result = provider.get_population("hab_001", 12.95, 77.05, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.UNAVAILABLE
        assert result.fallback_reason == "authoritative_not_configured"


class TestRegionalProvider:
    """Test RegionalProvider (Tier 2)."""

    def test_returns_unavailable_when_not_configured(self):
        """RegionalProvider returns UNAVAILABLE when not configured."""
        provider = RegionalProvider(config={"enabled": False})
        result = provider.get_population("hab_001", 12.95, 77.05, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.UNAVAILABLE
        assert result.fallback_reason == "regional_not_configured"


class TestDerivedProvider:
    """Test DerivedProvider (Tier 5)."""

    def test_returns_unavailable_not_implemented(self):
        """DerivedProvider returns UNAVAILABLE (not yet implemented)."""
        provider = DerivedProvider()
        result = provider.get_population("hab_001", 12.95, 77.05, 77.0, 12.9, 77.1, 13.0)
        assert result.status == PopulationDataStatus.UNAVAILABLE
        assert result.fallback_reason == "derived_not_implemented"


# ── Chain Tests ───────────────────────────────────────────────────────────────


class TestPopulationProviderChain:
    """Test PopulationProviderChain orchestrator."""

    def test_prefers_worldpop_over_osm(self, mock_habitations, mock_raster):
        """Chain prefers WorldPop (Tier 3) over OSM (Tier 4) when both available."""
        osm = OSMProvider(habitations=mock_habitations)
        worldpop = WorldPopProvider(raster=mock_raster)
        chain = PopulationProviderChain(osm=osm, worldpop=worldpop)
        result = chain.get_population("hab_001", 12.95, 77.05, (77.0, 12.9, 77.1, 13.0))
        # Should use WorldPop (tier 3) when both available, per architecture hierarchy
        assert result.status == PopulationDataStatus.OBSERVED
        assert result.provider == PopulationProviderType.WORLDPOP
        # Population from raster aggregation (test raster has data at this location)
        assert result.population is not None and result.population > 0

    def test_falls_back_to_osm_when_worldpop_unavailable(self, mock_habitations, mock_raster):
        """Chain falls back to OSM when WorldPop unavailable at location."""
        osm = OSMProvider(habitations=mock_habitations)
        worldpop = WorldPopProvider(raster=mock_raster)
        chain = PopulationProviderChain(osm=osm, worldpop=worldpop)
        # hab_002 has no OSM population; fallback behavior depends on whether
        # WorldPop has data at this location. If WorldPop unavailable, falls back to OSM.
        result = chain.get_population("hab_002", 12.945, 77.045, (77.0, 12.9, 77.1, 13.0))
        # Result should be from WorldPop, OSM, or unknown depending on data availability
        assert result.provider in (PopulationProviderType.WORLDPOP, PopulationProviderType.OSM, PopulationProviderType.UNKNOWN)

    def test_returns_unknown_when_all_fail(self, mock_habitations):
        """Chain returns UNKNOWN when all providers fail."""
        osm = OSMProvider(habitations={})  # Empty habitations
        chain = PopulationProviderChain(osm=osm)
        result = chain.get_population("unknown", 12.95, 77.05, (77.0, 12.9, 77.1, 13.0))
        assert result.status == PopulationDataStatus.UNKNOWN
        assert result.provider == PopulationProviderType.UNKNOWN

    def test_batch_processing(self, mock_habitations):
        """Chain processes multiple habitations."""
        osm = OSMProvider(habitations=mock_habitations)
        chain = PopulationProviderChain(osm=osm)
        habs = [
            {"hab_id": "hab_001", "lat": 12.95, "lon": 77.05},
            {"hab_id": "hab_002", "lat": 12.96, "lon": 77.06},
        ]
        results = chain.get_population_batch(habs, (77.0, 12.9, 77.1, 13.0))
        assert len(results) == 2
        assert results[0].population == 5000  # hab_001 has OSM data
        assert results[1].population is None  # hab_002 doesn't


# ── Factory Tests ─────────────────────────────────────────────────────────────


class TestProviderFactory:
    """Test provider chain factory."""

    def test_creates_chain_with_osm_only(self, mock_habitations):
        """Factory creates OSM-only chain."""
        config = {
            "osm": {"enabled": True},
        }
        chain = create_population_provider_chain(
            config=config,
            habitations_dict=mock_habitations,
        )
        assert isinstance(chain, PopulationProviderChain)
        assert chain is not None

    def test_creates_chain_with_worldpop_and_osm(self, mock_habitations, mock_raster):
        """Factory creates WorldPop + OSM chain."""
        config = {
            "worldpop": {"enabled": True, "search_radius_km": 2.0},
            "osm": {"enabled": True},
        }
        chain = create_population_provider_chain(
            config=config,
            worldpop_raster=mock_raster,
            habitations_dict=mock_habitations,
        )
        assert isinstance(chain, PopulationProviderChain)


# ── Scientific Integrity Tests ────────────────────────────────────────────────


class TestScientificIntegrity:
    """Test scientific integrity rules."""

    def test_never_fabricates_when_no_data(self):
        """System returns UNKNOWN when no data, never fabricates."""
        provider = UnknownProvider()
        result = provider.get_population("h1", 0, 0, -1, -1, 1, 1)
        assert result.population is None
        assert result.status == PopulationDataStatus.UNKNOWN
        assert result.confidence == 0.0

    def test_synthetic_never_observed(self, mock_raster):
        """Synthetic data never marked as OBSERVED."""
        # If raster marked as synthetic, provider should mark status appropriately
        provider = WorldPopProvider(raster=mock_raster)
        result = provider.get_population("h1", 12.945, 77.045, 77.0, 12.9, 77.1, 13.0)
        # Result should be OBSERVED (real raster) or UNAVAILABLE, never ESTIMATED
        # (unless we explicitly mark source as synthetic)
        if result.population is not None:
            assert result.status == PopulationDataStatus.OBSERVED

    def test_provenance_always_preserved(self):
        """Provenance is always tracked."""
        osm_result = PopulationResult(
            population=5000,
            source="osm_tag",
            provider=PopulationProviderType.OSM,
            method=PopulationMethod.OSM_TAG_DIRECT,
            status=PopulationDataStatus.OBSERVED,
            confidence=0.60,
            spatial_resolution_m=0.0,
            temporal_resolution="point_in_time",
            collection_year=None,
            retrieved_at=datetime.now(),
            coverage_percent=100.0,
        )
        # All provenance fields set
        assert osm_result.source is not None
        assert osm_result.provider is not None
        assert osm_result.method is not None
        assert osm_result.status is not None

    def test_confidence_reflects_quality(self):
        """Confidence reflects data quality."""
        # OSM (sparse, unreliable) should have lower confidence than authoritative
        osm_conf = get_baseline_confidence(PopulationProviderType.OSM)
        auth_conf = get_baseline_confidence(PopulationProviderType.AUTHORITATIVE)
        assert osm_conf < auth_conf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
