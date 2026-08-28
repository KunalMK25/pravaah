"""PopulationResult — population data with complete provenance and confidence tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from flood_risk_zonation.population.enums import (
    PopulationDataStatus,
    PopulationMethod,
    PopulationProviderType,
)


@dataclass
class PopulationResult:
    """
    Population data with complete provenance and confidence tracking.

    Never fabricated. Always includes source, status, confidence.

    Attributes
    ----------
    population : Optional[int]
        Population count (None if UNKNOWN/UNAVAILABLE)
    source : str
        Source name (e.g., "worldpop_2020", "census_2020", "osm_tag")
    provider : PopulationProviderType
        Provider tier (AUTHORITATIVE, REGIONAL, WORLDPOP, OSM, DERIVED, UNKNOWN)
    method : Optional[PopulationMethod]
        How value was obtained
    status : PopulationDataStatus
        OBSERVED | ESTIMATED | CACHED | UNAVAILABLE | UNKNOWN
    confidence : float
        [0.0, 1.0] — data quality score
    spatial_resolution_m : float
        Spatial resolution in meters (0 for point, NaN for unknown)
    temporal_resolution : str
        Temporal granularity ("static", "annual", "point_in_time")
    collection_year : Optional[int]
        Year data was collected (None if unknown)
    retrieved_at : datetime
        When data was retrieved/computed
    coverage_percent : float
        [0, 100] — spatial coverage fraction
    limitations : list[str]
        Known limitations of this dataset
    fallback_reason : Optional[str]
        Why previous provider failed (if chain used)
    """

    # Primary data
    population: Optional[int]

    # Provenance
    source: str
    provider: PopulationProviderType
    method: Optional[PopulationMethod]
    status: PopulationDataStatus

    # Quality metrics
    confidence: float
    spatial_resolution_m: float
    temporal_resolution: str
    collection_year: Optional[int]

    # Retrieval metadata
    retrieved_at: datetime
    coverage_percent: float

    # Documentation
    limitations: list[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None

    def __post_init__(self):
        """Validate invariants."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not 0 <= self.coverage_percent <= 100:
            raise ValueError(f"coverage_percent must be in [0, 100], got {self.coverage_percent}")
        if self.spatial_resolution_m < 0 and not (
            isinstance(self.spatial_resolution_m, float)
            and self.spatial_resolution_m != self.spatial_resolution_m
        ):  # NaN check
            raise ValueError(f"spatial_resolution_m must be >= 0 or NaN, got {self.spatial_resolution_m}")

        # Status/population consistency
        if self.status == PopulationDataStatus.UNKNOWN and self.population is not None:
            raise ValueError("status=UNKNOWN but population is not None")
        if self.status == PopulationDataStatus.UNKNOWN and self.confidence != 0.0:
            raise ValueError("status=UNKNOWN must have confidence=0.0")
        if self.status == PopulationDataStatus.UNAVAILABLE and self.population is not None:
            raise ValueError("status=UNAVAILABLE but population is not None")

        # No fabrication guarantee
        if self.status in (PopulationDataStatus.OBSERVED, PopulationDataStatus.ESTIMATED) and self.population is None:
            raise ValueError(f"status={self.status} but population is None")

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.status in (PopulationDataStatus.UNKNOWN, PopulationDataStatus.UNAVAILABLE):
            return f"{self.status.value} ({self.provider.value})"
        status_label = "observed" if self.status == PopulationDataStatus.OBSERVED else "estimated"
        return f"{self.population:,} ({self.provider.value}, {status_label}, conf={self.confidence:.2f})"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "population": self.population,
            "source": self.source,
            "provider": self.provider.value,
            "method": self.method.value if self.method else None,
            "status": self.status.value,
            "confidence": float(self.confidence),
            "spatial_resolution_m": float(self.spatial_resolution_m) if not (
                isinstance(self.spatial_resolution_m, float) and self.spatial_resolution_m != self.spatial_resolution_m
            ) else None,
            "temporal_resolution": self.temporal_resolution,
            "collection_year": self.collection_year,
            "retrieved_at": self.retrieved_at.isoformat(),
            "coverage_percent": float(self.coverage_percent),
            "limitations": self.limitations,
            "fallback_reason": self.fallback_reason,
        }
