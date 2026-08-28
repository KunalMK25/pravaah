"""Sentinel-1 flood observation result dataclass with complete provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Sentinel1ObservationResult:
    """
    Sentinel-1-derived flood observation with complete scientific provenance.

    SCIENTIFIC INTEGRITY:
    - flood_observed: True = flooded area detected; False = no flood; None = UNKNOWN
    - Do NOT confuse "no detection" (UNKNOWN) with "no flood" (False)
    - confidence: [0.0–1.0] reflects input quality, not statistical hallucination
    - Provenance mandatory: source, platform, sensor, acquisition_time all preserved
    - Never fabricates satellite observations; status must be explicit (UNKNOWN/UNAVAILABLE)

    Attributes
    ----------
    observation_status : str
        "OBSERVED" (valid acquisition), "UNAVAILABLE" (no data), "UNKNOWN" (error/synthetic)
    flood_observed : bool | None
        True = flood detected, False = no flood detected, None = unknown
    inundation_fraction : float
        [0.0–1.0] fraction of observed area that is inundated (NaN if unknown)
    flooded_area_km2 : float
        Approximate area of observed flooding in km² (NaN if unknown)
    no_data_fraction : float
        [0.0–1.0] fraction of observation area with no data/invalid data
    confidence : float
        [0.0–1.0] quality of observation (input quality, not statistical)
    coverage_fraction : float
        [0.0–1.0] fraction of analysis area with valid observations
    source : str
        "sentinel1_geotiff", "sentinel1_geojson", "test_synthetic", "unknown"
    provider : str
        "Copernicus", "Google Earth Engine", "Local", "Test", "Unknown"
    platform : str
        "Sentinel-1A", "Sentinel-1B", "Test", "Unknown"
    sensor : str
        "SAR", "Test", "Unknown"
    acquisition_time : datetime
        Satellite acquisition timestamp (UTC)
    processing_time : datetime
        When the observation was processed/ingested
    method : str
        Processing method: "SAR_CHANGE_DETECTION", "DERIVED_FLOOD_MASK", "VECTOR_POLYGONS", "TEST_SYNTHETIC"
    spatial_resolution_m : float
        Nominal spatial resolution in meters (e.g., 10m for Sentinel-1)
    crs : str
        Coordinate reference system (e.g., "EPSG:4326")
    bbox : tuple[float, float, float, float]
        (min_lon, min_lat, max_lon, max_lat) observation extent
    input_format : str
        "GeoTIFF", "GeoJSON", "SYNTHETIC", "UNKNOWN"
    fallback_reason : str | None
        Reason for fallback state (e.g., "no data available for date range")
    limitations : list[str]
        Explicit limitations of this observation
    """

    observation_status: str  # "OBSERVED", "UNAVAILABLE", "UNKNOWN"
    flood_observed: bool | None
    inundation_fraction: float
    flooded_area_km2: float
    no_data_fraction: float
    confidence: float
    coverage_fraction: float
    source: str
    provider: str
    platform: str
    sensor: str
    acquisition_time: datetime
    processing_time: datetime
    method: str
    spatial_resolution_m: float
    crs: str
    bbox: tuple[float, float, float, float]
    input_format: str
    fallback_reason: str | None = None
    limitations: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate invariants."""
        # Validate status
        if self.observation_status not in ("OBSERVED", "UNAVAILABLE", "UNKNOWN"):
            raise ValueError(f"Invalid observation_status: {self.observation_status}")

        # Validate confidence in [0, 1]
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

        # Validate coverage in [0, 1]
        if not (0.0 <= self.coverage_fraction <= 1.0):
            raise ValueError(f"Coverage fraction must be in [0, 1], got {self.coverage_fraction}")

        # Validate inundation in [0, 1] or NaN
        import math

        if not (math.isnan(self.inundation_fraction) or 0.0 <= self.inundation_fraction <= 1.0):
            raise ValueError(f"Inundation fraction must be in [0, 1] or NaN, got {self.inundation_fraction}")

        # Validate no_data in [0, 1]
        if not (0.0 <= self.no_data_fraction <= 1.0):
            raise ValueError(f"No-data fraction must be in [0, 1], got {self.no_data_fraction}")

        # Scientific integrity: UNKNOWN observations should not have flood_observed=False
        if self.observation_status == "UNKNOWN" and self.flood_observed is False:
            raise ValueError(
                "Cannot mark UNKNOWN observation as 'no flood' — use None for unknown flood status"
            )

        # UNAVAILABLE observations should not have specific flood measurements
        if self.observation_status == "UNAVAILABLE" and self.flood_observed is not None:
            raise ValueError(
                "UNAVAILABLE observation should not have flood_observed != None"
            )


def create_unknown_sentinel1_result(
    bbox: tuple[float, float, float, float],
    reason: str = "No Sentinel-1 data available",
) -> Sentinel1ObservationResult:
    """
    Create a terminal UNKNOWN sentinel-1 result.

    Returns
    -------
    Sentinel1ObservationResult
        Explicit UNKNOWN state (not fabricated observations)
    """
    import math
    from datetime import datetime

    return Sentinel1ObservationResult(
        observation_status="UNKNOWN",
        flood_observed=None,
        inundation_fraction=math.nan,
        flooded_area_km2=math.nan,
        no_data_fraction=1.0,
        confidence=0.0,
        coverage_fraction=0.0,
        source="unknown",
        provider="Unknown",
        platform="Unknown",
        sensor="Unknown",
        acquisition_time=datetime.now(),
        processing_time=datetime.now(),
        method="UNKNOWN",
        spatial_resolution_m=math.nan,
        crs="EPSG:4326",
        bbox=bbox,
        input_format="UNKNOWN",
        fallback_reason=reason,
        limitations=[
            "No Sentinel-1 observation data available for this region/date.",
            "This is a terminal fallback state indicating data unavailability.",
        ],
    )


def create_unavailable_sentinel1_result(
    bbox: tuple[float, float, float, float],
    reason: str = "Sentinel-1 data provider unavailable",
) -> Sentinel1ObservationResult:
    """
    Create a terminal UNAVAILABLE sentinel-1 result.

    Returns
    -------
    Sentinel1ObservationResult
        Explicit UNAVAILABLE state (provider/API unavailable)
    """
    import math
    from datetime import datetime

    return Sentinel1ObservationResult(
        observation_status="UNAVAILABLE",
        flood_observed=None,
        inundation_fraction=math.nan,
        flooded_area_km2=math.nan,
        no_data_fraction=1.0,
        confidence=0.0,
        coverage_fraction=0.0,
        source="unknown",
        provider="Unknown",
        platform="Unknown",
        sensor="Unknown",
        acquisition_time=datetime.now(),
        processing_time=datetime.now(),
        method="UNKNOWN",
        spatial_resolution_m=math.nan,
        crs="EPSG:4326",
        bbox=bbox,
        input_format="UNKNOWN",
        fallback_reason=reason,
        limitations=[
            "Sentinel-1 data provider is unavailable.",
            "This is a terminal fallback state.",
        ],
    )
