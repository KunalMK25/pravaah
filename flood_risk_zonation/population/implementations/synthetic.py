"""
SyntheticProvider — Tier 6 (fallback) — synthetic population estimates.

Generates plausible but NOT real population data for regions where all
authoritative and OSM data are unavailable. Used ONLY as a final fallback
to ensure the pipeline completes with a best-guess estimate.

SCIENTIFIC INTEGRITY:
- Marked as SYNTHETIC (never confused with observed data)
- Confidence = 0.0 (lowest possible, indicates guesswork)
- Fallback reason recorded
- Limitations explicitly listed
- Used ONLY when all real data sources fail
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from flood_risk_zonation.population.confidence import compute_confidence
from flood_risk_zonation.population.enums import (
    PopulationDataStatus,
    PopulationMethod,
    PopulationProviderType,
)
from flood_risk_zonation.population.provider import PopulationProvider
from flood_risk_zonation.population.result import PopulationResult

logger = logging.getLogger(__name__)


class SyntheticProvider(PopulationProvider):
    """
    Tier 6 (fallback): Synthetic population estimates.

    Generates plausible population estimates for regions where real data is unavailable.
    Used as a terminal fallback when OSM, WorldPop, Authoritative, and Derived all fail.

    Heuristic approach:
    - Assumes uniform population density across the bounding box
    - Base density: ~100 persons/km² (typical rural area)
    - Varies by habitat type (not available, so uses default)
    - Adds small random variation for realism

    CRITICAL: This provider is marked SYNTHETIC with confidence=0.0.
    Never use this data for official decision-making without acknowledging
    that it is synthetic, not observed.

    Parameters
    ----------
    config : Optional[dict]
        Configuration dict with optional keys:
        - "base_density_per_km2": default population density (default: 100)
        - "random_seed": seed for synthetic variation (default: 42)
    """

    provider_type = PopulationProviderType.SYNTHETIC

    def __init__(self, config: Optional[dict] = None):
        """Initialize synthetic provider."""
        self.config = config or {}
        self.base_density_per_km2 = self.config.get("base_density_per_km2", 100)
        self.random_seed = self.config.get("random_seed", 42)

    def get_population(
        self,
        hab_id: str,
        lat: float,
        lon: float,
        bbox_min_lon: float,
        bbox_min_lat: float,
        bbox_max_lon: float,
        bbox_max_lat: float,
    ) -> PopulationResult:
        """
        Generate synthetic population estimate for a habitation.

        Uses a heuristic based on:
        1. Bounding box area
        2. Base population density
        3. Small random variation per-habitation

        Parameters
        ----------
        hab_id : str
            Habitation ID (used for seeding)
        lat, lon : float
            Habitation location
        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat : float
            Analysis bounding box

        Returns
        -------
        PopulationResult
            Synthetic population estimate with status=SYNTHETIC, confidence=0.0
        """
        try:
            # Calculate bounding box area in km²
            height_km = (bbox_max_lat - bbox_min_lat) * 111.32
            center_lat_rad = math.radians((bbox_min_lat + bbox_max_lat) / 2)
            width_km = (
                (bbox_max_lon - bbox_min_lon) * 111.32 * math.cos(center_lat_rad)
            )
            area_km2 = max(0.1, height_km * width_km)  # Avoid division by zero

            # Base synthetic population
            base_population = self.base_density_per_km2 * area_km2

            # Add small random variation (±10%) seeded by habitation ID
            # This ensures the same habitation always gets the same estimate
            # but different habitations get slightly different values
            hash_seed = hash(hab_id) % (2**31)
            variation = (hash_seed % 21) / 100.0 - 0.1  # Range: -10% to +10%
            synthetic_population = int(base_population * (1.0 + variation))

            # Ensure reasonable bounds
            synthetic_population = max(10, min(synthetic_population, 100000))

            logger.debug(
                "Synthetic population for %s at (%.2f, %.2f): %d (area=%.2f km²)",
                hab_id,
                lat,
                lon,
                synthetic_population,
                area_km2,
            )

            return PopulationResult(
                population=synthetic_population,
                source="synthetic",
                provider=self.provider_type,
                method=PopulationMethod.SYNTHETIC,
                status=PopulationDataStatus.SYNTHETIC,
                confidence=0.0,  # Lowest confidence: this is a guess
                spatial_resolution_m=5000.0,  # Very coarse estimate
                temporal_resolution="unknown",
                collection_year=None,
                retrieved_at=datetime.now(),
                coverage_percent=100.0,
                limitations=[
                    "This is a SYNTHETIC estimate, not based on real data.",
                    "Used only as a fallback when all real population sources are unavailable.",
                    "Accuracy is very low; use for pipeline completion only.",
                    "Do not use for official or policy-making decisions.",
                ],
                fallback_reason="all_real_sources_unavailable",
            )

        except Exception as exc:
            logger.error("Synthetic population generation failed for %s: %s", hab_id, exc)
            return PopulationResult(
                population=None,
                source="synthetic",
                provider=self.provider_type,
                method=None,
                status=PopulationDataStatus.UNKNOWN,
                confidence=0.0,
                spatial_resolution_m=float("nan"),
                temporal_resolution="unknown",
                collection_year=None,
                retrieved_at=datetime.now(),
                coverage_percent=0.0,
                limitations=[
                    "Synthetic population generation failed.",
                    "This habitation will have missing population data.",
                ],
                fallback_reason=f"synthetic_error: {str(exc)}",
            )
