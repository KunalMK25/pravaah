"""Confidence scoring for population data."""
from __future__ import annotations

import math
from typing import Optional

from flood_risk_zonation.population.enums import PopulationProviderType


def get_baseline_confidence(provider_type: PopulationProviderType) -> float:
    """
    Get baseline confidence for a provider type.

    Parameters
    ----------
    provider_type : PopulationProviderType
        Provider tier

    Returns
    -------
    float
        Baseline confidence [0.0, 1.0]
    """
    baselines = {
        PopulationProviderType.AUTHORITATIVE: 0.92,
        PopulationProviderType.REGIONAL: 0.75,
        PopulationProviderType.WORLDPOP: 0.78,
        PopulationProviderType.OSM: 0.60,
        PopulationProviderType.DERIVED: 0.40,
        PopulationProviderType.UNKNOWN: 0.00,
    }
    return baselines.get(provider_type, 0.0)


def apply_spatial_resolution_penalty(
    baseline_confidence: float,
    spatial_resolution_m: float,
) -> float:
    """
    Apply spatial resolution penalty to confidence.

    Higher resolution (smaller pixel) → lower penalty.
    Lower resolution (larger pixel) → higher penalty.

    Formula: penalty = 0.05 × log10(spatial_resolution_m)

    Parameters
    ----------
    baseline_confidence : float
        Baseline confidence [0.0, 1.0]
    spatial_resolution_m : float
        Spatial resolution in meters

    Returns
    -------
    float
        Adjusted confidence [0.0, 1.0]
    """
    if spatial_resolution_m <= 0 or math.isnan(spatial_resolution_m):
        # Point-level data; no penalty
        return baseline_confidence

    # log10(1000) ≈ 3.0, penalty ≈ 0.15
    # log10(10000) ≈ 4.0, penalty ≈ 0.20
    penalty = 0.05 * math.log10(spatial_resolution_m)
    return max(0.0, baseline_confidence - penalty)


def apply_temporal_age_penalty(
    baseline_confidence: float,
    collection_year: Optional[int],
    current_year: int = 2026,
) -> float:
    """
    Apply temporal age penalty to confidence.

    Non-linear decay: confidence *= 0.95^(current_year - collection_year)

    Parameters
    ----------
    baseline_confidence : float
        Baseline confidence [0.0, 1.0]
    collection_year : Optional[int]
        Year data was collected (None → no penalty)
    current_year : int
        Current year (default 2026)

    Returns
    -------
    float
        Adjusted confidence [0.0, 1.0]
    """
    if collection_year is None:
        # Unknown age; no penalty
        return baseline_confidence

    years_old = max(0, current_year - collection_year)
    if years_old == 0:
        return baseline_confidence

    # Decay multiplier: 0.95^years_old
    # 1 year: 0.95
    # 5 years: 0.95^5 ≈ 0.77
    # 10 years: 0.95^10 ≈ 0.60
    decay_multiplier = 0.95 ** years_old
    return max(0.0, baseline_confidence * decay_multiplier)


def apply_coverage_penalty(
    baseline_confidence: float,
    coverage_percent: float,
) -> float:
    """
    Apply coverage penalty to confidence.

    Partial coverage → lower confidence.

    Formula: penalty = (1 - coverage_percent / 100) × baseline × 0.5

    Parameters
    ----------
    baseline_confidence : float
        Baseline confidence [0.0, 1.0]
    coverage_percent : float
        Coverage fraction [0, 100]

    Returns
    -------
    float
        Adjusted confidence [0.0, 1.0]
    """
    if coverage_percent >= 100:
        return baseline_confidence

    if coverage_percent <= 0:
        return 0.0

    # Gap fraction [0, 1]
    gap_fraction = (100 - coverage_percent) / 100.0

    # Penalty: gap_fraction × baseline × 0.5 (50% of gap becomes penalty)
    penalty = gap_fraction * baseline_confidence * 0.5
    return max(0.0, baseline_confidence - penalty)


def compute_confidence(
    provider_type: PopulationProviderType,
    spatial_resolution_m: Optional[float] = None,
    collection_year: Optional[int] = None,
    coverage_percent: float = 100.0,
    current_year: int = 2026,
) -> float:
    """
    Compute final confidence for population value.

    Combines baseline + spatial/temporal/coverage adjustments.

    Parameters
    ----------
    provider_type : PopulationProviderType
        Provider tier
    spatial_resolution_m : Optional[float]
        Spatial resolution in meters
    collection_year : Optional[int]
        Year data was collected
    coverage_percent : float
        Coverage fraction [0, 100]
    current_year : int
        Current year

    Returns
    -------
    float
        Final confidence [0.0, 1.0]
    """
    baseline = get_baseline_confidence(provider_type)

    # Apply adjustments in sequence
    confidence = baseline

    if spatial_resolution_m is not None:
        confidence = apply_spatial_resolution_penalty(confidence, spatial_resolution_m)

    confidence = apply_temporal_age_penalty(confidence, collection_year, current_year)

    confidence = apply_coverage_penalty(confidence, coverage_percent)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, confidence))
