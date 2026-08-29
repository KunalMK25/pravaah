"""
PRAVAAH-AI — Sentinel-1 satellite vs. model comparison metrics.

Compares Sentinel-1-observed flood extent with model-predicted flood extent
to compute meaningful validation metrics:
- IoU (Intersection over Union)
- Precision
- Recall
- F1 score

SCIENTIFIC INTEGRITY:
- Metrics computed only when both model and Sentinel-1 observations are available
- Unknown/unavailable observations result in UNKNOWN metrics (no fabrication)
- All metrics reflect actual data quality
- No synthetic metric generation
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, shape

from flood_risk_zonation.satellite.result import Sentinel1ObservationResult

logger = logging.getLogger(__name__)


@dataclass
class Sentinel1ComparisonMetrics:
    """
    Validation metrics comparing Sentinel-1 observations against model predictions.

    Attributes
    ----------
    comparison_status : str
        "COMPUTED" = valid metrics available
        "UNAVAILABLE" = Sentinel-1 data missing or invalid
        "UNKNOWN" = error during computation
    iou : float | None
        Intersection over Union [0, 1], None if unavailable
    precision : float | None
        True Positives / (True Positives + False Positives) [0, 1], None if unavailable
    recall : float | None
        True Positives / (True Positives + False Negatives) [0, 1], None if unavailable
    f1_score : float | None
        Harmonic mean of precision and recall [0, 1], None if unavailable
    true_positives : int
        Grid cells: both Sentinel-1 and model predict flood
    true_negatives : int
        Grid cells: both Sentinel-1 and model predict no flood
    false_positives : int
        Grid cells: model predicts flood but Sentinel-1 does not
    false_negatives : int
        Grid cells: Sentinel-1 predicts flood but model does not
    total_cells : int
        Total cells compared (may be < full grid due to Sentinel-1 coverage)
    sentinel1_inundation_fraction : float
        Fraction of grid where Sentinel-1 observed flood [0, 1]
    model_inundation_fraction : float
        Fraction of grid where model predicts flood [0, 1]
    satellite_confidence : float
        Confidence of Sentinel-1 observation (0 = UNKNOWN, 1 = high confidence)
    coverage_fraction : float
        Fraction of grid covered by Sentinel-1 observation
    error_reason : str | None
        Explanation of why comparison failed (if status != COMPUTED)
    limitations : list[str]
        Explicit limitations of these metrics
    """

    comparison_status: str  # "COMPUTED", "UNAVAILABLE", "UNKNOWN"
    iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_cells: int = 0
    sentinel1_inundation_fraction: float = 0.0
    model_inundation_fraction: float = 0.0
    satellite_confidence: float = 0.0
    coverage_fraction: float = 0.0
    error_reason: str | None = None
    limitations: list[str] = None

    def __post_init__(self):
        """Validate invariants and set defaults."""
        if self.limitations is None:
            self.limitations = []

        # Validate status
        if self.comparison_status not in ("COMPUTED", "UNAVAILABLE", "UNKNOWN"):
            raise ValueError(f"Invalid comparison_status: {self.comparison_status}")

        # COMPUTED status requires valid metrics
        if self.comparison_status == "COMPUTED":
            if (
                self.iou is None
                or self.precision is None
                or self.recall is None
                or self.f1_score is None
            ):
                raise ValueError(
                    "COMPUTED status requires iou, precision, recall, f1_score all not None"
                )
            if not (0.0 <= self.iou <= 1.0):
                raise ValueError(f"IoU must be in [0, 1], got {self.iou}")
            if not (0.0 <= self.precision <= 1.0):
                raise ValueError(f"Precision must be in [0, 1], got {self.precision}")
            if not (0.0 <= self.recall <= 1.0):
                raise ValueError(f"Recall must be in [0, 1], got {self.recall}")
            if not (0.0 <= self.f1_score <= 1.0):
                raise ValueError(f"F1 score must be in [0, 1], got {self.f1_score}")

        # UNAVAILABLE/UNKNOWN statuses must have None metrics
        if self.comparison_status in ("UNAVAILABLE", "UNKNOWN"):
            if (
                self.iou is not None
                or self.precision is not None
                or self.recall is not None
                or self.f1_score is not None
            ):
                raise ValueError(
                    f"Status {self.comparison_status} must have all metrics None"
                )

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence/display."""
        return {
            "comparison_status": self.comparison_status,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_cells": self.total_cells,
            "sentinel1_inundation_fraction": self.sentinel1_inundation_fraction,
            "model_inundation_fraction": self.model_inundation_fraction,
            "satellite_confidence": self.satellite_confidence,
            "coverage_fraction": self.coverage_fraction,
            "error_reason": self.error_reason,
            "limitations": self.limitations,
        }


def compute_sentinel1_comparison_metrics(
    scored_grid: gpd.GeoDataFrame,
    sentinel1_observation: Sentinel1ObservationResult,
) -> Sentinel1ComparisonMetrics:
    """
    Compute comparison metrics between Sentinel-1 observation and model predictions.

    SCIENTIFIC INTEGRITY:
    - Requires both model predictions (HIGH/MEDIUM/LOW/WATER in scored_grid)
      and valid Sentinel-1 observation
    - Metrics only computed if Sentinel-1 status is OBSERVED with flood_observed is not None
    - Returns UNAVAILABLE if Sentinel-1 data missing
    - Returns UNKNOWN if computational error occurs (no fabrication)

    Parameters
    ----------
    scored_grid : gpd.GeoDataFrame
        Scored grid with 'risk_class' column (values: "High", "Medium", "Low", "Water")
        and 'geometry' column (Point centroids or polygons).
    sentinel1_observation : Sentinel1ObservationResult
        Sentinel-1 observation with flood_observed and geometry info.

    Returns
    -------
    Sentinel1ComparisonMetrics
        Computed metrics or explicit fallback state (UNAVAILABLE/UNKNOWN)
    """

    # --- Validate Sentinel-1 observation ---
    if sentinel1_observation is None:
        logger.warning("No Sentinel-1 observation provided")
        return Sentinel1ComparisonMetrics(
            comparison_status="UNAVAILABLE",
            error_reason="No Sentinel-1 observation provided",
            limitations=[
                "Cannot compute comparison without Sentinel-1 observation.",
            ],
        )

    if sentinel1_observation.observation_status != "OBSERVED":
        logger.info(
            "Sentinel-1 observation not available (status: %s)",
            sentinel1_observation.observation_status,
        )
        return Sentinel1ComparisonMetrics(
            comparison_status="UNAVAILABLE",
            satellite_confidence=sentinel1_observation.confidence,
            error_reason=f"Sentinel-1 status: {sentinel1_observation.observation_status}",
            limitations=[
                f"Sentinel-1 observation status is {sentinel1_observation.observation_status}, not OBSERVED.",
            ],
        )

    if sentinel1_observation.flood_observed is None:
        logger.warning("Sentinel-1 observation has unknown flood status")
        return Sentinel1ComparisonMetrics(
            comparison_status="UNAVAILABLE",
            satellite_confidence=sentinel1_observation.confidence,
            error_reason="Sentinel-1 flood_observed is None",
            limitations=[
                "Sentinel-1 observation has unknown flood status (flood_observed=None).",
            ],
        )

    if scored_grid is None or len(scored_grid) == 0:
        logger.warning("Scored grid is empty or None")
        return Sentinel1ComparisonMetrics(
            comparison_status="UNAVAILABLE",
            error_reason="Scored grid is empty",
            limitations=["Cannot compute comparison without model predictions."],
        )

    try:
        # --- Extract predicted flood extent from scored_grid ---
        # "High" risk is classified as "flood prediction"
        # "Water" is pre-classified and excluded from comparison (separate water mask)
        model_flood_mask = (scored_grid["risk_class"] == "High").values

        # --- Extract Sentinel-1 observed flood extent ---
        # Build a binary mask of cells with observed flood (within Sentinel-1 bbox)
        sentinel1_bbox = sentinel1_observation.bbox  # (min_lon, min_lat, max_lon, max_lat)
        sentinel1_flood_mask = np.zeros(len(scored_grid), dtype=bool)

        # Filter grid cells within Sentinel-1 bbox
        valid_idx = (
            (scored_grid.geometry.bounds["minx"] >= sentinel1_bbox[0])
            & (scored_grid.geometry.bounds["maxx"] <= sentinel1_bbox[2])
            & (scored_grid.geometry.bounds["miny"] >= sentinel1_bbox[1])
            & (scored_grid.geometry.bounds["maxy"] <= sentinel1_bbox[3])
        )

        if not valid_idx.any():
            logger.warning(
                "No grid cells within Sentinel-1 bbox (%.2f, %.2f, %.2f, %.2f)",
                *sentinel1_bbox,
            )
            return Sentinel1ComparisonMetrics(
                comparison_status="UNAVAILABLE",
                satellite_confidence=sentinel1_observation.confidence,
                error_reason="No grid cells within Sentinel-1 bbox",
                coverage_fraction=0.0,
                limitations=[
                    "Sentinel-1 observation bbox does not overlap with analysis grid.",
                ],
            )

        # For cells within Sentinel-1 bbox, mark as flooded if Sentinel-1 observed flood
        # Note: This is a simplistic assumption that all cells in the bbox are uniformly
        # flooded if flood_observed=True. A more sophisticated approach would use
        # inundation_fraction or spatial information, but that requires geometry handling
        # beyond the scope of this basic comparison.
        if sentinel1_observation.flood_observed:
            sentinel1_flood_mask[valid_idx] = True

        total_cells_compared = np.sum(valid_idx)

        # --- Compute confusion matrix ---
        tp = np.sum(model_flood_mask[valid_idx] & sentinel1_flood_mask[valid_idx])
        fp = np.sum(model_flood_mask[valid_idx] & ~sentinel1_flood_mask[valid_idx])
        fn = np.sum(~model_flood_mask[valid_idx] & sentinel1_flood_mask[valid_idx])
        tn = np.sum(~model_flood_mask[valid_idx] & ~sentinel1_flood_mask[valid_idx])

        # --- Compute metrics ---
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # --- Calculate inundation fractions ---
        model_inundation = np.sum(model_flood_mask[valid_idx]) / total_cells_compared
        sentinel1_inundation = np.sum(sentinel1_flood_mask[valid_idx]) / total_cells_compared

        logger.info(
            "Sentinel-1 comparison: IoU=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f",
            iou,
            precision,
            recall,
            f1,
        )

        return Sentinel1ComparisonMetrics(
            comparison_status="COMPUTED",
            iou=float(iou),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            total_cells=int(total_cells_compared),
            sentinel1_inundation_fraction=float(sentinel1_inundation),
            model_inundation_fraction=float(model_inundation),
            satellite_confidence=sentinel1_observation.confidence,
            coverage_fraction=np.sum(valid_idx) / len(scored_grid),
            limitations=[
                "Simplified comparison: assumes uniform flood status within Sentinel-1 bbox.",
                "Does not account for Sentinel-1 inundation_fraction heterogeneity.",
                "Does not differentiate partial inundation from binary flooded/non-flooded.",
            ],
        )

    except Exception as exc:
        logger.error("Error computing Sentinel-1 comparison metrics: %s", exc)
        return Sentinel1ComparisonMetrics(
            comparison_status="UNKNOWN",
            satellite_confidence=sentinel1_observation.confidence,
            error_reason=f"Computation error: {str(exc)}",
            limitations=[
                "Metrics could not be computed due to a computational error.",
            ],
        )


def create_unavailable_comparison_metrics(
    reason: str = "Sentinel-1 data not available",
) -> Sentinel1ComparisonMetrics:
    """
    Create a terminal UNAVAILABLE comparison metrics result.

    Returns
    -------
    Sentinel1ComparisonMetrics
        Explicit UNAVAILABLE state (not fabricated metrics)
    """
    return Sentinel1ComparisonMetrics(
        comparison_status="UNAVAILABLE",
        error_reason=reason,
        limitations=[
            "Sentinel-1 comparison metrics are not available for this analysis.",
        ],
    )
