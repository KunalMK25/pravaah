"""
Configuration dataclasses for PRAVAAH.

BoundingBox    — immutable geographic extent with validation.
PipelineConfig — all tunable parameters for a pipeline run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, radians
from pathlib import Path
from typing import Literal, Optional

from flood_risk_zonation.exceptions import ConfigurationError


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable geographic bounding box in WGS84 decimal degrees.

    Parameters
    ----------
    min_lon : float
        Western boundary (−180 … 180).
    min_lat : float
        Southern boundary (−90 … 90).
    max_lon : float
        Eastern boundary (−180 … 180), must be > min_lon.
    max_lat : float
        Northern boundary (−90 … 90), must be > min_lat.
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if self.min_lon >= self.max_lon:
            raise ConfigurationError(
                f"min_lon ({self.min_lon}) must be strictly less than max_lon ({self.max_lon})."
            )
        if self.min_lat >= self.max_lat:
            raise ConfigurationError(
                f"min_lat ({self.min_lat}) must be strictly less than max_lat ({self.max_lat})."
            )
        if self.min_lon < -180:
            raise ConfigurationError(
                f"min_lon ({self.min_lon}) is out of range; must be >= -180."
            )
        if self.max_lon > 180:
            raise ConfigurationError(
                f"max_lon ({self.max_lon}) is out of range; must be <= 180."
            )
        if self.min_lat < -90:
            raise ConfigurationError(
                f"min_lat ({self.min_lat}) is out of range; must be >= -90."
            )
        if self.max_lat > 90:
            raise ConfigurationError(
                f"max_lat ({self.max_lat}) is out of range; must be <= 90."
            )

    @property
    def center(self) -> tuple[float, float]:
        """Return (center_lat, center_lon) of the bounding box."""
        return (
            (self.min_lat + self.max_lat) / 2,
            (self.min_lon + self.max_lon) / 2,
        )

    @property
    def area_km2(self) -> float:
        """
        Approximate area in km² using an equirectangular projection.

        height_km = (max_lat - min_lat) * 111.32
        width_km  = (max_lon - min_lon) * 111.32 * cos(radians(center_lat))
        area_km2  = width_km * height_km
        """
        center_lat, _ = self.center
        height_km = (self.max_lat - self.min_lat) * 111.32
        width_km = (self.max_lon - self.min_lon) * 111.32 * cos(radians(center_lat))
        return width_km * height_km

    def width_km(self) -> float:
        """East-west extent in km."""
        _, center_lon = self.center
        center_lat, _ = self.center
        return (self.max_lon - self.min_lon) * 111.32 * cos(radians(center_lat))

    def height_km(self) -> float:
        """North-south extent in km."""
        return (self.max_lat - self.min_lat) * 111.32


@dataclass
class PipelineConfig:
    """
    All tunable parameters for a single flood risk pipeline run.

    Raises ConfigurationError in __post_init__ for invalid combinations.
    """

    cell_size_meters: float = 500.0
    model_type: Literal["random_forest", "lightgbm", "weighted_susceptibility", "ensemble"] = "ensemble"
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 5
    cv_folds: int = 5
    low_threshold: float = 33.0
    medium_threshold: float = 66.0
    use_cache: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    model_artifact_dir: Path = field(default_factory=lambda: Path("model/artifacts"))
    google_maps_api_key: Optional[str] = None
    random_seed: int = 42
    max_grid_cells: int = 100_000
    allow_network: bool = False

    def __post_init__(self) -> None:
        if self.cell_size_meters <= 0:
            raise ConfigurationError(
                f"cell_size_meters must be positive, got {self.cell_size_meters}."
            )
        if self.low_threshold >= self.medium_threshold:
            raise ConfigurationError(
                f"low_threshold ({self.low_threshold}) must be strictly less than "
                f"medium_threshold ({self.medium_threshold})."
            )
        if not 0.0 <= self.low_threshold <= 100.0:
            raise ConfigurationError("low_threshold must be within [0, 100].")
        if not 0.0 <= self.medium_threshold <= 100.0:
            raise ConfigurationError("medium_threshold must be within [0, 100].")
        if self.max_grid_cells <= 0:
            raise ConfigurationError("max_grid_cells must be a positive integer.")


# ── Bbox size limits ──────────────────────────────────────────────────────────
# These constants define the acceptable range for UI-submitted bounding boxes.
# Using km² area as the primary gate and per-side km as a secondary check so
# that degenerate thin-strip bboxes are also rejected.

BBOX_MIN_SIDE_KM: float = 2.0    # each side must be at least 2 km
BBOX_MAX_SIDE_KM: float = 50.0   # each side must be at most 50 km


def validate_bbox_size(bbox: BoundingBox) -> str | None:
    """
    Check that a BoundingBox is within the acceptable size range for the UI.

    Returns a friendly error string if the bbox is out of range, or None if OK.
    The check uses latitude-aware degree-to-km conversion (same formula as
    BoundingBox.area_km2) so it works correctly at any latitude.

    Parameters
    ----------
    bbox : BoundingBox

    Returns
    -------
    str | None
        Human-readable error message, or None if the bbox is acceptable.
    """
    center_lat = (bbox.min_lat + bbox.max_lat) / 2.0
    height_km = (bbox.max_lat - bbox.min_lat) * 111.32
    width_km = (bbox.max_lon - bbox.min_lon) * 111.32 * cos(radians(center_lat))

    if width_km < BBOX_MIN_SIDE_KM or height_km < BBOX_MIN_SIDE_KM:
        return (
            f"Selected area is too small ({width_km:.1f} km × {height_km:.1f} km). "
            f"Please select an area at least {BBOX_MIN_SIDE_KM:.0f} km wide and tall "
            "for meaningful results."
        )
    if width_km > BBOX_MAX_SIDE_KM or height_km > BBOX_MAX_SIDE_KM:
        return (
            f"Selected area is too large ({width_km:.1f} km × {height_km:.1f} km). "
            f"Please select an area no more than {BBOX_MAX_SIDE_KM:.0f} km wide and tall "
            "to keep processing times reasonable."
        )
    return None
