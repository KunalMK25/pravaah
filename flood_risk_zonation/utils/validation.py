"""
Input validation utilities for the Flood Risk Zonation System.

Functions
---------
validate_bounding_box  — raise ConfigurationError for invalid BoundingBox.
validate_config        — raise ConfigurationError for invalid PipelineConfig.
impute_missing_values  — fill NaN values in a NumPy array using mean imputation.
"""

from __future__ import annotations

import numpy as np

from flood_risk_zonation.exceptions import ConfigurationError


def validate_bounding_box(bbox: object) -> None:
    """
    Validate a BoundingBox instance and raise ConfigurationError with a
    descriptive message if any constraint is violated.

    The BoundingBox.__post_init__ already performs these checks; this
    function is provided as a standalone utility for callers that receive
    a pre-constructed BoundingBox and want to re-validate it explicitly.

    Parameters
    ----------
    bbox : BoundingBox
        The bounding box to validate.

    Raises
    ------
    ConfigurationError
        If any geographic constraint is violated.
    """
    min_lon = bbox.min_lon  # type: ignore[attr-defined]
    min_lat = bbox.min_lat  # type: ignore[attr-defined]
    max_lon = bbox.max_lon  # type: ignore[attr-defined]
    max_lat = bbox.max_lat  # type: ignore[attr-defined]

    if min_lon < -180:
        raise ConfigurationError(
            f"min_lon ({min_lon}) is out of range; must be >= -180."
        )
    if max_lon > 180:
        raise ConfigurationError(
            f"max_lon ({max_lon}) is out of range; must be <= 180."
        )
    if min_lat < -90:
        raise ConfigurationError(
            f"min_lat ({min_lat}) is out of range; must be >= -90."
        )
    if max_lat > 90:
        raise ConfigurationError(
            f"max_lat ({max_lat}) is out of range; must be <= 90."
        )
    if min_lon >= max_lon:
        raise ConfigurationError(
            f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})."
        )
    if min_lat >= max_lat:
        raise ConfigurationError(
            f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})."
        )


def validate_config(config: object) -> None:
    """
    Validate a PipelineConfig instance and raise ConfigurationError with a
    descriptive message if any constraint is violated.

    Parameters
    ----------
    config : PipelineConfig
        The pipeline configuration to validate.

    Raises
    ------
    ConfigurationError
        If any configuration constraint is violated.
    """
    cell_size = config.cell_size_meters  # type: ignore[attr-defined]
    low_t = config.low_threshold         # type: ignore[attr-defined]
    medium_t = config.medium_threshold   # type: ignore[attr-defined]

    if cell_size <= 0:
        raise ConfigurationError(
            f"cell_size_meters must be positive, got {cell_size}."
        )
    if low_t >= medium_t:
        raise ConfigurationError(
            f"low_threshold ({low_t}) must be strictly less than "
            f"medium_threshold ({medium_t})."
        )


def impute_missing_values(array: np.ndarray) -> np.ndarray:
    """
    Fill NaN values in *array* using mean imputation.

    - If the array contains at least one non-NaN value, NaN cells are
      replaced with the mean of all non-NaN values (np.nanmean).
    - If **all** values are NaN, the entire array is filled with 0.0.
    - Both 1-D and 2-D arrays are supported; the array is not modified
      in-place — a copy is returned.

    Parameters
    ----------
    array : np.ndarray
        Input array of any shape containing zero or more NaN values.

    Returns
    -------
    np.ndarray
        Array of the same shape and dtype (promoted to float64 if needed)
        with no NaN values.
    """
    result = array.astype(np.float64, copy=True)
    nan_mask = np.isnan(result)

    if not nan_mask.any():
        return result

    fill_value = np.nanmean(result)
    if np.isnan(fill_value):
        # All values were NaN
        fill_value = 0.0

    result[nan_mask] = fill_value
    return result
