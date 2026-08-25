"""
Exception hierarchy for PRAVAAH.

All custom exceptions inherit from FloodRiskError so callers can catch
the entire family with a single except clause when needed.
"""


class FloodRiskError(Exception):
    """Base exception for all flood risk zonation errors."""


class DataIngestionError(FloodRiskError):
    """Raised when a dataset cannot be loaded or parsed."""


class DataAlignmentError(FloodRiskError):
    """Raised when raster or vector datasets cannot be spatially aligned."""


class FeatureExtractionError(FloodRiskError):
    """Raised when feature computation fails for a grid cell or dataset."""


class ModelTrainingError(FloodRiskError):
    """Raised when model training or cross-validation fails."""


class ScoringError(FloodRiskError):
    """Raised when risk score normalization or classification fails."""


class ConfigurationError(FloodRiskError):
    """Raised when a configuration value is invalid or out of range."""
