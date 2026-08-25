"""PRAVAAH — spatial Red/Yellow/Green zone classification package."""
from flood_risk_zonation.spatial_zones.classifier import (
    classify_spatial_zones,
    ZONE_RED,
    ZONE_YELLOW,
    ZONE_GREEN,
    ZONE_WATER,
)

__all__ = [
    "classify_spatial_zones",
    "ZONE_RED",
    "ZONE_YELLOW",
    "ZONE_GREEN",
    "ZONE_WATER",
]
