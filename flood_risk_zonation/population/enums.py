"""Population data enums — status, provider type, method."""
from enum import Enum


class PopulationDataStatus(str, Enum):
    """Data status — whether value is observed, estimated, or unavailable."""

    OBSERVED = "OBSERVED"           # Real data from authoritative/reliable source
    ESTIMATED = "ESTIMATED"         # Derived/synthetic estimate
    SYNTHETIC = "SYNTHETIC"         # Synthetic fallback (lowest confidence)
    CACHED = "CACHED"               # Previously retrieved, now stale but valid
    UNAVAILABLE = "UNAVAILABLE"     # Data source configured but has no coverage
    UNKNOWN = "UNKNOWN"             # No data source available


class PopulationProviderType(str, Enum):
    """Provider tier — which source provided the data."""

    AUTHORITATIVE = "AUTHORITATIVE"   # Tier 1: Local census / government
    REGIONAL = "REGIONAL"             # Tier 2: Regional public database
    WORLDPOP = "WORLDPOP"             # Tier 3: WorldPop gridded raster
    OSM = "OSM"                       # Tier 4: OpenStreetMap tags
    DERIVED = "DERIVED"               # Tier 5: Estimates (building count, etc.)
    SYNTHETIC = "SYNTHETIC"           # Tier 6: Synthetic/fallback estimates
    UNKNOWN = "UNKNOWN"               # Tier 7: No data available


class PopulationMethod(str, Enum):
    """Method used to obtain population value."""

    OSM_TAG_DIRECT = "osm_tag_direct"                 # From OSM population tag
    RASTER_AGGREGATION = "raster_aggregation"         # Aggregated from gridded raster
    RASTER_NEAREST = "raster_nearest"                 # Nearest raster pixel
    BUILDING_COUNT = "building_count"                 # Estimated from building count
    LANDUSE_DENSITY = "landuse_density"               # Estimated from landuse area
    CENSUS_SPATIAL_JOIN = "census_spatial_join"       # From census via spatial join
    REGIONAL_INTERPOLATION = "regional_interpolation" # Interpolated from regional data
    SYNTHETIC = "synthetic"                           # Synthetic/fallback
