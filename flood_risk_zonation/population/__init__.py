"""
PRAVAAH-AI — Multi-source population data system.

Architecture:
    PopulationProvider (abstract)
        ├─ AuthoritativeProvider (Tier 1)
        ├─ RegionalProvider (Tier 2)
        ├─ WorldPopProvider (Tier 3)
        ├─ OSMProvider (Tier 4)
        ├─ DerivedProvider (Tier 5)
        └─ UnknownProvider (Tier 6)
    
    PopulationProviderChain orchestrates fallback.

Data structures:
    PopulationDataStatus: OBSERVED | ESTIMATED | CACHED | UNAVAILABLE | UNKNOWN
    PopulationProviderType: AUTHORITATIVE | REGIONAL | WORLDPOP | OSM | DERIVED | UNKNOWN
    PopulationMethod: how value was obtained
    PopulationResult: complete provenance + confidence

Scientific integrity:
    - Never fabricate
    - Never label derived as observed
    - Never silently replace
    - Always preserve provenance
    - Always preserve uncertainty
    - UNKNOWN is an explicit legitimate state
"""

from .enums import PopulationDataStatus, PopulationProviderType, PopulationMethod
from .result import PopulationResult
from .provider import PopulationProvider, UnknownProvider
from .chain import PopulationProviderChain
from .implementations import (
    AuthoritativeProvider,
    RegionalProvider,
    WorldPopProvider,
    OSMProvider,
    DerivedProvider,
)
from .factory import create_population_provider_chain

__all__ = [
    "PopulationDataStatus",
    "PopulationProviderType",
    "PopulationMethod",
    "PopulationResult",
    "PopulationProvider",
    "UnknownProvider",
    "PopulationProviderChain",
    "AuthoritativeProvider",
    "RegionalProvider",
    "WorldPopProvider",
    "OSMProvider",
    "DerivedProvider",
    "create_population_provider_chain",
]
