"""
PRAVAAH-AI — Emergency Facilities & Evacuation Routes

METHODOLOGY (transparent, decision-support):
───────────────────────────────────────────────────────────────────────────
This module models emergency response data for hazard-aware decision support:

1. EmergencyFacility — hospitals, clinics, shelters, community centres
   sourced from OSM Overpass queries.

2. EvacuationRoute — a recommended route from a habitation to a facility,
   prioritizing hazard-zone avoidance (GREEN preferred > YELLOW > RED).

All routes are RECOMMENDATIONS ONLY. The system does NOT autonomously decide
evacuation policy. Authority decision-makers use route suggestions to inform
evacuation planning.

HAZARD AVOIDANCE STRATEGY:
  - Routes are computed using a hazard-weighted graph where:
    GREEN zones have weight = 1.0x (preferred)
    YELLOW zones have weight ≈ 2.0x (penalized but not avoided)
    RED zones have weight ≈ 20.0x (heavily penalized; avoided if possible)
    WATER zones are impassable
  - The routing engine finds the lowest-cost path, which naturally
    prefers longer GREEN routes over shorter RED routes.
  - Hazard exposure is calculated by sampling the route at regular intervals
    and reporting the % distance travelled through each zone.

NO EVACUATION AUTONOMY:
  - The system does NOT automatically dispatch people.
  - The system does NOT make official evacuation orders.
  - Authority decision-makers review recommendations, verify facility capacity,
    and issue official orders.

LIMITATIONS:
  - Facilities are matched to nearest road node; actual routing FROM facility
    is not performed (one-way, only TO facility).
  - No facility capacity constraints; routes are recommended without considering
    occupancy or carrying capacity.
  - Hazard zones are static (grid-based); real evacuation may encounter
    dynamic conditions (congestion, road damage).
  - "No safe route" typically means the route must pass through RED zones;
    this does NOT mean evacuation is impossible (may use secondary routes,
    temporary shelter, etc.).
───────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmergencyFacility:
    """
    A single emergency facility (hospital, clinic, shelter, community centre).

    Attributes
    ----------
    facility_id : str
        Unique identifier (e.g., "hospital_osm_123456").
    name : str
        Facility name from OSM or data source.
    facility_type : str
        One of: "hospital", "clinic", "health_centre", "doctors",
        "shelter", "community_centre".
    latitude : float
        WGS84 latitude of facility centre.
    longitude : float
        WGS84 longitude of facility centre.
    source : str
        Data source: "osm_overpass", "osm_cache", "curated", etc.
    osm_id : Optional[int]
        OpenStreetMap node/way ID if sourced from OSM.
    metadata : dict
        Additional attributes from OSM: amenity, description, contact:phone,
        operator, capacity, etc.
    """

    facility_id: str
    name: str
    facility_type: str
    latitude: float
    longitude: float
    source: str
    osm_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate facility type."""
        valid_types = {
            "hospital", "clinic", "health_centre", "doctors",
            "shelter", "community_centre"
        }
        if self.facility_type not in valid_types:
            raise ValueError(
                f"facility_type must be one of {valid_types}, "
                f"got '{self.facility_type}'"
            )

    def __repr__(self) -> str:
        return (
            f"EmergencyFacility(id={self.facility_id}, name={self.name!r}, "
            f"type={self.facility_type}, lat={self.latitude:.4f}, "
            f"lon={self.longitude:.4f})"
        )


@dataclass
class EvacuationRoute:
    """
    A recommended evacuation route from a habitation to an emergency facility.

    Attributes
    ----------
    hab_id : str
        Origin habitation ID.
    hab_name : str
        Origin habitation name.
    facility_id : str
        Destination facility ID.
    facility_name : str
        Destination facility name.
    facility_type : str
        Type of destination ("hospital", "shelter", etc.).
    route_geometry : list[tuple[float, float]]
        List of (latitude, longitude) coordinates along the route.
    distance_km : float
        Total route distance in kilometers.
    routing_method : str
        How distance was calculated: "network_routing" (Dijkstra on road graph)
        or "straight_line_fallback" (Haversine, if network unavailable).
    hazard_exposure : dict
        Dictionary with keys "RED", "YELLOW", "GREEN", "WATER" and
        float values representing percentage of route distance in each zone.
        Example: {"RED": 5.0, "YELLOW": 15.0, "GREEN": 80.0, "WATER": 0.0}
    status : str
        Route status. One of:
        - "FOUND": Route computed successfully.
        - "NO_SAFE_ROUTE_AVAILABLE": No facility reachable or all routes pass through RED.
        - "NO_FACILITY_AVAILABLE": No facilities exist in study area.
        - "FACILITY_UNREACHABLE": Specific facility not connected to road network.
        - "DATA_UNAVAILABLE": Graph or hazard data missing.
    details : str
        Explanation or debug information if status != "FOUND".
    """

    hab_id: str
    hab_name: str
    facility_id: str
    facility_name: str
    facility_type: str
    route_geometry: list[tuple[float, float]]
    distance_km: float
    routing_method: str
    hazard_exposure: dict
    status: str
    details: str = ""

    def __post_init__(self):
        """Validate status and hazard_exposure."""
        valid_statuses = {
            "FOUND",
            "NO_SAFE_ROUTE_AVAILABLE",
            "NO_FACILITY_AVAILABLE",
            "FACILITY_UNREACHABLE",
            "DATA_UNAVAILABLE",
        }
        if self.status not in valid_statuses:
            raise ValueError(
                f"status must be one of {valid_statuses}, got '{self.status}'"
            )

        # Validate hazard_exposure keys
        required_zones = {"RED", "YELLOW", "GREEN", "WATER"}
        if set(self.hazard_exposure.keys()) != required_zones:
            raise ValueError(
                f"hazard_exposure must have exactly keys {required_zones}, "
                f"got {set(self.hazard_exposure.keys())}"
            )

        # Validate hazard_exposure values sum to ~100 (allow 1% tolerance for rounding)
        # For failure states (NO_FACILITY_AVAILABLE, etc.), allow zero sum
        total = sum(self.hazard_exposure.values())
        is_failure_state = self.status in {"NO_FACILITY_AVAILABLE", "DATA_UNAVAILABLE", "FACILITY_UNREACHABLE"}
        if not is_failure_state and not (99.0 <= total <= 101.0):
            raise ValueError(
                f"hazard_exposure percentages must sum to ~100%, got {total}%"
            )
        if is_failure_state and total != 0.0:
            raise ValueError(
                f"hazard_exposure must be all zeros for failure state {self.status}, got {total}%"
            )

    def __repr__(self) -> str:
        return (
            f"EvacuationRoute(hab={self.hab_id}->{self.facility_id}, "
            f"distance={self.distance_km:.2f}km, status={self.status}, "
            f"hazard={{RED:{self.hazard_exposure['RED']:.1f}%, "
            f"YELLOW:{self.hazard_exposure['YELLOW']:.1f}%, "
            f"GREEN:{self.hazard_exposure['GREEN']:.1f}%}})"
        )

    def is_safe(self, red_threshold: float = 10.0) -> bool:
        """
        Convenience check: route is "safe" if RED zone exposure <= threshold.

        Parameters
        ----------
        red_threshold : float
            Maximum acceptable RED zone percentage. Default 10.0 (%).

        Returns
        -------
        bool
            True if RED exposure <= threshold AND status == "FOUND".
        """
        return (
            self.status == "FOUND"
            and self.hazard_exposure.get("RED", 0.0) <= red_threshold
        )

    def hazard_summary(self) -> str:
        """Return human-readable hazard exposure summary."""
        if self.status != "FOUND":
            return f"N/A ({self.status})"
        red = self.hazard_exposure.get("RED", 0.0)
        yellow = self.hazard_exposure.get("YELLOW", 0.0)
        green = self.hazard_exposure.get("GREEN", 0.0)
        water = self.hazard_exposure.get("WATER", 0.0)
        return (
            f"RED:{red:.1f}% | YELLOW:{yellow:.1f}% | "
            f"GREEN:{green:.1f}% | WATER:{water:.1f}%"
        )

