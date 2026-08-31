"""
Core data model dataclasses for PRAVAAH.

These are plain dataclasses (no validation logic) that act as typed
containers passed between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class RasterDataset:
    """
    A single-band raster dataset loaded from a GeoTIFF or similar source.

    Attributes
    ----------
    array : np.ndarray
        2-D float32 array of raster values.
    transform : Any
        Rasterio Affine transform mapping pixel coordinates to CRS coordinates.
    crs : Any
        PyProj CRS object describing the coordinate reference system.
    nodata : Optional[float]
        Sentinel value representing missing / no-data pixels.
    source : str
        Human-readable provenance string (e.g. file path or dataset name).
    """

    array: np.ndarray
    transform: Any  # rasterio.transform.Affine
    crs: Any        # pyproj.CRS
    nodata: Optional[float]
    source: str


@dataclass
class RainfallDataset:
    """
    Gridded rainfall statistics derived from GPM IMERG or IMD data.

    Attributes
    ----------
    mean_annual_mm : np.ndarray
        2-D array of mean annual rainfall in millimetres.
    max_24h_mm : np.ndarray
        2-D array of maximum 24-hour rainfall in millimetres.
    transform : Any
        Rasterio Affine transform.
    crs : Any
        PyProj CRS object.
    temporal_range : tuple
        (start_date, end_date) of the underlying data record.
    source : str
        Provenance string, e.g. "IMD", "NASA_GPM", or "synthetic".
    """

    mean_annual_mm: np.ndarray
    max_24h_mm: np.ndarray
    transform: Any
    crs: Any
    temporal_range: tuple
    source: str


@dataclass
class DrainageDataset:
    """
    Per-cell drainage feature scores used by the flood-risk model.

    The ``source`` attribute records whether the scores were derived from
    real mapped drainage infrastructure (OSM proxy) or a synthetic fallback:

    ``"osm_proxy"``
        Scores derived from the density and proximity of mapped OSM
        drainage linestrings (drain, canal, stream, river ways) within a
        search radius of each cell centroid. Spatially meaningful and
        deterministic. **Not** a hydraulic capacity measurement — labelled
        as a drainage infrastructure availability proxy.

    ``"synthetic_fallback"``
        Scores inversely correlated with population density (or uniform
        random [0.2, 1.0] if population is absent). Used only when no
        OSM linestring data is available.

    Attributes
    ----------
    capacity_scores : np.ndarray
        1-D array of drainage proxy scores in [0, 1] — one per grid cell.
        1.0 = dense nearby mapped drainage infrastructure;
        0.0 = no nearby mapped drainage features.
    cell_ids : list[str]
        Ordered list of cell identifiers matching capacity_scores.
    source : str
        Provenance tag: ``"osm_proxy"`` | ``"synthetic_fallback"``.
    """

    capacity_scores: np.ndarray  # per-cell scores in [0, 1]
    cell_ids: list[str]
    source: str = "synthetic_fallback"


@dataclass
class AnalysisResult:
    """Metadata and artefacts produced by a susceptibility analysis."""

    model: Any
    feature_names: list[str]
    feature_importances: dict[str, float]
    method: str
    validation_note: str
    # CV metrics — populated for RF and Ensemble methods
    mean_cv_auc: Optional[float] = None
    mean_cv_f1: Optional[float] = None
    mean_cv_accuracy: Optional[float] = None
    mean_cv_precision: Optional[float] = None
    mean_cv_recall: Optional[float] = None
    cv_auc_scores: Optional[list] = None
    cv_f1_scores: Optional[list] = None
    cv_accuracy_scores: Optional[list] = None
    cv_precision_scores: Optional[list] = None
    cv_recall_scores: Optional[list] = None
    scorer: Any = None  # FloodRiskScorer instance for probability calibration; used by scenarios


@dataclass
class TrainingResult:
    """
    Artefacts produced by a completed model training run.

    Attributes
    ----------
    model : Any
        Fitted scikit-learn or LightGBM estimator.
    feature_names : list[str]
        Ordered list of feature column names used during training.
    feature_importances : dict[str, float]
        Mapping of feature name → importance score (sums to ~1.0 for RF).
    cv_scores : dict[str, list[float]]
        Per-fold scores keyed by metric name, e.g. {"auc": [...], "f1": [...]}.
    mean_cv_auc : float
        Mean AUC-ROC across all cross-validation folds.
    mean_cv_f1 : float
        Mean F1 score across all cross-validation folds.
    training_timestamp : Any
        datetime object recording when training completed.
    """

    model: Any
    feature_names: list[str]
    feature_importances: dict[str, float]
    cv_scores: dict[str, list[float]]
    mean_cv_auc: float
    mean_cv_f1: float
    training_timestamp: Any  # datetime


@dataclass
class FloodRiskResult:
    """
    Complete output of a flood risk pipeline run.

    Attributes
    ----------
    scored_grid : Any
        gpd.GeoDataFrame with all grid cells, feature columns, risk_score,
        and risk_class populated.
    training_result : TrainingResult
        Model training artefacts from this run.
    bounding_box : BoundingBox
        Geographic extent that was analysed.
    config : PipelineConfig
        Configuration used for this run.
    pipeline_duration_seconds : float
        Wall-clock time for the full pipeline execution.
    cell_count : int
        Total number of grid cells in scored_grid.
    """

    scored_grid: Any  # gpd.GeoDataFrame
    analysis_result: AnalysisResult
    bounding_box: Any  # BoundingBox — avoid circular import at module level
    config: Any        # PipelineConfig
    pipeline_duration_seconds: float
    cell_count: int
    data_provenance: dict[str, str] = field(default_factory=dict)
    data_tier: int = 3
    sentinel1_observation: Any = None  # Sentinel1ObservationResult if available
    sentinel1_comparison_metrics: Any = None  # Sentinel1ComparisonMetrics if available

    @property
    def training_result(self) -> AnalysisResult:
        """Backward-compatible alias for callers using the pre-0.2 API."""
        return self.analysis_result

    @property
    def risk_distribution(self) -> dict[str, int]:
        """Return count of cells per risk class."""
        return self.scored_grid["risk_class"].value_counts().to_dict()

    @property
    def high_risk_cells(self) -> Any:
        """Return a GeoDataFrame containing only High-risk cells."""
        return self.scored_grid[self.scored_grid["risk_class"] == "High"]


# ═══════════════════════════════════════════════════════════════════════════════
# SIH26191 — HAZARD & VULNERABLE HABITATION DECISION SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Habitation:
    """
    A single settlement / habitation entity obtained from OSM or curated data.

    Attributes
    ----------
    hab_id : str
        Unique identifier (e.g. "osm_123456789").
    name : str
        Settlement name; empty string if unnamed.
    hab_type : str
        OSM place tag value: city | town | village | hamlet | suburb |
        neighbourhood | locality | isolated_dwelling | farm | allotments.
    lat : float
        Centroid latitude (WGS84).
    lon : float
        Centroid longitude (WGS84).
    source : str
        Data provenance: "osm_overpass" | "osm_cache" | "curated" | "fallback".
    population : Optional[int]
        Known population from OSM population tag; None if unavailable.
    osm_id : Optional[int]
        Raw OSM node/way/relation ID for traceability.
    metadata : dict
        Additional key-value pairs (e.g. admin_level, is_in).
    """

    hab_id: str
    name: str
    hab_type: str
    lat: float
    lon: float
    source: str
    population: Optional[int] = None
    osm_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class HabitationDataset:
    """
    Collection of habitation entities for a bounding box.

    Attributes
    ----------
    habitations : list[Habitation]
        All settlements found within the analysis bbox.
    source : str
        Provenance tag for the whole dataset.
    bbox_key : str
        String key identifying the bounding box (for cache lookups).
    """

    habitations: list  # list[Habitation]
    source: str
    bbox_key: str


@dataclass
class ExposureResult:
    """
    Exposure of a single habitation to the hazard grid.

    Attributes
    ----------
    hab_id : str
    name : str
    hab_type : str
    lat, lon : float
    hazard_score : float
        Mean hazard score of all grid cells that intersect / contain the point.
    hazard_class : str
        "High" | "Medium" | "Low" | "Water" — the dominant class in the
        intersecting cells.
    pct_high_risk : float
        Fraction (0–1) of intersecting cells classified as High or Critical.
    population_source : str
        "osm_tag" | "worldpop" | "regional" | "authoritative" | "unknown"
    population_exposed : Optional[int]
        Population count with appropriate provenance labelling.
    population_confidence : float
        [0.0, 1.0] — confidence in population data
    population_status : str
        "OBSERVED" | "ESTIMATED" | "CACHED" | "UNAVAILABLE" | "UNKNOWN"
    population_method : Optional[str]
        How population was obtained (e.g., "osm_tag_direct", "raster_aggregation")
    population_metadata : Optional[dict]
        Full PopulationResult as dict (for API/reports)
    is_in_red_zone : bool
        True when hazard_class is High.
    intersecting_cell_ids : list[str]
        Grid cell IDs that cover or neighbour this habitation point.
    """

    hab_id: str
    name: str
    hab_type: str
    lat: float
    lon: float
    hazard_score: float
    hazard_class: str
    pct_high_risk: float
    population_source: str
    population_exposed: Optional[int]
    population_confidence: float = 0.0
    population_status: str = "UNKNOWN"
    population_method: Optional[str] = None
    population_metadata: Optional[dict] = None
    is_in_red_zone: bool = False
    intersecting_cell_ids: list = field(default_factory=list)


@dataclass
class VulnerabilityResult:
    """
    Habitation-level vulnerability assessment.

    Uses a transparent weighted indicator model.  Each component is stored
    individually so the UI and report can show a full breakdown.

    Attributes
    ----------
    hab_id : str
    vulnerability_score : float
        Composite score in [0, 1].
    vulnerability_class : str
        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    component_scores : dict[str, float]
        Individual normalised component scores (each in [0, 1]).
    component_weights : dict[str, float]
        Declared weights used (sums to 1.0).
    factors : list[str]
        Human-readable list of the dominant contributing factors.
    """

    hab_id: str
    vulnerability_score: float
    vulnerability_class: str
    component_scores: dict = field(default_factory=dict)
    component_weights: dict = field(default_factory=dict)
    factors: list = field(default_factory=list)


@dataclass
class CarryingCapacityResult:
    """
    Carrying-capacity assessment for a single habitation.

    Attributes
    ----------
    hab_id : str
    capacity_score : float
        Composite score in [0, 1]; lower = more stressed.
    capacity_status : str
        "ADEQUATE" | "STRESSED" | "CRITICAL"
    safe_area_km2 : float
        Nearby low-risk land area within search_radius_km.
    search_radius_km : float
        Radius used for safe-area search (documented for provenance).
    nearest_healthcare_km : float
        Straight-line distance to nearest OSM hospital or clinic (km).
        -1 if not found.
    nearest_road_km : float
        Straight-line distance to nearest major road (km).
        -1 if not found.
    shelter_capacity : Optional[int]
        Known shelter capacity if a curated shelter dataset is available;
        None otherwise.
    shelter_source : str
        "curated" | "estimated" | "unavailable"
    notes : str
        Free-text capacity notes for UI display.
    """

    hab_id: str
    capacity_score: float
    capacity_status: str
    safe_area_km2: float
    search_radius_km: float
    nearest_healthcare_km: float
    nearest_road_km: float
    shelter_capacity: Optional[int] = None
    shelter_source: str = "unavailable"
    notes: str = ""


@dataclass
class RelocationPriorityResult:
    """
    Relocation priority assessment for a single habitation.

    Formula (declared, documented, transparent):
        relocation_score =
            w_hazard   × norm(hazard_score)
          + w_exposure  × norm(population_exposed)
          + w_vuln      × norm(vulnerability_score)
          + w_capacity  × (1 − norm(capacity_score))

    Action classes:
        LOW      ≤ 0.25   → Routine monitoring
        MEDIUM   ≤ 0.50   → Preparedness / monitoring
        HIGH     ≤ 0.75   → Priority intervention / evacuation planning
        CRITICAL > 0.75   → Immediate relocation priority

    Attributes
    ----------
    hab_id : str
    name : str
    relocation_score : float
        Composite score in [0, 1].
    priority_class : str
        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    recommended_action : str
        Plain-language recommendation for authority dashboard.
    contributing_factors : list[str]
        Ordered list of the main factors driving the priority.
    component_scores : dict[str, float]
        Raw component scores before weighting.
    weights : dict[str, float]
        Declared weights used.
    hazard_score : float
    vulnerability_score : float
    capacity_score : float
    population_exposed : Optional[int]
    population_source : str
    is_coastal : bool
    explanation : str
        Full structured narrative for the detail panel ("WHY CRITICAL").
    """

    hab_id: str
    name: str
    relocation_score: float
    priority_class: str
    recommended_action: str
    contributing_factors: list = field(default_factory=list)
    component_scores: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    hazard_score: float = 0.0
    vulnerability_score: float = 0.0
    capacity_score: float = 1.0
    population_exposed: Optional[int] = None
    population_source: str = "UNKNOWN"
    is_coastal: bool = False
    explanation: str = ""
    time_horizon: str = "MEDIUM-TERM"  # "SHORT-TERM" | "MEDIUM-TERM" | "LONG-TERM"
    time_horizon_explanation: str = ""


@dataclass
class AuthorityAlert:
    """
    Government authority alert derived from relocation priority and risk data.

    Attributes
    ----------
    alert_id : str
        Unique alert identifier (e.g., "PRAVAAH-2026-082701-001")
    severity : str
        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    affected_area : str
        Settlement name or geographic description
    affected_population : int
        Estimated affected population
    triggering_condition : str
        Plain-language description of why alert was generated
    evidence : dict
        Supporting metrics (hazard_score, priority_class, etc.)
    recommended_action : str
        Authority-facing recommendation
    relocation_horizon : str
        "SHORT-TERM" | "MEDIUM-TERM" | "LONG-TERM"
    authority_category : str
        "LOCAL" | "REGIONAL" | "NATIONAL" | "SPECIALIZED"
    generated_at : str
        ISO timestamp
    """
    alert_id: str
    severity: str
    affected_area: str
    affected_population: int
    triggering_condition: str
    evidence: dict = field(default_factory=dict)
    recommended_action: str = ""
    relocation_horizon: str = "MEDIUM-TERM"
    authority_category: str = "LOCAL"
    generated_at: str = ""


@dataclass
class SIHAnalysisResult:
    """
    Complete SIH26191 analysis result — wraps FloodRiskResult with all
    SIH-specific layers.

    Attributes
    ----------
    flood_risk_result : FloodRiskResult
        The underlying Phase 1 hazard analysis output.
    habitation_dataset : HabitationDataset
        All settlements found in the study area.
    exposure_results : list[ExposureResult]
        Per-habitation exposure assessments.
    vulnerability_results : list[VulnerabilityResult]
        Per-habitation vulnerability assessments.
    capacity_results : list[CarryingCapacityResult]
        Per-habitation carrying-capacity assessments.
    relocation_results : list[RelocationPriorityResult]
        Per-habitation relocation priority assessments.
    sih_duration_seconds : float
        Wall-clock time for SIH-specific stages only.
    """

    flood_risk_result: Any  # FloodRiskResult
    habitation_dataset: Any  # HabitationDataset
    exposure_results: list = field(default_factory=list)
    vulnerability_results: list = field(default_factory=list)
    capacity_results: list = field(default_factory=list)
    relocation_results: list = field(default_factory=list)
    sih_duration_seconds: float = 0.0

    # ── Convenience helpers ────────────────────────────────────────────────────

    @property
    def critical_habitations(self) -> list:
        """Return RelocationPriorityResult objects with CRITICAL priority."""
        return [r for r in self.relocation_results if r.priority_class == "CRITICAL"]

    @property
    def high_priority_habitations(self) -> list:
        """Return RelocationPriorityResult objects with HIGH or CRITICAL priority."""
        return [r for r in self.relocation_results if r.priority_class in ("HIGH", "CRITICAL")]

    @property
    def red_zone_habitations(self) -> list:
        """Return ExposureResult objects where the habitation is in a red/high-risk zone."""
        return [r for r in self.exposure_results if r.is_in_red_zone]

    def get_relocation_by_id(self, hab_id: str) -> "RelocationPriorityResult | None":
        for r in self.relocation_results:
            if r.hab_id == hab_id:
                return r
        return None

    def get_exposure_by_id(self, hab_id: str) -> "ExposureResult | None":
        for r in self.exposure_results:
            if r.hab_id == hab_id:
                return r
        return None

    def get_vulnerability_by_id(self, hab_id: str) -> "VulnerabilityResult | None":
        for r in self.vulnerability_results:
            if r.hab_id == hab_id:
                return r
        return None

    def get_capacity_by_id(self, hab_id: str) -> "CarryingCapacityResult | None":
        for r in self.capacity_results:
            if r.hab_id == hab_id:
                return r
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# PRAVAAH PHASE 3 — SPATIAL ZONES, RELOCATION CANDIDATES, AGENTIC LAYER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RelocationCandidate:
    """
    A potential relocation destination area identified from GREEN spatial zones.

    IMPORTANT: A RelocationCandidate is a DECISION-SUPPORT RECOMMENDATION.
    It is NOT a legally designated evacuation shelter, approved relocation
    site, or official safe zone.  It is a lower-risk area that scores well
    on measurable spatial and infrastructure factors and warrants further
    evaluation by the relevant authorities.

    Attributes
    ----------
    candidate_id : str
        Unique identifier for this candidate area (e.g. "cand_001").
    source_hab_id : str
        The habitation for which this candidate was identified.
    centroid_lat, centroid_lon : float
        Approximate centre of the candidate area.
    distance_km : float
        Straight-line distance from source habitation to candidate centre.
    area_km2 : float
        Estimated area of the candidate GREEN zone cluster (km²).
    candidate_score : float
        Composite candidate quality score in [0, 1]; higher = better.
    mean_hazard_score : float
        Mean hazard score of cells in this candidate area.
    nearest_road_km : float
        Estimated distance to nearest major road (km); -1 if unknown.
    nearest_healthcare_km : float
        Estimated distance to nearest healthcare facility (km); -1 if unknown.
    cell_count : int
        Number of grid cells in this candidate cluster.
    notes : str
        Plain-language summary of candidate strengths and constraints.
    data_provenance : str
        Source of the candidate ("spatial_zone_green").
    """

    candidate_id: str
    source_hab_id: str
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    area_km2: float
    candidate_score: float
    mean_hazard_score: float = 0.0
    nearest_road_km: float = -1.0
    nearest_healthcare_km: float = -1.0
    cell_count: int = 0
    notes: str = ""
    data_provenance: str = "spatial_zone_green"


@dataclass
class AgentEvidence:
    """
    Structured evidence record produced by a single PRAVAAH agent.

    Attributes
    ----------
    agent_name : str
        Identifier of the agent that produced this record.
    summary : str
        Plain-language summary of the agent's finding.
    severity : str
        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    key_factors : list[str]
        Ordered list of dominant factors driving the finding.
    metrics : dict
        Raw metric values the agent was given (for traceability).
    ai_assisted : bool
        True if an LLM contributed to this output; False if rule-based fallback.
    """

    agent_name: str
    summary: str
    severity: str
    key_factors: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    ai_assisted: bool = False


@dataclass
class AgentDecision:
    """
    Final structured decision produced by the PRAVAAH Agent Orchestrator.

    This is the authoritative output of the agentic layer.  It is always
    backed by structured GIS/ML evidence — the LLM (if available) provides
    explanatory language, but all numeric facts come from the pipeline.

    Attributes
    ----------
    hab_id : str
    hab_name : str
    priority_class : str
        "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" — from deterministic pipeline.
    relocation_score : float
        Deterministic relocation score [0, 1].
    spatial_zone : str
        "RED" | "YELLOW" | "GREEN" | "WATER"
    summary : str
        Plain-language summary of the overall decision.
    recommended_action : str
        Specific action recommendation.
    evidence : list[AgentEvidence]
        Per-agent evidence records.
    candidate_areas : list[RelocationCandidate]
        Ranked relocation candidate areas.
    top_candidate_reason : str
        Plain-language explanation of why the top candidate was recommended.
    ai_assisted : bool
        True if any LLM contributed; False if fully rule-based.
    fallback_reason : str
        If ai_assisted=False, explains why (e.g. "LLM unavailable").
    """

    hab_id: str
    hab_name: str
    priority_class: str
    relocation_score: float
    spatial_zone: str
    summary: str
    recommended_action: str
    evidence: list = field(default_factory=list)
    candidate_areas: list = field(default_factory=list)
    top_candidate_reason: str = ""
    ai_assisted: bool = False
    fallback_reason: str = ""


@dataclass
class FullSIHResult:
    """
    Extended SIH result that includes spatial zones, relocation candidates,
    and (optionally) agentic decision-support outputs.

    This wraps SIHAnalysisResult and adds the Phase 3 intelligence layers.
    The underlying SIHAnalysisResult is always present; the Phase 3 fields
    are populated when run_phase3() has been called.
    """

    sih_result: Any   # SIHAnalysisResult
    zoned_grid: Any = None    # gpd.GeoDataFrame with spatial_zone column
    habitation_zones: dict = field(default_factory=dict)
    # hab_id → spatial zone string (RED/YELLOW/GREEN/WATER)
    relocation_candidates: dict = field(default_factory=dict)
    # hab_id → list[RelocationCandidate]
    agent_decisions: dict = field(default_factory=dict)
    # hab_id → AgentDecision
    phase3_duration_seconds: float = 0.0

    # ── Convenience helpers ────────────────────────────────────────────────────

    @property
    def red_zone_count(self) -> int:
        from flood_risk_zonation.spatial_zones.classifier import ZONE_RED
        if self.zoned_grid is None:
            return 0
        return int((self.zoned_grid["spatial_zone"] == ZONE_RED).sum())

    @property
    def yellow_zone_count(self) -> int:
        from flood_risk_zonation.spatial_zones.classifier import ZONE_YELLOW
        if self.zoned_grid is None:
            return 0
        return int((self.zoned_grid["spatial_zone"] == ZONE_YELLOW).sum())

    @property
    def green_zone_count(self) -> int:
        from flood_risk_zonation.spatial_zones.classifier import ZONE_GREEN
        if self.zoned_grid is None:
            return 0
        return int((self.zoned_grid["spatial_zone"] == ZONE_GREEN).sum())

    def get_zone_for(self, hab_id: str) -> str:
        return self.habitation_zones.get(hab_id, "UNKNOWN")

    def get_candidates_for(self, hab_id: str) -> list:
        return self.relocation_candidates.get(hab_id, [])

    def get_decision_for(self, hab_id: str) -> "AgentDecision | None":
        return self.agent_decisions.get(hab_id)


# ═══════════════════════════════════════════════════════════════════════════════
# PRAVAAH INTELLIGENCE ENHANCEMENT — WEATHER, FORECAST, VALIDATION, SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WeatherObservation:
    """
    A single weather observation or short-term forecast point.

    Attributes
    ----------
    timestamp : str
        ISO-8601 datetime string (UTC).
    rainfall_mm : float
        Precipitation in millimetres.  -1.0 if unavailable.
    temperature_c : float
        Temperature in Celsius.  -999.0 if unavailable.
    humidity_pct : float
        Relative humidity [0–100].  -1.0 if unavailable.
    wind_speed_ms : float
        Wind speed in m/s.  -1.0 if unavailable.
    description : str
        Human-readable weather description (e.g. "heavy rain").
    """
    timestamp: str
    rainfall_mm: float = -1.0
    temperature_c: float = -999.0
    humidity_pct: float = -1.0
    wind_speed_ms: float = -1.0
    description: str = ""


@dataclass
class WeatherData:
    """
    Live or cached weather data for a geographic point.

    Attributes
    ----------
    lat, lon : float
        Query location.
    current : WeatherObservation | None
        Current conditions; None if unavailable.
    forecast : list[WeatherObservation]
        Short-term forecast observations (up to 72 h).
    source : str
        "openweather_live" | "openweather_cache" | "fallback" | "unavailable"
    fetched_at : str
        ISO-8601 UTC timestamp of when data was fetched.
    data_status : str
        "LIVE" | "CACHED" | "FALLBACK" | "UNAVAILABLE"
    location_name : str
        Location label returned by the provider.
    dynamic_risk_adjustment : float
        Normalised [0, 1] rainfall-based risk multiplier.
        0.0 = no additional risk from current weather.
        1.0 = maximum additional risk.
    dynamic_risk_reason : str
        Plain-language explanation of the adjustment (e.g. "Heavy rainfall (82 mm) elevates risk").
    """
    lat: float
    lon: float
    current: Optional[Any] = None       # WeatherObservation
    forecast: list = field(default_factory=list)  # list[WeatherObservation]
    source: str = "unavailable"
    fetched_at: str = ""
    data_status: str = "UNAVAILABLE"
    location_name: str = ""
    dynamic_risk_adjustment: float = 0.0
    dynamic_risk_reason: str = ""


@dataclass
class ForecastPoint:
    """
    A single forecast horizon risk projection for one location.

    Attributes
    ----------
    horizon_h : int
        Forecast horizon in hours (e.g. 24, 48, 72).
    forecast_rainfall_mm : float
        Total forecast precipitation for this horizon.
    baseline_risk_score : float
        ML hazard score without dynamic weather adjustment.
    adjusted_risk_score : float
        Hazard score after applying the rainfall-based dynamic adjustment.
    risk_change : float
        adjusted_risk_score - baseline_risk_score.  Positive = increased risk.
    spatial_zone : str
        Projected spatial zone at this horizon: RED | YELLOW | GREEN | WATER.
    confidence : str
        "HIGH" | "MEDIUM" | "LOW" — based on forecast data quality.
    provenance : str
        "forecast_rainfall_adjusted" | "baseline_only" | "unavailable"
    """
    horizon_h: int
    forecast_rainfall_mm: float
    baseline_risk_score: float
    adjusted_risk_score: float
    risk_change: float
    spatial_zone: str
    confidence: str = "MEDIUM"
    provenance: str = "forecast_rainfall_adjusted"


@dataclass
class ForecastResult:
    """
    Short-term (24–72 h) flood-risk forecast for a bounding box.

    IMPORTANT: This is a risk PROJECTION (estimate), not a deterministic
    prediction.  It combines the baseline ML hazard model with available
    precipitation forecast data.  Always labelled as FORECAST/ESTIMATE.

    Attributes
    ----------
    bbox_key : str
        Bounding box identifier.
    baseline_zone_counts : dict[str, int]
        Current zone counts (RED/YELLOW/GREEN/WATER) from the baseline run.
    horizons : list[ForecastPoint]
        Per-horizon projections (typically 24h, 48h, 72h).
    weather_source : str
        Provenance of the forecast precipitation data.
    forecast_timestamp : str
        When this forecast was generated (ISO-8601 UTC).
    methodology : str
        Plain-language description of the forecast approach.
    """
    bbox_key: str
    baseline_zone_counts: dict = field(default_factory=dict)
    horizons: list = field(default_factory=list)   # list[ForecastPoint]
    weather_source: str = "unavailable"
    forecast_timestamp: str = ""
    methodology: str = (
        "Baseline ML hazard susceptibility adjusted by forecast precipitation. "
        "ESTIMATE only — not a deterministic flood prediction."
    )

    def get_horizon(self, hours: int) -> "ForecastPoint | None":
        for h in self.horizons:
            if h.horizon_h == hours:
                return h
        return None


@dataclass
class HistoricalFloodEvent:
    """
    A single historical flood event used for independent model validation.

    Attributes
    ----------
    event_id : str
        Unique identifier (e.g. "bangalore_2022_09").
    event_name : str
        Human-readable event name.
    event_date : str
        Date or date-range string (ISO-8601 or descriptive).
    region : str
        Geographic region description.
    source : str
        Data source identifier (e.g. "Dartmouth_FO", "NASA_MODIS", "Manual").
    source_url : str
        URL or reference for traceability.
    flood_area_km2 : float
        Total observed flood extent in km² (-1 if unknown).
    affected_cells : list[str]
        Grid cell IDs that intersected the observed flood extent.
    flood_geojson : Any | None
        GeoJSON dict of the observed flood polygon, if available.
    notes : str
        Any caveats about data quality or coverage.
    """
    event_id: str
    event_name: str
    event_date: str
    region: str
    source: str
    source_url: str = ""
    flood_area_km2: float = -1.0
    affected_cells: list = field(default_factory=list)
    flood_geojson: Any = None
    notes: str = ""


@dataclass
class ValidationMetrics:
    """
    Spatial overlap metrics between PRAVAAH high-risk prediction and
    an independently observed flood extent.

    NOTE: These are INDEPENDENT VALIDATION metrics, distinct from the
    cross-validation metrics computed during ML training on WSI pseudo-labels.

    Attributes
    ----------
    event_id : str
    precision : float
        Fraction of predicted high-risk cells that were observed to flood.
    recall : float
        Fraction of observed flood cells that were predicted as high-risk.
    f1_score : float
        Harmonic mean of precision and recall.
    iou : float
        Intersection-over-Union of predicted and observed flood extents.
    predicted_high_count : int
    observed_flood_count : int
    overlap_count : int
    notes : str
    """
    event_id: str
    precision: float
    recall: float
    f1_score: float
    iou: float
    predicted_high_count: int
    observed_flood_count: int
    overlap_count: int
    notes: str = ""


@dataclass
class ValidationResult:
    """
    Complete historical validation output for one analysis run.

    Attributes
    ----------
    events : list[HistoricalFloodEvent]
        Validated events.
    metrics : list[ValidationMetrics]
        Per-event metrics.
    overall_notes : str
        Scientific caveats and methodology statement.
    data_status : str
        "VALIDATED" | "PARTIAL" | "NO_EVENTS_AVAILABLE"
    """
    events: list = field(default_factory=list)
    metrics: list = field(default_factory=list)
    overall_notes: str = (
        "Independent validation against historical flood observations. "
        "Metrics measure spatial overlap between PRAVAAH high-risk predictions "
        "and observed flood extents from independent sources. "
        "These are distinct from ML cross-validation metrics (which use WSI pseudo-labels)."
    )
    data_status: str = "NO_EVENTS_AVAILABLE"


@dataclass
class ScenarioParameters:
    """
    User-defined parameters for a what-if scenario simulation.

    Attributes
    ----------
    scenario_id : str
    label : str
        Human-readable label (e.g. "+30% Rainfall").
    rainfall_multiplier : float
        Multiply baseline rainfall by this factor. 1.0 = no change.
    extra_rainfall_mm : float
        Additional absolute rainfall added on top of the multiplier.
    population_multiplier : float
        Scale population density. 1.0 = no change.
    drainage_capacity_multiplier : float
        Scale drainage capacity. Values < 1 simulate degraded drainage.
    description : str
        User-supplied description.
    """
    scenario_id: str
    label: str
    rainfall_multiplier: float = 1.0
    extra_rainfall_mm: float = 0.0
    population_multiplier: float = 1.0
    drainage_capacity_multiplier: float = 1.0
    description: str = ""


@dataclass
class ScenarioResult:
    """
    Comparison between the baseline PRAVAAH run and a what-if scenario.

    IMPORTANT: Scenario results are clearly labelled SIMULATION/ESTIMATE.
    They must never overwrite or replace the baseline result.

    Attributes
    ----------
    scenario_id : str
    parameters : ScenarioParameters
    baseline_zone_counts : dict[str, int]
    scenario_zone_counts : dict[str, int]
    delta_zone_counts : dict[str, int]
        scenario − baseline for each zone.
    baseline_critical : int
    scenario_critical : int
    delta_critical : int
    baseline_high : int
    scenario_high : int
    habitations_escalated : list[str]
        hab_ids where relocation priority increased under the scenario.
    habitations_deescalated : list[str]
        hab_ids where relocation priority decreased.
    narrative : str
        Plain-language comparison summary.
    provenance : str
        Always "SIMULATION — user-defined parameter override".
    """
    scenario_id: str
    parameters: Any          # ScenarioParameters
    baseline_zone_counts: dict = field(default_factory=dict)
    scenario_zone_counts: dict = field(default_factory=dict)
    delta_zone_counts: dict = field(default_factory=dict)
    baseline_critical: int = 0
    scenario_critical: int = 0
    delta_critical: int = 0
    baseline_high: int = 0
    scenario_high: int = 0
    habitations_escalated: list = field(default_factory=list)
    habitations_deescalated: list = field(default_factory=list)
    narrative: str = ""
    provenance: str = "SIMULATION — user-defined parameter override"


@dataclass
class SHAPExplanation:
    """
    SHAP explanation for a single grid cell's ML hazard score.

    Attributes
    ----------
    cell_id : str
    shap_values : dict[str, float]
        Feature name → SHAP value (positive = increases risk score).
    base_value : float
        Model expected value (average prediction across training data).
    predicted_value : float
        The model's actual prediction for this cell.
    top_positive_features : list[tuple[str, float]]
        Features driving risk up, sorted by |SHAP|.
    top_negative_features : list[tuple[str, float]]
        Features driving risk down, sorted by |SHAP|.
    explanation_text : str
        Plain-language summary.
    provenance : str
        "shap_tree_explainer" | "shap_kernel_explainer" | "unavailable"
    """
    cell_id: str
    shap_values: dict = field(default_factory=dict)
    base_value: float = 0.0
    predicted_value: float = 0.0
    top_positive_features: list = field(default_factory=list)
    top_negative_features: list = field(default_factory=list)
    explanation_text: str = ""
    provenance: str = "shap_tree_explainer"
