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
    Per-cell synthetic drainage capacity scores.

    Attributes
    ----------
    capacity_scores : np.ndarray
        1-D array of drainage capacity scores in [0, 1] — one per grid cell.
        1.0 = excellent drainage; 0.0 = no drainage capacity.
    cell_ids : list[str]
        Ordered list of cell identifiers matching capacity_scores.
    """

    capacity_scores: np.ndarray  # per-cell scores in [0, 1]
    cell_ids: list[str]


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
        "osm_tag" | "estimated" | "UNKNOWN"
    population_exposed : Optional[int]
        Population estimate with appropriate provenance labelling.
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
    is_in_red_zone: bool
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
