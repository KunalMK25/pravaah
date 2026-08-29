"""
PRAVAAH-AI — Habitation Intelligence Pipeline Orchestrator.

Extends the hazard analysis engine with the full habitation intelligence layer:
settlement ingestion → exposure analysis → vulnerability assessment →
carrying-capacity assessment → relocation priority scoring →
spatial zone classification → relocation candidate discovery →
agentic decision support.

Architecture:
  FloodRiskPipeline.run()      (Phase 1 — hazard analysis)
          ↓
  SIHPipeline.run_sih_stages() (Phase 2 — habitation intelligence)
          ↓
  SIHPipeline.run_phase3()     (Phase 3 — zones + candidates + agents)
          ↓
  FullSIHResult                (combined output)

The hazard engine is never modified — this pipeline adds new stages
exclusively downstream of FloodRiskResult.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
import numpy as np

from flood_risk_zonation.capacity.assessment import (
    assess_capacity,
    _load_healthcare,
    _load_roads,
)
from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exposure.analysis import analyse_exposure
from flood_risk_zonation.habitation.ingest import load_habitations
from flood_risk_zonation.models import (
    CarryingCapacityResult,
    ExposureResult,
    FloodRiskResult,
    FullSIHResult,
    SIHAnalysisResult,
    VulnerabilityResult,
)
from flood_risk_zonation.pipeline import FloodRiskPipeline
from flood_risk_zonation.population.factory import create_population_provider_chain
from flood_risk_zonation.relocation.priority import score_relocation_priority
from flood_risk_zonation.utils.routing import build_road_graph
from flood_risk_zonation.vulnerability.scorer import score_vulnerability

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str], None]]


def _extract_cell_stats(
    hab_lat: float,
    hab_lon: float,
    scored_grid: gpd.GeoDataFrame,
    n_cells: int = 4,
) -> dict:
    """
    Return mean physical stats for the N grid cells nearest to a habitation.

    Used to feed context-aware values into the vulnerability scorer without
    requiring a separate raster lookup.
    """
    import math

    if "centroid_lat" in scored_grid.columns:
        glat = scored_grid["centroid_lat"].values.astype(float)
        glon = scored_grid["centroid_lon"].values.astype(float)
    else:
        glat = scored_grid.geometry.centroid.y.values.astype(float)
        glon = scored_grid.geometry.centroid.x.values.astype(float)

    dlat = glat - hab_lat
    dlon = (glon - hab_lon) * math.cos(math.radians(hab_lat))
    dist_sq = dlat ** 2 + dlon ** 2
    n = min(n_cells, len(dist_sq))
    idx = np.argpartition(dist_sq, n - 1)[:n]
    nearby = scored_grid.iloc[idx]

    def _mean_col(col: str, default: float) -> float:
        if col not in nearby.columns:
            return default
        vals = nearby[col].dropna().values.astype(float)
        return float(vals.mean()) if len(vals) > 0 else default

    return {
        "elevation_m":         _mean_col("elevation_m", 50.0),
        "dist_water_m":        _mean_col("dist_water_m", 2000.0),
        "drainage_capacity":   _mean_col("drainage_capacity", 0.5),
    }


def _compute_area_reference_ranges(scored_grid: gpd.GeoDataFrame) -> dict:
    """
    Compute whole-area normalisation ranges from the scored grid.

    Returns a dict with (min, max) tuples for elevation, dist_water.
    These are passed to the vulnerability scorer so normalisation is
    relative to the actual study area, not arbitrary fixed values.
    """
    def _range(col: str, lo_default: float, hi_default: float) -> tuple:
        if col not in scored_grid.columns:
            return (lo_default, hi_default)
        vals = scored_grid[col].dropna().values.astype(float)
        if len(vals) < 2:
            return (lo_default, hi_default)
        p5, p95 = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
        if p5 >= p95:
            return (lo_default, hi_default)
        return (p5, p95)

    return {
        "elev_range":      _range("elevation_m", 0.0, 200.0),
        "dist_water_range":_range("dist_water_m", 0.0, 5000.0),
    }


class SIHPipeline:
    """
    SIH26191 Decision Support Pipeline.

    Wraps FloodRiskPipeline and adds habitation intelligence stages.

    Parameters
    ----------
    config : PipelineConfig | None
        Forwarded to the underlying FloodRiskPipeline.
    hab_cache_dir : str | Path
        Cache directory for habitation data.
    capacity_cache_dir : str | Path
        Cache directory for capacity (healthcare / road) data.
    allow_network : bool
        Whether live OSM / Overpass fetches are permitted.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        hab_cache_dir: str | Path = "data/cache/habitations",
        capacity_cache_dir: str | Path = "data/cache/capacity",
        allow_network: bool = True,
    ) -> None:
        self._config = config or PipelineConfig()
        self._hazard_pipeline = FloodRiskPipeline(self._config)
        self._hab_cache_dir = Path(hab_cache_dir)
        self._capacity_cache_dir = Path(capacity_cache_dir)
        self._allow_network = allow_network

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        bounding_box: BoundingBox,
        progress_callback: ProgressCallback = None,
    ) -> SIHAnalysisResult:
        """
        Run the full SIH pipeline (Phase 1 hazard + Phase 2 intelligence).

        Parameters
        ----------
        bounding_box : BoundingBox
        progress_callback : Callable[[str], None] | None

        Returns
        -------
        SIHAnalysisResult
        """
        def _cb(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        _cb("⛰️ Running hazard analysis (Phase 1)…")
        hazard_result = self._hazard_pipeline.run(
            bounding_box, progress_callback=progress_callback
        )
        return self.run_sih_stages(
            hazard_result, bounding_box, progress_callback=progress_callback
        )

    def run_sih_stages(
        self,
        hazard_result: FloodRiskResult,
        bounding_box: BoundingBox,
        progress_callback: ProgressCallback = None,
    ) -> SIHAnalysisResult:
        """
        Run only the SIH intelligence stages given an already-computed
        FloodRiskResult.

        This is the extension point for the Streamlit app — it can call the
        existing pipeline, then call this method to layer SIH intelligence
        on top without repeating the expensive hazard computation.

        Parameters
        ----------
        hazard_result : FloodRiskResult
            Output of FloodRiskPipeline.run() or run_from_ingested_data().
        bounding_box : BoundingBox
        progress_callback : Callable[[str], None] | None

        Returns
        -------
        SIHAnalysisResult
        """
        def _cb(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        t0 = time.time()
        scored_grid = hazard_result.scored_grid
        config = hazard_result.config

        # ── Stage 1: Habitation ingestion ──────────────────────────────────────
        _cb("🏘️ Loading habitation data…")
        hab_dataset = load_habitations(
            bounding_box,
            cache_dir=self._hab_cache_dir,
            allow_network=self._allow_network,
        )
        logger.info(
            "Habitation dataset: %d habitations (source: %s)",
            len(hab_dataset.habitations),
            hab_dataset.source,
        )

        if not hab_dataset.habitations:
            logger.warning("No habitations found — SIH stages will produce empty results.")
            return SIHAnalysisResult(
                flood_risk_result=hazard_result,
                habitation_dataset=hab_dataset,
                sih_duration_seconds=time.time() - t0,
            )

        # ── Stage 2: Exposure analysis ─────────────────────────────────────────
        _cb("🔍 Analysing habitation exposure…")
        
        # Create population provider chain (Phase 1B)
        population_chain = create_population_provider_chain(
            config.population_config if hasattr(config, 'population_config') else {}
        )
        
        # Compute bounding box for population aggregation
        bbox = (
            scored_grid["centroid_lon"].min(),
            scored_grid["centroid_lat"].min(),
            scored_grid["centroid_lon"].max(),
            scored_grid["centroid_lat"].max(),
        )
        
        exposure_results = analyse_exposure(
            hab_dataset,
            scored_grid,
            population_chain=population_chain,
            bbox=bbox,
            low_threshold=config.low_threshold,
            medium_threshold=config.medium_threshold,
        )

        if not exposure_results:
            return SIHAnalysisResult(
                flood_risk_result=hazard_result,
                habitation_dataset=hab_dataset,
                sih_duration_seconds=time.time() - t0,
            )

        # ── Stage 3: Capacity assessment (per habitation) ─────────────────────
        _cb("🏥 Assessing carrying capacity…")
        area_ranges = _compute_area_reference_ranges(scored_grid)

        # Load infrastructure ONCE for the entire analysis bounding box
        # to avoid redundant OSM API calls for each habitation
        from flood_risk_zonation.capacity.assessment import _bbox_expanded
        infra_bbox = _bbox_expanded(bounding_box, extra_deg=0.08)
        cache_path = Path(self._capacity_cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load healthcare and roads once, then reuse for all habitations
        hc_points = _load_healthcare(infra_bbox, cache_path, self._allow_network)
        road_points = _load_roads(infra_bbox, cache_path, self._allow_network)
        
        # Guard against large routing graphs that cause O(N²) complexity
        # For graphs >500 nodes, skip routing and use straight-line fallback
        MAX_ROUTING_NODES = 500
        if road_points and len(road_points) > MAX_ROUTING_NODES:
            logger.warning(
                "Routing graph skipped: %d road points exceeds maximum routing "
                "graph size of %d; using straight-line fallback.",
                len(road_points), MAX_ROUTING_NODES
            )
            road_graph = None
        else:
            road_graph = build_road_graph(road_points) if road_points else None

        capacity_results: list[CarryingCapacityResult] = []
        for exp in exposure_results:
            cap = assess_capacity(
                exp,
                scored_grid,
                bounding_box,
                cache_dir=self._capacity_cache_dir,
                allow_network=self._allow_network,
                hc_points=hc_points,
                road_points=road_points,
                road_graph=road_graph,
            )
            capacity_results.append(cap)

        capacity_map = {c.hab_id: c for c in capacity_results}

        # ── Stage 4: Vulnerability assessment ────────────────────────────────
        _cb("⚠️ Scoring vulnerability…")
        vulnerability_results: list[VulnerabilityResult] = []

        for exp in exposure_results:
            cap = capacity_map.get(exp.hab_id)
            cell_stats = _extract_cell_stats(exp.lat, exp.lon, scored_grid)

            vuln = score_vulnerability(
                exp,
                mean_elevation_m=cell_stats["elevation_m"],
                mean_dist_water_m=cell_stats["dist_water_m"],
                mean_drainage_capacity=cell_stats["drainage_capacity"],
                nearest_road_km=cap.nearest_road_km if cap else -1.0,
                nearest_healthcare_km=cap.nearest_healthcare_km if cap else -1.0,
                elev_range=area_ranges["elev_range"],
                dist_water_range=area_ranges["dist_water_range"],
            )
            vulnerability_results.append(vuln)

        vuln_map = {v.hab_id: v for v in vulnerability_results}

        # ── Stage 5: Relocation priority ──────────────────────────────────────
        _cb("🚨 Computing relocation priority…")
        relocation_results = []

        # Build a set of coastal-flagged habitation IDs from the hazard grid
        coastal_cell_ids: set[str] = set()
        if "is_coastal_tsunami_risk" in scored_grid.columns:
            coastal_cells = scored_grid[scored_grid["is_coastal_tsunami_risk"] == True]
            if "cell_id" in coastal_cells.columns:
                coastal_cell_ids = set(coastal_cells["cell_id"].values)

        for exp in exposure_results:
            vuln = vuln_map.get(exp.hab_id)
            cap = capacity_map.get(exp.hab_id)
            if vuln is None or cap is None:
                continue

            # Determine coastal flag for this habitation
            is_coastal = bool(
                set(exp.intersecting_cell_ids) & coastal_cell_ids
            )

            rel = score_relocation_priority(exp, vuln, cap, is_coastal=is_coastal)
            relocation_results.append(rel)

        # Sort relocation results by priority (CRITICAL first)
        _priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        relocation_results.sort(
            key=lambda r: (_priority_order.get(r.priority_class, 9), -r.relocation_score)
        )

        duration = time.time() - t0
        logger.info(
            "SIH stages complete in %.1fs. "
            "%d habitations | %d red zone | %d high/critical priority",
            duration,
            len(hab_dataset.habitations),
            sum(1 for e in exposure_results if e.is_in_red_zone),
            sum(1 for r in relocation_results if r.priority_class in ("HIGH", "CRITICAL")),
        )

        return SIHAnalysisResult(
            flood_risk_result=hazard_result,
            habitation_dataset=hab_dataset,
            exposure_results=exposure_results,
            vulnerability_results=vulnerability_results,
            capacity_results=capacity_results,
            relocation_results=relocation_results,
            sih_duration_seconds=duration,
        )

    def run_phase3(
        self,
        sih_result: SIHAnalysisResult,
        progress_callback: ProgressCallback = None,
        run_agents: bool = True,
        agent_priority_filter: tuple = ("CRITICAL", "HIGH", "MEDIUM", "LOW"),
        adjacency: str = "8-neighbour",
    ) -> "FullSIHResult":
        """
        Run Phase 3: spatial zone classification, relocation candidate
        discovery, and (optionally) agentic decision-support analysis.

        This method is additive — it wraps SIHAnalysisResult in a FullSIHResult
        without modifying the underlying hazard or habitation data.

        Parameters
        ----------
        sih_result : SIHAnalysisResult
            Output of run_sih_stages().
        progress_callback : Callable | None
        run_agents : bool
            If True, invoke the agentic orchestrator for each habitation.
            Set False to skip LLM calls and only compute zones + candidates.
        agent_priority_filter : tuple[str, ...]
            Only analyse habitations with these priority classes.
        adjacency : str
            "8-neighbour" (default) or "4-neighbour" for zone classification.

        Returns
        -------
        FullSIHResult
        """
        from flood_risk_zonation.spatial_zones.classifier import (
            classify_spatial_zones,
            get_zone_for_habitation,
        )
        from flood_risk_zonation.relocation.candidates import find_relocation_candidates
        from flood_risk_zonation.agents.orchestrator import PravaahOrchestrator

        def _cb(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        t0 = time.time()
        scored_grid = sih_result.flood_risk_result.scored_grid

        # ── Stage 6: Spatial zone classification ──────────────────────────────
        _cb("🗺️ Classifying spatial zones (RED/YELLOW/GREEN)…")
        zoned_grid = classify_spatial_zones(scored_grid, adjacency=adjacency)

        # ── Stage 7: Assign zone to each habitation ───────────────────────────
        habitation_zones: dict[str, str] = {}
        for exp in sih_result.exposure_results:
            zone = get_zone_for_habitation(exp.lat, exp.lon, zoned_grid)
            habitation_zones[exp.hab_id] = zone

        # ── Stage 8: Relocation candidate discovery ───────────────────────────
        _cb("🔍 Discovering relocation candidates…")
        relocation_candidates: dict[str, list] = {}
        cap_map = {c.hab_id: c for c in sih_result.capacity_results}

        # Only discover candidates for HIGH/CRITICAL habitations
        for rel in sih_result.relocation_results:
            if rel.priority_class not in ("HIGH", "CRITICAL"):
                continue
            exp = sih_result.get_exposure_by_id(rel.hab_id)
            if exp is None:
                continue
            cap = cap_map.get(rel.hab_id)
            candidates = find_relocation_candidates(
                hab_lat=exp.lat,
                hab_lon=exp.lon,
                hab_id=rel.hab_id,
                hab_name=rel.name or exp.name or "Unnamed",
                zoned_grid=zoned_grid,
                source_capacity=cap,
                search_radius_km=10.0,
                max_candidates=5,
            )
            relocation_candidates[rel.hab_id] = candidates
            logger.debug(
                "Candidates for %s: %d found", rel.hab_id, len(candidates)
            )

        # Assemble FullSIHResult (needed before agents)
        full_result = FullSIHResult(
            sih_result=sih_result,
            zoned_grid=zoned_grid,
            habitation_zones=habitation_zones,
            relocation_candidates=relocation_candidates,
            agent_decisions={},
            phase3_duration_seconds=time.time() - t0,
        )

        # ── Stage 9: Agentic decision support ─────────────────────────────────
        if run_agents and sih_result.relocation_results:
            _cb("🤖 Running agentic decision support…")
            orchestrator = PravaahOrchestrator(full_result, verbose=False)
            agent_decisions = orchestrator.analyse_all(
                priority_filter=agent_priority_filter,
                max_habitations=50,
            )
            full_result.agent_decisions = agent_decisions

        full_result.phase3_duration_seconds = time.time() - t0
        logger.info(
            "Phase 3 complete in %.1fs — "
            "RED=%d  YELLOW=%d  GREEN=%d  "
            "candidates=%d habitations  agents=%d decisions",
            full_result.phase3_duration_seconds,
            full_result.red_zone_count,
            full_result.yellow_zone_count,
            full_result.green_zone_count,
            len(relocation_candidates),
            len(full_result.agent_decisions),
        )
        return full_result
