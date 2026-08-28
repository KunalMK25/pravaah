"""
Pipeline orchestrator for PRAVAAH.

Executes the full DAG: validate → grid → ingest → features → model → score.
Implements a three-tier data fallback strategy (real → partial → synthetic).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import geopandas as gpd
import numpy as np

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.exceptions import FloodRiskError
from flood_risk_zonation.features.extractor import FEATURE_COLUMNS, extract_features
from flood_risk_zonation.grid.generator import generate_grid
from flood_risk_zonation.ingest.drainage import generate_drainage_proxy, generate_synthetic_drainage
from flood_risk_zonation.ingest.elevation import generate_synthetic_elevation
from flood_risk_zonation.ingest.population import load_population
from flood_risk_zonation.ingest.rainfall import generate_synthetic_rainfall
from flood_risk_zonation.ingest.water_bodies import load_water_bodies
from flood_risk_zonation.models import AnalysisResult, FloodRiskResult
from flood_risk_zonation.scoring.scorer import FloodRiskScorer
from flood_risk_zonation.scoring.susceptibility import WeightedSusceptibilityModel, RandomForestSusceptibilityModel, EnsembleSusceptibilityModel
from flood_risk_zonation.satellite.sentinel1 import load_sentinel1_observation
from flood_risk_zonation.utils.cache import cache_key, get_cache_path, is_cached, load_geodataframe, save_geodataframe
from flood_risk_zonation.utils.validation import validate_bounding_box, validate_config

# Optional callback invoked with a short human-readable status string at each
# major pipeline stage. Signature: (message: str) -> None. Passing None (the
# default) disables progress reporting entirely — existing callers are
# unaffected. UI layers (e.g. Streamlit) can supply e.g. `st.write`.
ProgressCallback = Optional[Callable[[str], None]]

logger = logging.getLogger(__name__)

_LAND_MASK_PATH = Path(__file__).parent.parent / "data" / "landmask" / "ne_50m_land.geojson"
_LAND_MASK_GEOM: "shapely.geometry.base.BaseGeometry | None" = None


def _load_land_mask():
    """Load and cache the Natural Earth land polygon from disk.

    Returns the dissolved land geometry (a single Shapely BaseGeometry).
    Raises FloodRiskError if the file is missing, unparseable, or empty.
    """
    global _LAND_MASK_GEOM
    if _LAND_MASK_GEOM is not None:
        return _LAND_MASK_GEOM
    if not _LAND_MASK_PATH.exists():
        raise FloodRiskError(f"Land mask file not found: {_LAND_MASK_PATH}")
    try:
        gdf = gpd.read_file(_LAND_MASK_PATH)
    except Exception as exc:
        raise FloodRiskError(f"Failed to parse land mask: {exc}") from exc
    from shapely.ops import unary_union as _unary_union
    geom = _unary_union(gdf.geometry.tolist())
    if geom is None or geom.is_empty:
        raise FloodRiskError("Land mask geometry is empty after dissolve")
    _LAND_MASK_GEOM = geom
    return _LAND_MASK_GEOM


class FloodRiskPipeline:
    """
    End-to-end flood risk zonation pipeline.

    Parameters
    ----------
    config : PipelineConfig
        All tunable parameters for this run.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._data_tier: int = 3  # 1=real, 2=partial, 3=synthetic

    def run(
        self,
        bounding_box: BoundingBox,
        progress_callback: ProgressCallback = None,
    ) -> FloodRiskResult:
        """
        Execute the full pipeline for a given bounding box.

        This is the canonical, single-source-of-truth entry point for the
        real/live data path: it ingests elevation, rainfall, water bodies,
        and population (each with its own real → API → synthetic fallback),
        then delegates grid generation through final result assembly to
        :meth:`run_from_ingested_data`, which is shared with any caller that
        supplies its own pre-fetched datasets (e.g. an offline/demo mode).

        Parameters
        ----------
        bounding_box : BoundingBox
            Geographic extent to analyse.
        progress_callback : Callable[[str], None] | None
            Optional hook invoked with a short status string at each major
            stage (e.g. for a UI to display "Fetching elevation…"). Ignored
            (no-op) when None — existing callers are unaffected.

        Returns
        -------
        FloodRiskResult
            The scored GeoDataFrame plus model/config/provenance metadata.
        """
        t0 = time.time()
        config = self.config

        # --- Validate inputs ---
        validate_bounding_box(bounding_box)
        validate_config(config)

        seed = config.random_seed

        # --- Sentinel-1 satellite observation (optional, metadata-only) ---
        sentinel1_observation = None
        try:
            # Load Sentinel-1 observation from optional input paths
            # If neither path is provided, returns UNKNOWN gracefully
            sentinel1_geotiff = getattr(config, 'sentinel1_geotiff_path', None)
            sentinel1_geojson = getattr(config, 'sentinel1_geojson_path', None)
            sentinel1_observation = load_sentinel1_observation(
                bounding_box,
                geotiff_path=sentinel1_geotiff,
                geojson_path=sentinel1_geojson,
            )
            logger.info(
                "Sentinel-1 observation status: %s (confidence: %.2f)",
                sentinel1_observation.observation_status,
                sentinel1_observation.confidence,
            )
        except Exception as exc:
            logger.warning("Sentinel-1 observation unavailable: %s", exc)
            sentinel1_observation = None

        # --- Data ingestion — use real data if available, else synthetic fallback ---
        logger.info("Ingesting data...")
        provenance: dict[str, str] = {}

        # Elevation — search ALL tif files, not just Gottigere
        if progress_callback:
            progress_callback("⛰️ Fetching elevation…")
        elev_dir = Path("data/elevation")
        elevation = None
        if elev_dir.exists():
            from flood_risk_zonation.ingest.elevation import load_elevation
            try:
                elevation = load_elevation(bounding_box, elev_dir)
                logger.info("Real elevation loaded from %s.", elevation.source)
            except FloodRiskError as exc:
                logger.warning("Real elevation unavailable (%s).", exc)
        # Fallback 1: fetch from OpenTopoData SRTM API (gives real ocean=0 values)
        if elevation is None and config.allow_network:
            from flood_risk_zonation.ingest.elevation import fetch_elevation_api
            logger.info("Fetching elevation from OpenTopoData SRTM API...")
            elevation = fetch_elevation_api(bounding_box, resolution_m=500)
            if elevation is not None:
                logger.info("OpenTopoData elevation fetched successfully.")
        # Fallback 2: synthetic (no ocean detection possible)
        if elevation is None:
            logger.warning("No SRTM file or API available, using synthetic elevation.")
            elevation = generate_synthetic_elevation(bounding_box, resolution_m=500, seed=seed)
        provenance["elevation"] = elevation.source

        # Water bodies — fetch live from Overpass API for any bbox worldwide
        # Results are cached locally so subsequent runs are instant
        if progress_callback:
            progress_callback("💧 Fetching water bodies…")
        logger.info("Fetching water bodies from Overpass API...")
        water_bodies = load_water_bodies(
            bounding_box,
            data_dir="data/water_bodies",
            allow_network=config.allow_network,
        )
        logger.info("Water bodies loaded: %d features.", len(water_bodies))
        provenance["water_bodies"] = water_bodies.attrs.get("source", "unavailable")

        # Rainfall
        if progress_callback:
            progress_callback("🌧️ Fetching rainfall…")
        rain_dir = Path("data/rainfall")
        if list(rain_dir.glob("*.tif")):
            from flood_risk_zonation.ingest.rainfall import load_rainfall
            try:
                rainfall = load_rainfall(bounding_box, rain_dir)
                logger.info("Real rainfall loaded.")
            except Exception as e:
                logger.warning("Real rainfall failed (%s), using synthetic.", e)
                rainfall = generate_synthetic_rainfall(bounding_box, resolution_m=1000, seed=seed)
        else:
            rainfall = generate_synthetic_rainfall(bounding_box, resolution_m=1000, seed=seed)
        provenance["rainfall"] = rainfall.source

        population = load_population(bounding_box, data_dir=str(config.cache_dir))
        provenance["population"] = population.source

        # Add Sentinel-1 observation metadata to provenance
        if sentinel1_observation:
            provenance["sentinel1_status"] = sentinel1_observation.observation_status
            provenance["sentinel1_confidence"] = str(sentinel1_observation.confidence)
            provenance["sentinel1_source"] = sentinel1_observation.source
            provenance["sentinel1_platform"] = sentinel1_observation.platform

        core_real = [
            provenance["elevation"] != "synthetic",
            provenance["rainfall"] != "synthetic",
            provenance["water_bodies"] in {"osm_overpass", "osm_cache"},
        ]
        data_tier = 1 if all(core_real) else (2 if any(core_real) else 3)

        return self.run_from_ingested_data(
            bounding_box=bounding_box,
            elevation=elevation,
            rainfall=rainfall,
            water_bodies=water_bodies,
            population=population,
            provenance=provenance,
            data_tier=data_tier,
            sentinel1_observation=sentinel1_observation,
            progress_callback=progress_callback,
            start_time=t0,
        )

    def run_from_ingested_data(
        self,
        bounding_box: BoundingBox,
        elevation,
        rainfall,
        water_bodies: gpd.GeoDataFrame,
        population,
        provenance: dict[str, str],
        data_tier: int,
        progress_callback: ProgressCallback = None,
        start_time: float | None = None,
        sentinel1_observation = None,
    ) -> FloodRiskResult:
        """
        Execute the pipeline stages shared by every caller, given
        already-fetched (or synthetically generated) input datasets.

        :meth:`run` calls this after its own real/API/synthetic ingestion.
        Callers with a fundamentally different data source — e.g. the
        offline/demo UI mode, which serves deterministic per-region sample
        data instead of fetching real elevation/rainfall/water bodies — can
        call this directly instead, so that grid generation, drainage
        synthesis, feature extraction, susceptibility modelling, scoring,
        and water-mask post-processing remain a single implementation
        rather than being duplicated per caller.

        Parameters
        ----------
        bounding_box : BoundingBox
            Geographic extent to analyse.
        elevation : RasterDataset
        rainfall : RainfallDataset
        water_bodies : gpd.GeoDataFrame
        population : RasterDataset
        provenance : dict[str, str]
            Provenance so far (must include "elevation", "rainfall",
            "water_bodies", "population" keys); "drainage" is added here.
        data_tier : int
            1 = all core real data, 2 = partial, 3 = fully synthetic.
            Callers are responsible for computing this (see `run()` for the
            real-data-path calculation); offline/demo callers should pass 3.
        progress_callback : Callable[[str], None] | None
            Optional status hook — see `run()`.
        start_time : float | None
            `time.time()` value to measure `pipeline_duration_seconds` from.
            `run()` passes its own pre-ingestion timestamp so the reported
            duration covers ingestion too, matching prior behaviour. Callers
            that skip `run()` (e.g. offline/demo mode) may omit this — it
            then defaults to the start of this method, i.e. duration covers
            only grid generation through scoring/masking for that caller.

        Returns
        -------
        FloodRiskResult
        """
        t0 = start_time if start_time is not None else time.time()
        config = self.config
        seed = config.random_seed

        validate_bounding_box(bounding_box)
        validate_config(config)

        # --- Grid generation (with optional cache) ---
        ck = cache_key(bounding_box, config)
        cache_path = get_cache_path(ck + "_grid", config.cache_dir)

        if config.use_cache and is_cached(ck + "_grid", config.cache_dir):
            logger.info("Loading grid from cache: %s", cache_path)
            grid = load_geodataframe(cache_path)
        else:
            logger.info("Generating grid (cell_size=%dm)…", int(config.cell_size_meters))
            grid = generate_grid(
                bounding_box,
                config.cell_size_meters,
                max_cells=config.max_grid_cells,
            )
            if config.use_cache:
                save_geodataframe(grid, cache_path)

        drainage = generate_drainage_proxy(
            grid,
            water_bodies,
            cell_size_m=config.cell_size_meters,
            seed=seed,
        )
        provenance = dict(provenance)  # avoid mutating the caller's dict
        provenance["drainage"] = drainage.source

        if progress_callback:
            progress_callback("🔬 Computing features…")
        logger.info("Extracting features for %d cells…", len(grid))
        featured_grid = extract_features(
            grid, elevation, rainfall, water_bodies, population, drainage
        )

        # --- Susceptibility model ---
        # WSI: transparent weighted index, no training needed.
        # RF: Random Forest trained on WSI pseudo-labels with 5-fold CV.
        # Ensemble (default): blends WSI + RF, reports full CV metrics.
        if progress_callback:
            progress_callback("🤖 Running susceptibility model…")
        X = featured_grid[FEATURE_COLUMNS].copy()
        model_type = getattr(config, "model_type", "ensemble")

        if model_type == "ensemble":
            logger.info("Training Ensemble (WSI + RF) susceptibility model…")
            model = EnsembleSusceptibilityModel(
                n_estimators=config.rf_n_estimators,
                cv_folds=config.cv_folds,
                random_state=config.random_seed,
            ).fit(X)
            analysis_result = AnalysisResult(
                model=model,
                feature_names=list(model.feature_names),
                feature_importances=model.feature_importances,
                method="ensemble",
                validation_note=(
                    f"Ensemble (WSI + RF blend). "
                    f"5-fold CV — AUC: {model.mean_cv_auc:.3f}, "
                    f"F1: {model.mean_cv_f1:.3f}, "
                    f"Accuracy: {model.mean_cv_accuracy:.3f}. "
                    "Labels derived from WSI; not calibrated against observed flood events."
                ),
                mean_cv_auc=model.mean_cv_auc,
                mean_cv_f1=model.mean_cv_f1,
                mean_cv_accuracy=model.mean_cv_accuracy,
                mean_cv_precision=model.mean_cv_precision,
                mean_cv_recall=model.mean_cv_recall,
                cv_auc_scores=model.cv_auc_scores,
                cv_f1_scores=model.cv_f1_scores,
                cv_accuracy_scores=model.cv_accuracy_scores,
                cv_precision_scores=model.cv_precision_scores,
                cv_recall_scores=model.cv_recall_scores,
            )
        elif model_type == "random_forest":
            logger.info("Training Random Forest susceptibility model…")
            model = RandomForestSusceptibilityModel(
                n_estimators=config.rf_n_estimators,
                cv_folds=config.cv_folds,
                random_state=config.random_seed,
            ).fit(X)
            analysis_result = AnalysisResult(
                model=model,
                feature_names=list(model.feature_names),
                feature_importances=model.feature_importances,
                method="random_forest",
                validation_note=(
                    f"Random Forest trained on WSI pseudo-labels. "
                    f"5-fold CV — AUC: {model.mean_cv_auc:.3f}, F1: {model.mean_cv_f1:.3f}. "
                    "Labels derived from WSI; not calibrated against observed flood events."
                ),
                mean_cv_auc=model.mean_cv_auc,
                mean_cv_f1=model.mean_cv_f1,
                cv_auc_scores=model.cv_auc_scores,
                cv_f1_scores=model.cv_f1_scores,
            )
        else:
            # Weighted Susceptibility Index (fully transparent, no training)
            model = WeightedSusceptibilityModel().fit(X)
            analysis_result = AnalysisResult(
                model=model,
                feature_names=list(model.feature_names),
                feature_importances=model.feature_importances,
                method="weighted_susceptibility_index",
                validation_note=(
                    "Relative susceptibility index; not calibrated against observed flood events."
                ),
            )

        # --- Risk scoring ---
        if progress_callback:
            progress_callback("🗺️ Scoring and rendering map…")
        logger.info("Scoring grid…")
        scorer = FloodRiskScorer()
        scorer.p_min = 0.0
        scorer.p_max = 1.0
        thresholds = {"low_max": config.low_threshold, "medium_max": config.medium_threshold}
        scored_grid = scorer.score_grid(featured_grid, model, FEATURE_COLUMNS, thresholds)

        # --- Post-processing: water masking + proximity boosting ---
        scored_grid = self._apply_water_mask_and_proximity_boost(
            scored_grid, water_bodies, config,
            elevation_source=provenance.get("elevation", "synthetic"),
        )

        duration = time.time() - t0
        self._data_tier = data_tier
        logger.info("Pipeline complete in %.1fs. Cells: %d", duration, len(scored_grid))

        return FloodRiskResult(
            scored_grid=scored_grid,
            analysis_result=analysis_result,
            bounding_box=bounding_box,
            config=config,
            pipeline_duration_seconds=duration,
            cell_count=len(scored_grid),
            data_provenance=provenance,
            data_tier=data_tier,
        )


    def _apply_water_mask_and_proximity_boost(
        self,
        scored_grid,
        water_bodies,
        config,
        elevation_source="real",
    ):
        """
        Post-scoring pipeline:
        1. ELEVATION MASK  - cells with elevation <= 2 m -> Water.
                             Uses real SRTM from OpenTopoData API (ocean = 0 m).
                             Skipped for synthetic/offline elevation.
        2. OSM AREA MASK   - cells with >= 60% area covered by OSM water
                             polygons (lakes, reservoirs, bays) -> Water.
                             Coastline LineStrings excluded (unreliable for masking).
        3. PROXIMITY BOOST - graduated distance-based boost over 5.0 x cell_size radius.
                             boost(d) = 100 * max(0, 1 - d / boost_radius_m)
                             d=0  -> score=100 (HIGH); d=2 cells -> score=60 (MEDIUM);
                             d=5+ cells -> 0 boost (baseline preserved).
                             Adds 'water_proximity_score' column (max boost applied).
        4. SPATIAL CONTINUITY - after proximity boost, HIGH neighbours propagate a
                             bounded influence to adjacent land cells (at least MEDIUM).
        5. COASTAL FLAG    - land cells within 1.5 x cell_size of ocean -> tsunami flag.
        """
        from shapely.geometry import Point
        from shapely.ops import unary_union
        import geopandas as _gpd

        result = scored_grid.copy()
        result["is_coastal_tsunami_risk"] = False
        result["water_mask_reason"] = ""
        result["water_coverage_pct"] = 0.0
        # water_proximity_score: records the maximum proximity-boost score applied
        # to each cell (0.0 = no boost; 100.0 = directly adjacent to water).
        # Used for candidate scoring transparency.
        result["water_proximity_score"] = 0.0

        OCEAN_TYPES = {"coastline", "bay", "sea", "ocean"}
        AREA_WATER_TYPES = {"water", "reservoir", "basin", "bay", "sea", "ocean", "coastline"}
        LINEAR_TYPES = {"river", "canal", "stream", "drain", "ditch"}

        area_water_geoms = []
        ocean_area_geoms = []
        boost_geoms = []

        if water_bodies is not None and len(water_bodies) > 0:
            wb = water_bodies.copy()
            if wb.crs and str(wb.crs).upper() != "EPSG:4326":
                wb = wb.to_crs("EPSG:4326")
            for _, row in wb.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                geom = geom if geom.is_valid else geom.buffer(0)
                if geom.is_empty or not geom.is_valid:
                    continue
                wtype = str(row.get("water_type", "")).lower()
                boost_geoms.append(geom)
                if geom.geom_type in {"Polygon", "MultiPolygon"}:
                    if wtype in AREA_WATER_TYPES:
                        area_water_geoms.append(geom)
                        if wtype in OCEAN_TYPES:
                            ocean_area_geoms.append(geom)
                    elif wtype not in LINEAR_TYPES:
                        area_water_geoms.append(geom)

        # Ocean_Detector: point-in-polygon check against the static Natural Earth land mask.
        # Cells whose centroid is NOT inside the land polygon are classified as ocean → Water.
        # This replaces the deleted coastline-LineString polygonization path.
        try:
            land_geom = _load_land_mask()
            from shapely.geometry import box as shapely_box

            lon_min = float(result["centroid_lon"].min())
            lon_max = float(result["centroid_lon"].max())
            lat_min = float(result["centroid_lat"].min())
            lat_max = float(result["centroid_lat"].max())
            grid_bbox = shapely_box(lon_min, lat_min, lon_max, lat_max)

            try:
                land_clipped = land_geom.intersection(grid_bbox)
                ocean_in_bbox = grid_bbox.difference(land_clipped)
            except Exception as _bbox_exc:
                logger.warning("Ocean_Detector: bbox geometry failed: %s", _bbox_exc)
                land_clipped = None
                ocean_in_bbox = None

            already_water = result["risk_class"].values == "Water"
            ocean_mask = np.zeros(len(result), dtype=bool)

            # Vectorised ocean detection: build a GeoSeries of centroid Points
            # and use .within(land_geom) instead of a Python-level per-cell loop.
            # Semantics are identical to the original loop — cells whose centroid
            # is NOT within the land polygon are marked as ocean.
            try:
                centroid_pts = gpd.GeoSeries(
                    [Point(lon, lat)
                     for lon, lat in zip(
                         result["centroid_lon"].values,
                         result["centroid_lat"].values,
                     )],
                    crs="EPSG:4326",
                )
                # .within() returns a boolean Series; negate to get "not land"
                not_land = ~centroid_pts.within(land_geom)
                # Preserve original behaviour: cells already classified Water
                # are skipped (their existing classification is kept)
                ocean_mask = not_land.values & ~already_water
            except Exception as _vec_exc:
                logger.warning(
                    "Ocean_Detector vectorised path failed (%s) — "
                    "falling back to per-cell loop.",
                    _vec_exc,
                )
                # Fallback: original per-cell loop
                for i, (_, r) in enumerate(result.iterrows()):
                    if already_water[i]:
                        continue
                    try:
                        pt = Point(r["centroid_lon"], r["centroid_lat"])
                        if not land_geom.contains(pt):
                            ocean_mask[i] = True
                    except Exception as _cell_exc:
                        logger.warning(
                            "Ocean_Detector: cell %d geometry error (%s) — treating as land.",
                            i, _cell_exc,
                        )

            if ocean_mask.any():
                result.loc[ocean_mask, "risk_class"] = "Water"
                result.loc[ocean_mask, "risk_score"] = 0.0
                result.loc[ocean_mask, "water_mask_reason"] = "landmask"
                logger.info(
                    "Ocean_Detector (land mask): %d cells classified as Water.",
                    int(ocean_mask.sum()),
                )
                if ocean_in_bbox is not None and not ocean_in_bbox.is_empty:
                    ocean_area_geoms.append(ocean_in_bbox)
                    boost_geoms.append(ocean_in_bbox)
        except Exception as _det_exc:
            logger.warning("Ocean_Detector failed: %s — skipping ocean classification.", _det_exc)


        # Step 2: OSM polygon coverage mask (lakes, ponds, reservoirs)
        # Use 80% threshold for inland water bodies to avoid false masking
        # of cells where a small pond or canal clips the cell edge.
        # Use 60% threshold only for large confirmed water bodies (bay, sea, ocean).
        if area_water_geoms:
            try:
                water_union_4326 = unary_union(area_water_geoms)
                _wdf = _gpd.GeoDataFrame(geometry=[water_union_4326], crs="EPSG:4326")
                water_union_m = _wdf.to_crs("EPSG:3857").geometry.iloc[0]
                grid_m = _gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:3857")
                already_water = result["risk_class"].values == "Water"
                coverage_pct = np.zeros(len(result), dtype=float)
                coverage_water = np.zeros(len(result), dtype=bool)

                # Also build ocean-only union for lower threshold
                ocean_union_4326 = unary_union(ocean_area_geoms) if ocean_area_geoms else None
                ocean_union_m = None
                if ocean_union_4326 is not None:
                    _odf = _gpd.GeoDataFrame(geometry=[ocean_union_4326], crs="EPSG:4326")
                    ocean_union_m = _odf.to_crs("EPSG:3857").geometry.iloc[0]

                for i, cell_geom in enumerate(grid_m.geometry):
                    if already_water[i] or cell_geom is None or cell_geom.is_empty:
                        continue
                    try:
                        inter = cell_geom.intersection(water_union_m)
                        if inter.is_empty:
                            continue
                        pct = inter.area / cell_geom.area if cell_geom.area > 0 else 0
                        coverage_pct[i] = pct
                        # Lower threshold (60%) for ocean/bay/sea polygons
                        # Higher threshold (80%) for inland water bodies (ponds, tanks)
                        if ocean_union_m is not None:
                            ocean_inter = cell_geom.intersection(ocean_union_m)
                            ocean_pct = ocean_inter.area / cell_geom.area if cell_geom.area > 0 else 0
                            if ocean_pct >= 0.60 or (pct - ocean_pct) >= 0.80:
                                coverage_water[i] = True
                        else:
                            if pct >= 0.80:
                                coverage_water[i] = True
                    except Exception:
                        continue
                if coverage_water.any():
                    result.loc[coverage_water, "risk_class"] = "Water"
                    result.loc[coverage_water, "risk_score"] = 0.0
                    result.loc[coverage_water, "water_mask_reason"] = "coverage"
                    result["water_coverage_pct"] = (coverage_pct * 100).round(1)
                    logger.info("OSM coverage mask: %d cells -> Water.", int(coverage_water.sum()))
            except Exception as exc:
                logger.warning("OSM coverage mask failed: %s", exc)

        # Always compute centroid points in metric CRS — needed for both
        # proximity boost (when water bodies exist) AND spatial continuity
        # (which must run even when there are no water bodies, so that
        # pre-existing HIGH cells from the ML model propagate MEDIUM to
        # their immediate neighbours).
        try:
            centroid_pts_m = gpd.GeoSeries(
                [Point(r.centroid_lon, r.centroid_lat) for _, r in result.iterrows()],
                crs="EPSG:4326",
            ).to_crs("EPSG:3857")
        except Exception as _cpt_exc:
            logger.warning("Centroid computation failed: %s — skipping boost/continuity.", _cpt_exc)
            return result

        # Step 3: Proximity boost (only when water body geometries are available)
        if boost_geoms:
            try:
                boost_union_m = unary_union(
                    _gpd.GeoDataFrame(geometry=boost_geoms, crs="EPSG:4326")
                    .to_crs("EPSG:3857").geometry.tolist()
                )
                ocean_union_m = None
                if ocean_area_geoms:
                    ocean_union_m = unary_union(
                        _gpd.GeoDataFrame(geometry=ocean_area_geoms, crs="EPSG:4326")
                        .to_crs("EPSG:3857").geometry.tolist()
                    )

                # ── Graduated distance-based proximity boost ──────────────────
                # METHODOLOGY (declared, transparent):
                #
                # Root cause of GREEN coastal cells: the original boost_radius_m
                # (2.5 × cell_size = 1250m for 500m grid) was too narrow and
                # boost_max (medium_threshold + 10 = 76) too weak:
                #   - Only 2-3 cell rings received any influence
                #   - The gradient collapsed at 2 cell widths from water
                #   - Second-ring coastal cells received near-zero boost
                #
                # FIX: widen radius to 5.0 × cell_size and set boost_max = 100.0
                # so the gradient is:
                #   d = 0            → score = 100  → HIGH
                #   d = 1 × cell     → score = 80   → HIGH
                #   d = 2 × cell     → score = 60   → MEDIUM (≤ medium_threshold)
                #   d = 3 × cell     → score = 40   → MEDIUM
                #   d = 4 × cell     → score = 20   → LOW (< low_threshold)
                #   d ≥ 5 × cell     → score = 0    → no boost (baseline)
                #
                # Formula: boost(d) = boost_max × max(0, 1 − d / boost_radius_m)
                # Resulting score = max(original_score, boost(d))
                # risk_class is re-derived from the updated score.
                #
                # This preserves the existing scoring architecture — the WSI/RF model
                # output is the baseline; the boost is a post-scoring adjustment that
                # can only raise, never lower, a cell's risk.
                # ────────────────────────────────────────────────────────────────

                # boost_radius_m: 5 cell widths — provides a meaningful gradient
                # while remaining LOCAL (does not flood the entire map with influence).
                boost_radius_m = config.cell_size_meters * 5.0

                # boost_max = 100.0: cells immediately adjacent to water score 100
                # (HIGH). Cells at 2 cell widths from water score 60 (MEDIUM).
                # Cells at 4+ cell widths get minimal or zero boost, preserving
                # the baseline ML risk score.
                boost_max = 100.0

                now_water = result["risk_class"].values == "Water"

                for i, pt in enumerate(centroid_pts_m):
                    if now_water[i]:
                        continue
                    try:
                        dist_to_water = pt.distance(boost_union_m)
                        if dist_to_water >= boost_radius_m:
                            continue   # beyond influence radius — no boost
                        # Linear decay: 1.0 at dist=0, 0.0 at dist=boost_radius_m
                        strength = max(0.0, 1.0 - dist_to_water / boost_radius_m)
                        boosted_score = boost_max * strength
                        idx = result.index[i]
                        current_score = float(result.at[idx, "risk_score"])
                        new_score = max(current_score, boosted_score)
                        # Record the water-proximity boost for transparency/relocation
                        result.at[idx, "water_proximity_score"] = round(boosted_score, 2)
                        if new_score > current_score:
                            result.at[idx, "risk_score"] = round(new_score, 2)
                            if new_score > config.medium_threshold:
                                result.at[idx, "risk_class"] = "High"
                            elif new_score > config.low_threshold:
                                result.at[idx, "risk_class"] = "Medium"
                            # Note: new_score <= low_threshold means no class change
                    except Exception:
                        pass

                # Step 5: Coastal flag
                if ocean_union_m is not None:
                    coastal_m = config.cell_size_meters * 1.5
                    now_water2 = result["risk_class"].values == "Water"
                    for i, pt in enumerate(centroid_pts_m):
                        if not now_water2[i]:
                            try:
                                if pt.distance(ocean_union_m) <= coastal_m:
                                    result.iloc[i, result.columns.get_loc("is_coastal_tsunami_risk")] = True
                            except Exception:
                                pass

                logger.info("Water mask done: %d Water, %d coastal.",
                            int((result["risk_class"] == "Water").sum()),
                            int(result["is_coastal_tsunami_risk"].sum()))
            except Exception as exc:
                logger.warning("Proximity/coastal step failed: %s", exc)

        # Step 4: Spatial continuity propagation (ALWAYS runs — independent of water bodies)
        # ── Spatial continuity ─────────────────────────────────────────────────
        # Problem: a lone HIGH cell (either from the ML model or proximity boost)
        # may have GREEN immediate neighbours if those neighbours are just beyond
        # the proximity boost radius, or if the HIGH cell came from the base model
        # without any water body geometries being available.
        #
        # Requirement: a HIGH cell's immediate grid neighbours receive at least
        # MEDIUM influence (unless already HIGH or Water). Bounded: 1-ring only.
        # ──────────────────────────────────────────────────────────────────────
        try:
            neighbour_radius_m = config.cell_size_meters * 1.5
            # Minimum score for a neighbour of a HIGH cell: mid-point of MEDIUM band
            neighbour_min_score = (
                config.low_threshold
                + (config.medium_threshold - config.low_threshold) * 0.5
            )
            now_high = result["risk_class"].values == "High"
            now_water_sc = result["risk_class"].values == "Water"

            if now_high.any():
                # Build a union geometry of all HIGH cell centroids buffered by
                # neighbour_radius_m.  A single .within() check on each non-HIGH
                # cell replaces an O(N×M) loop with O(N log M).
                from shapely.ops import unary_union as _uu_sc
                high_union_m = _uu_sc([
                    centroid_pts_m.iloc[i].buffer(neighbour_radius_m)
                    for i in range(len(centroid_pts_m))
                    if now_high[i]
                ])
                for i, pt in enumerate(centroid_pts_m):
                    if now_high[i] or now_water_sc[i]:
                        continue
                    try:
                        if pt.within(high_union_m):
                            idx = result.index[i]
                            current = float(result.at[idx, "risk_score"])
                            if current < neighbour_min_score:
                                result.at[idx, "risk_score"] = round(
                                    neighbour_min_score, 2
                                )
                                result.at[idx, "risk_class"] = "Medium"
                    except Exception:
                        pass
            logger.info(
                "Spatial continuity: %d HIGH cells → neighbours boosted to ≥MEDIUM.",
                int(now_high.sum()),
            )
        except Exception as _sc_exc:
            logger.warning("Spatial continuity step failed: %s", _sc_exc)

        return result

    def run_stage(self, stage_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a single named pipeline stage."""
        stages = {
            "grid": generate_grid,
            "elevation": generate_synthetic_elevation,
            "rainfall": generate_synthetic_rainfall,
        }
        if stage_name not in stages:
            raise ValueError(f"Unknown stage: {stage_name}. Available: {list(stages)}")
        return stages[stage_name](*args, **kwargs)

