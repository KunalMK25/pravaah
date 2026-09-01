"""
PRAVAAH-AI — Habitation-level exposure analysis.

HAZARD vs EXPOSURE:
  Hazard   = "how dangerous is this location?" (grid cell scores from Phase 1)
  Exposure = "who / what is located in a hazardous area?" (habitations)

This module performs a spatial overlay between habitation point locations and
the scored hazard grid, then assigns each habitation an exposure class.

Population handling:
  - Multi-source provider chain (Authoritative → Regional → WorldPop → OSM → Derived → UNKNOWN)
  - Comprehensive provenance, confidence, and status tracking
  - Never fabricates population data

Population is never fabricated.
"""
from __future__ import annotations

import logging
from math import cos, radians, sqrt
from typing import Optional

import geopandas as gpd
import numpy as np

from flood_risk_zonation.models import ExposureResult, Habitation, HabitationDataset
from flood_risk_zonation.population.chain import PopulationProviderChain

logger = logging.getLogger(__name__)

# ── Search radius for "neighbouring cells" ─────────────────────────────────────
# A habitation point may sit exactly on a cell boundary.  We expand the search
# to include the N nearest cells (by centroid distance) rather than strictly
# point-in-polygon, so small habitations are never missed.
_NEIGHBOUR_CELLS = 4   # use up to 4 nearest cells when computing exposure


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate haversine distance in km between two WGS84 points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        (dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * (dlon / 2) ** 2
    )
    return R * 2 * sqrt(max(a, 0))


def _dominant_class(classes: list[str]) -> str:
    """Return the most severe risk class present in a list."""
    priority = {"High": 3, "Medium": 2, "Low": 1, "Water": 0}
    if not classes:
        return "Low"
    return max(classes, key=lambda c: priority.get(c, 0))


def _classify_hazard(hazard_score: float, pct_high: float, low_t: float, med_t: float) -> str:
    """Map composite hazard score and high-risk fraction to a class string."""
    if pct_high >= 0.5:
        return "High"
    if hazard_score > med_t:
        return "High"
    if hazard_score > low_t:
        return "Medium"
    return "Low"


def analyse_exposure(
    habitation_dataset: HabitationDataset,
    scored_grid: gpd.GeoDataFrame,
    population_chain: Optional[PopulationProviderChain] = None,
    bbox: Optional[tuple] = None,
    low_threshold: float = 33.0,
    medium_threshold: float = 66.0,
) -> list[ExposureResult]:
    """
    Overlay habitation points against the scored hazard grid and compute
    per-habitation exposure metrics.

    Algorithm:
      1. For each habitation, find the grid cell whose centroid is nearest
         (within _NEIGHBOUR_CELLS closest matches).
      2. Compute mean hazard_score and pct_high_risk across those cells.
      3. Determine hazard_class from combined score + cell class majority.
      4. Get population from provider chain (if provided) or OSM tag (fallback).
      5. Flag is_in_red_zone = (hazard_class == "High").

    Parameters
    ----------
    habitation_dataset : HabitationDataset
        All habitation entities for the study area.
    scored_grid : gpd.GeoDataFrame
        Phase 1 pipeline output — must have risk_score, risk_class,
        centroid_lat, centroid_lon, cell_id columns.
    population_chain : Optional[PopulationProviderChain]
        Multi-source population provider chain. If None, uses OSM tags only (legacy).
    bbox : Optional[tuple]
        Bounding box (min_lon, min_lat, max_lon, max_lat) for population aggregation.
        Required if population_chain provided.
    low_threshold : float
        Risk score boundary between Low and Medium (default 33).
    medium_threshold : float
        Risk score boundary between Medium and High (default 66).

    Returns
    -------
    list[ExposureResult]
        One ExposureResult per habitation in habitation_dataset.habitations.
    """
    if not habitation_dataset.habitations:
        return []

    # Precompute grid centroid arrays for vectorised nearest-cell lookup
    if "centroid_lat" not in scored_grid.columns or "centroid_lon" not in scored_grid.columns:
        # Derive from geometry centroids
        centroids = scored_grid.geometry.centroid
        grid_lats = centroids.y.values
        grid_lons = centroids.x.values
    else:
        grid_lats = scored_grid["centroid_lat"].values.astype(float)
        grid_lons = scored_grid["centroid_lon"].values.astype(float)

    grid_scores = scored_grid["risk_score"].values.astype(float)
    grid_classes = scored_grid["risk_class"].values
    grid_cell_ids = (
        scored_grid["cell_id"].values if "cell_id" in scored_grid.columns
        else np.array([str(i) for i in range(len(scored_grid))])
    )

    # Ensure bbox for population chain
    if population_chain and bbox is None:
        bbox = (
            scored_grid["centroid_lon"].min(),
            scored_grid["centroid_lat"].min(),
            scored_grid["centroid_lon"].max(),
            scored_grid["centroid_lat"].max(),
        )

    results: list[ExposureResult] = []

    for hab in habitation_dataset.habitations:
        try:
            # Distance (degrees proxy — good enough for nearest-cell lookup)
            dlat = grid_lats - hab.lat
            dlon = (grid_lons - hab.lon) * cos(radians(hab.lat))
            dist_sq = dlat ** 2 + dlon ** 2
            n_cells = min(_NEIGHBOUR_CELLS, len(dist_sq))
            nearest_idx = np.argpartition(dist_sq, n_cells - 1)[:n_cells]

            nearby_scores = grid_scores[nearest_idx]
            nearby_classes = [str(grid_classes[i]) for i in nearest_idx]
            nearby_ids = [str(grid_cell_ids[i]) for i in nearest_idx]

            # Exclude pure-water cells from hazard score computation
            non_water = [s for s, c in zip(nearby_scores, nearby_classes) if c != "Water"]
            hazard_score = float(np.mean(non_water)) if non_water else 0.0
            pct_high = sum(1 for c in nearby_classes if c == "High") / len(nearby_classes)
            hazard_class = _dominant_class(
                [c for c in nearby_classes if c != "Water"] or nearby_classes
            )

            # Get population from provider chain or OSM tag (fallback)
            if population_chain and bbox:
                pop_result = population_chain.get_population(
                    hab_id=hab.hab_id,
                    lat=hab.lat,
                    lon=hab.lon,
                    bbox=bbox,
                )
                pop_exposed = pop_result.population
                pop_source = pop_result.provider.value
                pop_confidence = pop_result.confidence
                pop_status = pop_result.status.value
                pop_method = pop_result.method.value if pop_result.method else None
                pop_metadata = {
                    "population": pop_result.population,
                    "source": pop_result.source,
                    "provider": pop_result.provider.value,
                    "method": pop_result.method.value if pop_result.method else None,
                    "status": pop_result.status.value,
                    "confidence": pop_result.confidence,
                }
            else:
                # Legacy: OSM-only population handling
                if hab.population is not None and hab.population > 0:
                    pop_exposed = hab.population
                    pop_source = "osm_tag"
                    pop_confidence = 0.60  # OSM baseline confidence
                    pop_status = "OBSERVED"
                    pop_method = "osm_tag_direct"
                    pop_metadata = {
                        "population": pop_exposed,
                        "source": "osm_tag",
                        "provider": "OSM",
                        "method": "osm_tag_direct",
                        "status": "OBSERVED",
                        "confidence": pop_confidence,
                    }
                else:
                    pop_exposed = None
                    pop_source = "UNKNOWN"
                    pop_confidence = 0.0
                    pop_status = "UNKNOWN"
                    pop_method = None
                    pop_metadata = {
                        "population": None,
                        "source": "unknown",
                        "provider": "UNKNOWN",
                        "method": None,
                        "status": "UNKNOWN",
                        "confidence": 0.0,
                    }

            results.append(
                ExposureResult(
                    hab_id=hab.hab_id,
                    name=hab.name or f"Unnamed ({hab.hab_type})",
                    hab_type=hab.hab_type,
                    lat=hab.lat,
                    lon=hab.lon,
                    hazard_score=round(hazard_score, 2),
                    hazard_class=hazard_class,
                    pct_high_risk=round(pct_high, 3),
                    population_source=pop_source,
                    population_exposed=pop_exposed,
                    population_confidence=pop_confidence,
                    population_status=pop_status,
                    population_method=pop_method,
                    population_metadata=pop_metadata,
                    is_in_red_zone=(hazard_class == "High"),
                    intersecting_cell_ids=nearby_ids,
                )
            )
        except Exception as exc:
            logger.warning("Exposure analysis failed for %s: %s", hab.hab_id, exc)
            continue

    # Filter out water-cell habitations: they cannot have land-based analysis.
    # A habitation with hazard_class="Water" means all nearby grid cells are water.
    # Such habitations must not participate in relocation scoring or UI reporting.
    land_habitations = [r for r in results if r.hazard_class != "Water"]
    
    n_water = len(results) - len(land_habitations)
    if n_water > 0:
        logger.info(
            "Exposure analysis: %d habitations total, %d land-based, %d water-cell (filtered).",
            len(results),
            len(land_habitations),
            n_water,
        )
    else:
        logger.info(
            "Exposure analysis: %d habitations, %d in red zone.",
            len(land_habitations),
            sum(1 for r in land_habitations if r.is_in_red_zone),
        )
    return land_habitations
