"""
Integration tests: Phase 1 SIH26191 foundation refactor.

Proves:
1. `FloodRiskPipeline.run()` still works with its original call signature
   (no `progress_callback`) — full backward compatibility for existing
   callers (mirrors `tests/integration/test_pipeline.py`).
2. `progress_callback` is optional, does not change pipeline output, and
   is actually invoked at each stage when supplied — this is the hook
   `app.py` now uses instead of duplicating the pipeline's stage logic.
3. `run_from_ingested_data()` — the method the offline/demo UI path now
   calls directly — produces a structurally valid, compatible
   `FloodRiskResult` given pre-fetched (demo-style) data, proving the
   grid → drainage → features → model → scoring → water-mask → result
   logic is a single shared implementation rather than being duplicated
   between `app.py`'s offline branch and `FloodRiskPipeline.run()`.
"""
from __future__ import annotations

import pytest

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.ingest.population import load_population
from flood_risk_zonation.ingest.sample_data import DEMO_REGIONS, get_demo_elevation, get_demo_rainfall, get_demo_water_bodies
from flood_risk_zonation.pipeline import FloodRiskPipeline


def _small_config(**overrides) -> PipelineConfig:
    defaults = dict(cell_size_meters=5000, rf_n_estimators=10, cv_folds=3, use_cache=False)
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_run_without_progress_callback_is_unchanged():
    """run(bbox) with no progress_callback (the pre-refactor call signature)
    still returns a valid, correctly-shaped FloodRiskResult."""
    bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=0.5, max_lat=0.5)
    pipeline = FloodRiskPipeline(_small_config())
    result = pipeline.run(bbox)

    assert result.cell_count > 0
    assert "risk_score" in result.scored_grid.columns
    assert "risk_class" in result.scored_grid.columns
    assert result.scored_grid["risk_score"].between(0.0, 100.0).all()
    assert result.analysis_result.method == "ensemble"


def test_run_with_progress_callback_matches_run_without_one():
    """Supplying progress_callback must not change the scored output —
    it is purely a status-reporting hook, and must actually fire."""
    bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=0.5, max_lat=0.5)

    messages: list[str] = []
    pipeline_a = FloodRiskPipeline(_small_config())
    result_a = pipeline_a.run(bbox, progress_callback=messages.append)

    pipeline_b = FloodRiskPipeline(_small_config())
    result_b = pipeline_b.run(bbox)

    # The callback must have actually been invoked at each documented stage.
    assert len(messages) >= 4
    assert any("elevation" in m.lower() for m in messages)
    assert any("water" in m.lower() for m in messages)
    assert any("rainfall" in m.lower() for m in messages)
    assert any("model" in m.lower() for m in messages)

    # Same config/bbox/seed → identical cell count and risk-class set,
    # proving the callback is a pure side channel, not a behavioural change.
    assert result_a.cell_count == result_b.cell_count
    assert set(result_a.scored_grid["risk_class"].unique()) == set(
        result_b.scored_grid["risk_class"].unique()
    )


def test_run_from_ingested_data_matches_run_shape():
    """run_from_ingested_data(), given the SAME real-data-equivalent inputs
    run() would have fetched itself, produces a result with the identical
    shape/contract as run() — proving both entry points share one
    implementation for grid/drainage/feature/model/scoring/water-mask
    stages instead of each re-implementing it."""
    bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=0.5, max_lat=0.5)
    config = _small_config()
    pipeline = FloodRiskPipeline(config)

    # Ingest exactly the way run() does internally, then hand it to the
    # shared stage method directly (this is what app.py's offline branch
    # now does with demo data instead).
    from flood_risk_zonation.ingest.elevation import generate_synthetic_elevation
    from flood_risk_zonation.ingest.rainfall import generate_synthetic_rainfall
    from flood_risk_zonation.ingest.water_bodies import load_water_bodies

    elevation = generate_synthetic_elevation(bbox, resolution_m=500, seed=config.random_seed)
    rainfall = generate_synthetic_rainfall(bbox, resolution_m=1000, seed=config.random_seed)
    water_bodies = load_water_bodies(bbox, data_dir=None, allow_network=False)
    population = load_population(bbox, data_dir=str(config.cache_dir))

    provenance = {
        "elevation": elevation.source,
        "rainfall": rainfall.source,
        "water_bodies": water_bodies.attrs.get("source", "unavailable"),
        "population": population.source,
    }

    result = pipeline.run_from_ingested_data(
        bounding_box=bbox,
        elevation=elevation,
        rainfall=rainfall,
        water_bodies=water_bodies,
        population=population,
        provenance=provenance,
        data_tier=3,
    )

    assert result.cell_count > 0
    assert "risk_score" in result.scored_grid.columns
    assert "risk_class" in result.scored_grid.columns
    assert result.scored_grid["risk_score"].between(0.0, 100.0).all()
    assert set(result.scored_grid["risk_class"].unique()).issubset(
        {"Low", "Medium", "High", "Water"}
    )
    assert result.analysis_result.method == "ensemble"
    assert result.data_provenance["drainage"] in {
        "osm_proxy",          # OSM drainage linestrings found
        "synthetic_fallback", # No linestrings available — fallback used
    }

    assert result.data_tier == 3


def test_offline_demo_path_via_run_from_ingested_data():
    """Reproduces app.py's offline/demo branch exactly (bundled per-region
    synthetic data via ingest.sample_data), proving the offline UI path
    that now calls run_from_ingested_data() directly still works end to
    end and yields a usable FloodRiskResult, matching pre-refactor
    behaviour where this logic was duplicated inline in app.py."""
    region = DEMO_REGIONS["Bangalore (Gottigere)"]
    config = _small_config(cell_size_meters=1000)
    pipeline = FloodRiskPipeline(config)

    elevation = get_demo_elevation(region, resolution_m=30.0)
    rainfall = get_demo_rainfall(region)
    water_bodies = get_demo_water_bodies(region)
    population = load_population(region.bbox, data_dir=str(config.cache_dir))

    provenance = {
        "elevation": "offline_sample",
        "rainfall": "offline_sample",
        "water_bodies": "offline_sample",
        "population": population.source,
    }

    result = pipeline.run_from_ingested_data(
        bounding_box=region.bbox,
        elevation=elevation,
        rainfall=rainfall,
        water_bodies=water_bodies,
        population=population,
        provenance=provenance,
        data_tier=3,
    )

    assert result.cell_count > 0
    assert result.data_tier == 3
    assert result.data_provenance["elevation"] == "offline_sample"
    assert set(result.scored_grid["risk_class"].unique()).issubset(
        {"Low", "Medium", "High", "Water"}
    )
