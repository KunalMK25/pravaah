"""
Integration tests: five reference-region verification for the static land/sea mask.

Each parametrised case calls ``_apply_water_mask_and_proximity_boost`` directly with
the real ``ne_50m_land.geojson`` and an empty water-bodies GeoDataFrame.

Skip the entire module when ``data/landmask/ne_50m_land.geojson`` is absent.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""
from __future__ import annotations

import pathlib
import logging

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

import flood_risk_zonation.pipeline as _pipeline_mod
from flood_risk_zonation.config import PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline

# ── Path to the real land mask ────────────────────────────────────────────────

_LANDMASK_PATH = (
    pathlib.Path(__file__).parent.parent.parent
    / "data" / "landmask" / "ne_50m_land.geojson"
)

# Skip every test in this module when the file is absent.
pytestmark = pytest.mark.skipif(
    not _LANDMASK_PATH.exists(),
    reason=f"Real land mask file not found: {_LANDMASK_PATH}",
)

# ── Reset the module-level cache once for the whole module ────────────────────
# The file is loaded on the first test call and re-used for all subsequent ones
# (standard module-level cache behaviour).  We only reset once here so the file
# is read exactly once across all five parametrised cases.

@pytest.fixture(scope="module", autouse=True)
def _reset_land_mask_cache():
    """Reset _LAND_MASK_GEOM once before the module runs, load real file, then restore."""
    original = _pipeline_mod._LAND_MASK_GEOM
    _pipeline_mod._LAND_MASK_GEOM = None
    yield
    # Restore original state (None normally; allows test isolation at module level)
    _pipeline_mod._LAND_MASK_GEOM = original


# ── Helper functions (mirrors tests/unit/test_water_masking.py helpers) ───────

def _make_grid(
    lons: list[float],
    lats: list[float],
    elevations: list[float],
    cell_size_deg: float = 0.001,
) -> gpd.GeoDataFrame:
    """Build a minimal scored grid suitable for masking tests."""
    n = len(lons)
    assert len(lats) == n and len(elevations) == n
    geoms = [
        box(lon, lat, lon + cell_size_deg, lat + cell_size_deg)
        for lon, lat in zip(lons, lats)
    ]
    return gpd.GeoDataFrame(
        {
            "cell_id": [str(i) for i in range(n)],
            "centroid_lon": [lon + cell_size_deg / 2 for lon in lons],
            "centroid_lat": [lat + cell_size_deg / 2 for lat in lats],
            "elevation_m": np.array(elevations, dtype=np.float32),
            "risk_score": np.full(n, 50.0, dtype=np.float32),
            "risk_class": ["Medium"] * n,
            "slope_deg": np.zeros(n, dtype=np.float32),
            "twi": np.zeros(n, dtype=np.float32),
            "rainfall_mean_mm": np.zeros(n, dtype=np.float32),
            "rainfall_max_24h_mm": np.zeros(n, dtype=np.float32),
            "dist_water_m": np.full(n, 5000.0, dtype=np.float32),
            "drainage_capacity": np.full(n, 0.5, dtype=np.float32),
            "population_density": np.zeros(n, dtype=np.float32),
            "aspect_deg": np.zeros(n, dtype=np.float32),
            "curvature": np.zeros(n, dtype=np.float32),
        },
        geometry=geoms,
        crs="EPSG:4326",
    )


def _water_gdf_empty() -> gpd.GeoDataFrame:
    """Return an empty water-bodies GeoDataFrame (no OSM data)."""
    return gpd.GeoDataFrame(
        {"geometry": [], "water_type": [], "name": []},
        crs="EPSG:4326",
    )


def _config(cell_size: float = 100.0) -> PipelineConfig:
    return PipelineConfig(
        cell_size_meters=cell_size,
        use_cache=False,
        allow_network=False,
    )


def _make_bbox_grid(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    cell_size_deg: float = 0.01,
) -> gpd.GeoDataFrame:
    """
    Build a grid of cells that tiles ``[min_lon, max_lon] × [min_lat, max_lat]``
    with ``cell_size_deg``-sized square cells.
    """
    import numpy as _np

    lon_edges = _np.arange(min_lon, max_lon, cell_size_deg)
    lat_edges = _np.arange(min_lat, max_lat, cell_size_deg)

    lons: list[float] = []
    lats: list[float] = []

    for lo in lon_edges:
        for la in lat_edges:
            lons.append(float(lo))
            lats.append(float(la))

    n = len(lons)
    elevations = [50.0] * n  # neutral elevation — won't trigger elevation mask
    return _make_grid(lons, lats, elevations, cell_size_deg=cell_size_deg)


# ── Parametrised test cases ───────────────────────────────────────────────────

_REGIONS = [
    pytest.param(
        # 1. Gottigere — fully inland Bangalore suburb, zero ocean cells expected
        "gottigere",
        77.58, 12.83, 77.62, 12.87,
        "zero_landmask",       # expect 0 cells with water_mask_reason == "landmask"
        id="gottigere_inland",
    ),
    pytest.param(
        # 2. Chennai Marina — coastal; expect ≥1 ocean AND ≥1 non-Water
        "chennai_marina",
        80.27, 13.03, 80.30, 13.07,
        "coastal_mix",
        id="chennai_marina_coastal",
    ),
    pytest.param(
        # 3. Dal Lake — inland lake in Srinagar; land mask must not fire
        "dal_lake",
        74.83, 34.07, 74.89, 34.13,
        "zero_landmask",
        id="dal_lake_inland",
    ),
    pytest.param(
        # 4. Puri — coastal Odisha; expect ≥1 ocean AND ≥1 non-Water
        "puri",
        85.81, 19.78, 85.85, 19.82,
        "coastal_mix",
        id="puri_coastal",
    ),
    pytest.param(
        # 5. Arabian Sea — pure open ocean; expect 100 % landmask
        "arabian_sea",
        65.00, 18.00, 65.20, 18.20,
        "all_landmask",
        id="arabian_sea_ocean",
    ),
]


@pytest.mark.parametrize(
    "region_name, min_lon, min_lat, max_lon, max_lat, expectation",
    _REGIONS,
)
def test_reference_region(
    region_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    expectation: str,
    caplog,
):
    """
    Verify land/sea classification for a known reference bounding box.

    Calls ``_apply_water_mask_and_proximity_boost`` directly with the real
    ``ne_50m_land.geojson`` and an empty water-bodies GDF.

    Expectation codes
    -----------------
    ``zero_landmask``  — 0 cells must have ``water_mask_reason == "landmask"``
    ``coastal_mix``    — ≥1 landmask cell AND ≥1 non-Water cell
    ``all_landmask``   — 100 % of cells must have ``water_mask_reason == "landmask"``

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    grid = _make_bbox_grid(min_lon, min_lat, max_lon, max_lat, cell_size_deg=0.01)
    empty_wb = _water_gdf_empty()
    cfg = _config()
    pipeline = FloodRiskPipeline(cfg)

    with caplog.at_level(logging.INFO, logger="flood_risk_zonation.pipeline"):
        result = pipeline._apply_water_mask_and_proximity_boost(
            grid, empty_wb, cfg, elevation_source="synthetic"
        )

    total_cells = len(result)
    landmask_cells = int((result["water_mask_reason"] == "landmask").sum())
    non_water_cells = int((result["risk_class"] != "Water").sum())
    water_pct = (landmask_cells / total_cells * 100.0) if total_cells > 0 else 0.0

    # Always print the water-cell percentage so it is visible with ``pytest -s``
    print(
        f"\n[{region_name}] total={total_cells}, "
        f"landmask={landmask_cells} ({water_pct:.1f}%), "
        f"non-Water={non_water_cells}"
    )

    if expectation == "zero_landmask":
        # Requirements 5.1, 5.3 — fully inland; Ocean_Detector must produce 0 ocean cells
        assert landmask_cells == 0, (
            f"[{region_name}] Expected 0 landmask cells but got {landmask_cells} "
            f"({water_pct:.1f}%). First offending rows:\n"
            f"{result[result['water_mask_reason'] == 'landmask'][['centroid_lon', 'centroid_lat', 'risk_class', 'water_mask_reason']].head()}"
        )

    elif expectation == "coastal_mix":
        # Requirements 5.2, 5.4 — coastal; both ocean AND land cells expected
        assert landmask_cells >= 1, (
            f"[{region_name}] Expected ≥1 landmask cell but got {landmask_cells}. "
            "The coastline should produce at least one ocean cell."
        )
        assert non_water_cells >= 1, (
            f"[{region_name}] Expected ≥1 non-Water cell but got {non_water_cells}. "
            "The bbox should contain at least one land cell."
        )

    elif expectation == "all_landmask":
        # Requirement 5.5 — pure open ocean; every cell must be ocean
        assert landmask_cells == total_cells, (
            f"[{region_name}] Expected 100% landmask cells ({total_cells}) "
            f"but only {landmask_cells} ({water_pct:.1f}%) were classified as ocean."
        )

    else:
        raise ValueError(f"Unknown expectation code: {expectation!r}")
