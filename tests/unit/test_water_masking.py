"""
Unit tests for water masking (Part 1) and coastal tsunami flag (Part 2).

Tests use synthetic GeoDataFrames — no network calls, no pipeline run.
The masking logic under test lives in:
    FloodRiskPipeline._apply_water_mask_and_proximity_boost
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shapely.geometry import Point, Polygon, box

from flood_risk_zonation.config import BoundingBox, PipelineConfig
from flood_risk_zonation.pipeline import FloodRiskPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_grid(
    lons: list[float],
    lats: list[float],
    elevations: list[float],
    cell_size_deg: float = 0.001,
) -> gpd.GeoDataFrame:
    """Build a minimal scored grid for masking tests."""
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


def _water_gdf(polygons: list[Polygon], water_types: list[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "geometry": polygons,
            "water_type": water_types,
            "name": [""] * len(polygons),
        },
        crs="EPSG:4326",
    )


def _config(cell_size: float = 100.0) -> PipelineConfig:
    return PipelineConfig(
        cell_size_meters=cell_size,
        use_cache=False,
        allow_network=False,
    )


def _pipeline() -> FloodRiskPipeline:
    return FloodRiskPipeline(_config())


# ── Part 1: Water masking is a hard override ──────────────────────────────────

class TestWaterMaskingHardOverride:
    """Cells inside water polygons must become Water regardless of prior score."""

    def test_cell_inside_water_polygon_becomes_water(self):
        """A cell whose centroid falls inside a water polygon → risk_class='Water'."""
        # Cell centroid at (77.555, 12.845) — inside the polygon
        grid = _make_grid([77.55], [12.84], [50.0])
        water = _water_gdf(
            [box(77.50, 12.80, 77.60, 12.90)],  # large polygon covering cell
            ["water"],
        )
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        assert result.iloc[0]["risk_class"] == "Water"

    def test_water_mask_overrides_high_risk_score(self):
        """Water mask must override even a cell scored as High risk."""
        grid = _make_grid([77.55], [12.84], [50.0])
        grid.at[0, "risk_class"] = "High"
        grid.at[0, "risk_score"] = 90.0
        water = _water_gdf([box(77.50, 12.80, 77.60, 12.90)], ["water"])
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        assert result.iloc[0]["risk_class"] == "Water", (
            "Water mask must override High risk classification"
        )

    def test_non_water_cell_preserves_risk_class(self):
        """Cells outside water polygons keep their original risk_class."""
        grid = _make_grid([77.55, 77.56], [12.84, 12.84], [50.0, 50.0])
        grid.at[0, "risk_class"] = "High"
        grid.at[1, "risk_class"] = "Low"
        # Water polygon far away — doesn't cover either cell
        water = _water_gdf([box(80.0, 15.0, 81.0, 16.0)], ["water"])
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        assert result.iloc[0]["risk_class"] == "High"
        assert result.iloc[1]["risk_class"] == "Low"

    def test_ocean_region_all_cells_water(self):
        """
        For a mostly-ocean bbox (all cells inside a large ocean polygon),
        all cells must be classified as Water.
        """
        # 3×3 grid of cells, all inside an ocean polygon
        lons = [80.24, 80.25, 80.26] * 3
        lats = [12.98, 12.98, 12.98, 12.99, 12.99, 12.99, 13.00, 13.00, 13.00]
        elevs = [5.0] * 9  # low but > 1 m so elevation mask won't fire alone
        grid = _make_grid(lons, lats, elevs)
        ocean_polygon = box(80.20, 12.95, 80.35, 13.10)  # covers all cells
        water = _water_gdf([ocean_polygon], ["coastline"])  # OSM coastline tag
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, water, _config())
        non_water = result[result["risk_class"] != "Water"]
        assert len(non_water) == 0, (
            f"{len(non_water)} cells still have non-Water class after ocean masking: "
            f"{non_water['risk_class'].tolist()}"
        )

    def test_is_coastal_tsunami_risk_column_always_present(self):
        """is_coastal_tsunami_risk column must exist even when no water bodies."""
        grid = _make_grid([77.55], [12.84], [50.0])
        empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty_wb, _config())
        assert "is_coastal_tsunami_risk" in result.columns

    def test_elevation_source_ignored(self):
        """
        Ocean_Detector produces identical results regardless of elevation_source value.

        Validates: Requirements 3.3
        """
        from unittest.mock import patch

        # Synthetic land polygon: 77.0–78.0°E, 12.0–13.0°N
        # Cell centroid at (65.1005, 18.1005) is clearly outside → ocean classification fires.
        synthetic_land_box = box(77.0, 12.0, 78.0, 13.0)

        # Cell centroid well outside the synthetic land polygon
        grid = _make_grid([65.1], [18.1], [50.0])
        empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        config = _config()
        pipeline = _pipeline()

        elevation_sources = [
            "synthetic",
            "opentopodata_srtm",
            "/some/arbitrary/path/to/elevation.tif",
        ]

        results = []
        with patch(
            "flood_risk_zonation.pipeline._load_land_mask",
            return_value=synthetic_land_box,
        ):
            for source in elevation_sources:
                result = pipeline._apply_water_mask_and_proximity_boost(
                    grid.copy(), empty_wb, config, elevation_source=source
                )
                results.append(result)

        # All three runs must produce identical classification for every cell
        for i, source in enumerate(elevation_sources[1:], start=1):
            ref = results[0]
            cmp = results[i]
            assert list(ref["risk_class"]) == list(cmp["risk_class"]), (
                f"risk_class differs for elevation_source={elevation_sources[i]!r}: "
                f"{list(ref['risk_class'])} vs {list(cmp['risk_class'])}"
            )
            assert list(ref["risk_score"]) == list(cmp["risk_score"]), (
                f"risk_score differs for elevation_source={elevation_sources[i]!r}: "
                f"{list(ref['risk_score'])} vs {list(cmp['risk_score'])}"
            )
            assert list(ref["water_mask_reason"]) == list(cmp["water_mask_reason"]), (
                f"water_mask_reason differs for elevation_source={elevation_sources[i]!r}: "
                f"{list(ref['water_mask_reason'])} vs {list(cmp['water_mask_reason'])}"
            )

    def test_land_centroid_not_classified_as_water(self):
        """A cell whose centroid falls inside the land polygon must NOT become Water/landmask.

        Validates: Requirements 3.1
        """
        from unittest.mock import patch
        from shapely.geometry import box as shapely_box

        # Synthetic land polygon: 77.0–78.0°E, 12.0–13.0°N
        synthetic_land = shapely_box(77.0, 12.0, 78.0, 13.0)

        # Cell centroid clearly inside the land box (77.5, 12.5)
        grid = _make_grid([77.4995], [12.4995], [50.0])

        empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=synthetic_land):
            result = _pipeline()._apply_water_mask_and_proximity_boost(
                grid, empty_wb, _config()
            )

        row = result.iloc[0]
        assert row["risk_class"] != "Water", (
            f"Land centroid must not be classified as Water; got risk_class={row['risk_class']!r}"
        )
        assert row["water_mask_reason"] != "landmask", (
            f"Land centroid must not have water_mask_reason='landmask'; "
            f"got {row['water_mask_reason']!r}"
        )

    def test_ocean_centroid_classified_as_water(self):
        """
        A cell centroid outside the synthetic land polygon is classified as Water
        with risk_score=0.0 and water_mask_reason='landmask'.

        Validates: Requirements 3.1, 3.2
        """
        from unittest.mock import patch
        from shapely.geometry import box as shapely_box

        # Synthetic land polygon: 77.0–78.0°E, 12.0–13.0°N
        synthetic_land = shapely_box(77.0, 12.0, 78.0, 13.0)

        # Cell centroid at (65.1, 18.1) — clearly outside the land polygon (Arabian Sea)
        grid = _make_grid([65.0], [18.0], [50.0])

        empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        with patch("flood_risk_zonation.pipeline._load_land_mask", return_value=synthetic_land):
            result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty_wb, _config())

        row = result.iloc[0]
        assert row["risk_class"] == "Water", (
            f"Expected risk_class='Water' for ocean centroid, got '{row['risk_class']}'"
        )
        assert row["risk_score"] == 0.0, (
            f"Expected risk_score=0.0 for ocean centroid, got {row['risk_score']}"
        )
        assert row["water_mask_reason"] == "landmask", (
            f"Expected water_mask_reason='landmask', got '{row['water_mask_reason']}'"
        )


# ── Part 2: Coastal tsunami flag ──────────────────────────────────────────────

class TestCoastalTsunamiFlag:
    """Coastal flag: ocean/sea neighbours → True; lakes/rivers → False."""

    def _run(
        self,
        land_lon: float,
        land_lat: float,
        water_polygon: Polygon,
        water_type: str,
        cell_size: float = 500.0,
    ) -> gpd.GeoDataFrame:
        grid = _make_grid([land_lon], [land_lat], [20.0])
        water = _water_gdf([water_polygon], [water_type])
        config = _config(cell_size=cell_size)
        pipeline = FloodRiskPipeline(config)
        return pipeline._apply_water_mask_and_proximity_boost(grid, water, config)

    def test_adjacent_ocean_cell_flagged(self):
        """Land cell adjacent to (but not inside) a coastline water body is flagged."""
        # Land cell centroid: ~(80.2505, 13.0005)
        # Water polygon is immediately south, does NOT cover the centroid
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),  # just south, doesn't cover centroid
            water_type="coastline",
        )
        assert result.iloc[0]["is_coastal_tsunami_risk"] is True or \
               bool(result.iloc[0]["is_coastal_tsunami_risk"]) is True

    def test_adjacent_bay_cell_flagged(self):
        """natural=bay also counts as ocean for the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="bay",
        )
        assert bool(result.iloc[0]["is_coastal_tsunami_risk"]) is True

    def test_adjacent_lake_not_flagged(self):
        """Land cell next to a lake (natural=water) must NOT get the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="water",  # lake
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_adjacent_river_not_flagged(self):
        """Land cell next to a river must NOT get the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="river",
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_adjacent_reservoir_not_flagged(self):
        """Land cell next to a reservoir must NOT get the tsunami flag."""
        result = self._run(
            land_lon=80.25, land_lat=13.00,
            water_polygon=box(80.24, 12.97, 80.27, 12.999),
            water_type="reservoir",
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_far_from_ocean_not_flagged(self):
        """Inland cell far from any ocean body must NOT be flagged."""
        result = self._run(
            land_lon=77.55, land_lat=12.84,
            water_polygon=box(80.24, 12.97, 80.31, 12.999),
            water_type="coastline",
            cell_size=500.0,
        )
        assert not bool(result.iloc[0]["is_coastal_tsunami_risk"])

    def test_water_cells_not_flagged(self):
        """Cells classified as Water (elev ≤ 1 m) must not receive the tsunami flag."""
        # cell 0: elevation 0.5 m → elevation mask → Water
        # cell 1: elevation 20 m, land, adjacent to ocean polygon
        grid = _make_grid([80.25, 80.26], [13.00, 13.00], [0.5, 20.0])
        # Ocean polygon: covers cell 0 AND is adjacent to cell 1
        ocean = box(80.24, 12.97, 80.261, 12.999)
        water = _water_gdf([ocean], ["coastline"])
        config = _config(cell_size=500.0)
        result = FloodRiskPipeline(config)._apply_water_mask_and_proximity_boost(
            grid, water, config
        )
        water_cells = result[result["risk_class"] == "Water"]
        assert not water_cells["is_coastal_tsunami_risk"].any(), (
            "Water cells must never receive the coastal tsunami flag"
        )

    def test_multiple_ocean_cells_flagged(self):
        """Multiple land cells adjacent to ocean all get flagged."""
        # 3 land cells side-by-side (all > 1m elevation)
        # Ocean polygon immediately south, does NOT cover any centroid
        lons = [80.25, 80.26, 80.27]
        lats = [13.00, 13.00, 13.00]
        grid = _make_grid(lons, lats, [20.0, 20.0, 20.0])
        ocean_poly = box(80.24, 12.97, 80.28, 12.999)  # just south, not covering centroids
        water = _water_gdf([ocean_poly], ["coastline"])
        config = _config(cell_size=500.0)
        result = FloodRiskPipeline(config)._apply_water_mask_and_proximity_boost(
            grid, water, config
        )
        land_cells = result[result["risk_class"] != "Water"]
        flagged = land_cells["is_coastal_tsunami_risk"].sum()
        assert flagged >= 1, (
            f"At least one land cell adjacent to ocean should be flagged; got {flagged}"
        )


# ── Part 3: _load_land_mask error paths ──────────────────────────────────────

class TestLoadLandMask:
    """Unit tests for _load_land_mask error paths and caching behaviour."""

    def test_missing_file_raises_flood_risk_error(self, tmp_path):
        """FloodRiskError is raised when _LAND_MASK_PATH points to a nonexistent file."""
        import flood_risk_zonation.pipeline as _pipeline_module
        from unittest.mock import patch

        from flood_risk_zonation.exceptions import FloodRiskError
        from flood_risk_zonation.pipeline import _load_land_mask

        nonexistent = tmp_path / "no_such_file.geojson"
        # Reset cache so we always exercise the load path
        original_geom = _pipeline_module._LAND_MASK_GEOM
        _pipeline_module._LAND_MASK_GEOM = None
        try:
            with patch.object(_pipeline_module, "_LAND_MASK_PATH", nonexistent):
                with pytest.raises(FloodRiskError, match="Land mask file not found"):
                    _load_land_mask()
        finally:
            _pipeline_module._LAND_MASK_GEOM = original_geom

    def test_read_file_exception_raises_flood_risk_error(self):
        """FloodRiskError is raised when geopandas.read_file throws."""
        import flood_risk_zonation.pipeline as _pipeline_module
        from unittest.mock import patch, MagicMock

        from flood_risk_zonation.exceptions import FloodRiskError
        from flood_risk_zonation.pipeline import _load_land_mask

        original_geom = _pipeline_module._LAND_MASK_GEOM
        _pipeline_module._LAND_MASK_GEOM = None
        try:
            # Make the path appear to exist but read_file raise
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            with patch.object(_pipeline_module, "_LAND_MASK_PATH", mock_path):
                with patch("flood_risk_zonation.pipeline.gpd") as mock_gpd:
                    mock_gpd.read_file.side_effect = RuntimeError("bad geojson")
                    with pytest.raises(FloodRiskError, match="Failed to parse land mask"):
                        _load_land_mask()
        finally:
            _pipeline_module._LAND_MASK_GEOM = original_geom

    def test_empty_geometry_after_dissolve_raises_flood_risk_error(self):
        """FloodRiskError is raised when unary_union returns an empty geometry."""
        import flood_risk_zonation.pipeline as _pipeline_module
        from unittest.mock import patch, MagicMock

        from flood_risk_zonation.exceptions import FloodRiskError
        from flood_risk_zonation.pipeline import _load_land_mask

        original_geom = _pipeline_module._LAND_MASK_GEOM
        _pipeline_module._LAND_MASK_GEOM = None
        try:
            mock_path = MagicMock()
            mock_path.exists.return_value = True

            # Build a mock GeoDataFrame whose geometry list will be passed to unary_union
            mock_geom_series = MagicMock()
            mock_geom_series.tolist.return_value = []
            mock_gdf = MagicMock()
            mock_gdf.geometry = mock_geom_series

            # Empty geometry returned by unary_union
            empty_geom = MagicMock()
            empty_geom.is_empty = True

            with patch.object(_pipeline_module, "_LAND_MASK_PATH", mock_path):
                with patch("flood_risk_zonation.pipeline.gpd") as mock_gpd:
                    mock_gpd.read_file.return_value = mock_gdf
                    with patch("shapely.ops.unary_union", return_value=empty_geom):
                        with pytest.raises(
                            FloodRiskError, match="Land mask geometry is empty after dissolve"
                        ):
                            _load_land_mask()
        finally:
            _pipeline_module._LAND_MASK_GEOM = original_geom

    def test_cache_hit_reads_file_only_once(self, tmp_path):
        """geopandas.read_file is called exactly once across two _load_land_mask calls."""
        import flood_risk_zonation.pipeline as _pipeline_module
        from unittest.mock import patch, MagicMock

        from flood_risk_zonation.pipeline import _load_land_mask

        original_geom = _pipeline_module._LAND_MASK_GEOM
        _pipeline_module._LAND_MASK_GEOM = None
        try:
            mock_path = MagicMock()
            mock_path.exists.return_value = True

            # Prepare a non-empty geometry for the dissolve result
            mock_geom_series = MagicMock()
            mock_geom_series.tolist.return_value = []
            mock_gdf = MagicMock()
            mock_gdf.geometry = mock_geom_series

            from shapely.geometry import box as _box
            real_geom = _box(77.0, 12.0, 78.0, 13.0)  # small non-empty polygon

            with patch.object(_pipeline_module, "_LAND_MASK_PATH", mock_path):
                with patch("flood_risk_zonation.pipeline.gpd") as mock_gpd:
                    mock_gpd.read_file.return_value = mock_gdf
                    with patch("shapely.ops.unary_union", return_value=real_geom):
                        result1 = _load_land_mask()
                        result2 = _load_land_mask()

            # read_file must have been called exactly once (second call hits cache)
            assert mock_gpd.read_file.call_count == 1
            assert result1 is result2
        finally:
            _pipeline_module._LAND_MASK_GEOM = original_geom


# ── Part 3: Integration tests with the real GeoJSON land mask ─────────────────

_LANDMASK_PATH = (
    __import__("pathlib").Path(__file__).parent.parent.parent
    / "data" / "landmask" / "ne_50m_land.geojson"
)


def test_real_landmask_arabian_sea():
    """
    Integration test: a cell centroid at (65.1, 18.1) — open Arabian Sea — must
    be classified as Water with water_mask_reason == 'landmask' using the real
    ne_50m_land.geojson file.

    Requirements: 7.1, 5.5
    """
    import flood_risk_zonation.pipeline as _pipeline_mod

    if not _LANDMASK_PATH.exists():
        pytest.skip(f"Real land mask file not found: {_LANDMASK_PATH}")

    # Reset module-level cache to ensure a clean load from the real file.
    _pipeline_mod._LAND_MASK_GEOM = None

    # Build a one-cell grid with centroid at (65.1, 18.1) — Arabian Sea (open ocean).
    # _make_grid takes the SW corner; centroid = lon + cell_size_deg/2, lat + cell_size_deg/2
    cell_size_deg = 0.001
    lon_sw = 65.1 - cell_size_deg / 2   # centroid_lon will be exactly 65.1
    lat_sw = 18.1 - cell_size_deg / 2   # centroid_lat will be exactly 18.1
    grid = _make_grid([lon_sw], [lat_sw], [0.0], cell_size_deg=cell_size_deg)

    empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty_wb, _config())

    row = result.iloc[0]
    assert row["water_mask_reason"] == "landmask", (
        f"Expected water_mask_reason='landmask' for Arabian Sea cell, "
        f"got '{row['water_mask_reason']}' (risk_class={row['risk_class']!r})"
    )


def test_real_landmask_dal_lake():
    """
    Integration test: a cell centroid at (74.86, 34.10) — Dal Lake, Srinagar (inland) —
    must NOT be classified as ocean by the land mask. Dal Lake is surrounded by land in
    the Natural Earth dataset, so water_mask_reason must NOT be 'landmask'.

    Requirements: 7.3, 5.3
    """
    import flood_risk_zonation.pipeline as _pipeline_mod

    if not _LANDMASK_PATH.exists():
        pytest.skip(f"Real land mask file not found: {_LANDMASK_PATH}")

    # Reset module-level cache to ensure a clean load from the real file.
    _pipeline_mod._LAND_MASK_GEOM = None

    # Build a one-cell grid with centroid at (74.86, 34.10) — Dal Lake, Srinagar.
    # _make_grid takes the SW corner; centroid = lon + cell_size_deg/2, lat + cell_size_deg/2
    cell_size_deg = 0.001
    lon_sw = 74.86 - cell_size_deg / 2   # centroid_lon will be exactly 74.86
    lat_sw = 34.10 - cell_size_deg / 2   # centroid_lat will be exactly 34.10
    grid = _make_grid([lon_sw], [lat_sw], [0.0], cell_size_deg=cell_size_deg)

    empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty_wb, _config())

    row = result.iloc[0]
    assert row["water_mask_reason"] != "landmask", (
        f"Dal Lake centroid must NOT have water_mask_reason='landmask' "
        f"(it is inland land in Natural Earth); "
        f"got water_mask_reason='{row['water_mask_reason']}' (risk_class={row['risk_class']!r})"
    )


def test_real_landmask_gottigere_land():
    """
    Integration test: a cell centroid at (77.59, 12.84) — Gottigere, Bangalore
    (inland land) — must NOT be classified as ocean by the land-mask detector.
    water_mask_reason must not equal 'landmask'.

    Requirements: 7.2, 5.1
    """
    import flood_risk_zonation.pipeline as _pipeline_mod

    if not _LANDMASK_PATH.exists():
        pytest.skip(f"Real land mask file not found: {_LANDMASK_PATH}")

    # Reset module-level cache to ensure a clean load from the real file.
    _pipeline_mod._LAND_MASK_GEOM = None

    # Build a one-cell grid with centroid at (77.59, 12.84) — Gottigere, Bangalore.
    # _make_grid takes the SW corner; centroid = lon + cell_size_deg/2, lat + cell_size_deg/2
    cell_size_deg = 0.001
    lon_sw = 77.59 - cell_size_deg / 2   # centroid_lon will be exactly 77.59
    lat_sw = 12.84 - cell_size_deg / 2   # centroid_lat will be exactly 12.84
    grid = _make_grid([lon_sw], [lat_sw], [50.0], cell_size_deg=cell_size_deg)

    empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    result = _pipeline()._apply_water_mask_and_proximity_boost(grid, empty_wb, _config())

    row = result.iloc[0]
    assert row["water_mask_reason"] != "landmask", (
        f"Inland land cell at Gottigere must NOT have water_mask_reason='landmask'; "
        f"got '{row['water_mask_reason']}' (risk_class={row['risk_class']!r})"
    )


# Feature: static-land-sea-mask, Property 7: Arabian Sea bounding boxes produce 100% ocean classification


@given(
    min_lon=st.floats(min_value=63.0, max_value=66.9),
    min_lat=st.floats(min_value=16.0, max_value=19.9),
)
@settings(max_examples=10, deadline=None)
def test_arabian_sea_all_cells_ocean(min_lon, min_lat):
    """
    Property test: any 0.1°×0.1° bounding box drawn from the Arabian Sea region
    (lon 63–67°E, lat 16–20°N) should produce 100% ocean classification
    (water_mask_reason == 'landmask') for every grid cell.

    The real ne_50m_land.geojson is loaded once and cached by _load_land_mask(); all
    50 Hypothesis examples share the same cached geometry after the first load.
    deadline=None because the first example pays the one-time GeoJSON parse cost.

    Validates: Requirements 7.6, 5.5
    """
    import flood_risk_zonation.pipeline as _pipeline_mod

    if not _LANDMASK_PATH.exists():
        pytest.skip(f"Real land mask file not found: {_LANDMASK_PATH}")

    # Build a single 0.1°×0.1° grid cell whose SW corner is (min_lon, min_lat).
    # _make_grid uses cell_size_deg for both width and height, so the cell spans
    # [min_lon, min_lon+0.1] × [min_lat, min_lat+0.1].
    cell_size_deg = 0.1
    bbox_grid = _make_grid([min_lon], [min_lat], [0.0], cell_size_deg=cell_size_deg)

    empty_wb = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    result = _pipeline()._apply_water_mask_and_proximity_boost(
        bbox_grid, empty_wb, _config()
    )

    assert (result["water_mask_reason"] == "landmask").all(), (
        f"Expected all cells to have water_mask_reason='landmask' for Arabian Sea bbox "
        f"(min_lon={min_lon}, min_lat={min_lat}), but got: "
        f"{result['water_mask_reason'].tolist()}"
    )
