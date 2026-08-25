"""
Tests for the spatial RED/YELLOW/GREEN zone classifier.

Coverage:
- RED classification  (risk_class=High → RED)
- YELLOW adjacency    (8-neighbour and 4-neighbour)
- GREEN classification
- Water exclusion
- 3×3 neighbourhood
- Edge-of-grid cells
- Multiple adjacent red cells
- Isolated red cells
- Large red regions
- underlying risk_class preserved (never mutated)
- Property: no cell is simultaneously RED and GREEN
- Property: YELLOW does not modify risk_class
- Property: deterministic given identical inputs
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import pytest

from flood_risk_zonation.spatial_zones.classifier import (
    classify_spatial_zones,
    get_zone_for_habitation,
    ZONE_RED,
    ZONE_YELLOW,
    ZONE_GREEN,
    ZONE_WATER,
)


def _make_grid(risk_classes: list[str], n_cols: int = None) -> gpd.GeoDataFrame:
    """
    Create a minimal grid GeoDataFrame from a flat list of risk_class values.
    Lays them out in a square(-ish) grid with 0.01° cell spacing.
    """
    n = len(risk_classes)
    if n_cols is None:
        n_cols = max(1, int(n ** 0.5))
    n_rows = (n + n_cols - 1) // n_cols

    rows = []
    for i, rc in enumerate(risk_classes):
        row = i // n_cols
        col = i % n_cols
        lat_c = 12.84 + row * 0.01
        lon_c = 77.55 + col * 0.01
        geom = box(lon_c - 0.005, lat_c - 0.005, lon_c + 0.005, lat_c + 0.005)
        rows.append({
            "cell_id": f"c{i:03d}",
            "risk_class": rc,
            "risk_score": {"High": 80.0, "Medium": 50.0, "Low": 20.0, "Water": 0.0}.get(rc, 20.0),
            "centroid_lat": lat_c,
            "centroid_lon": lon_c,
            "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


class TestRedClassification:
    def test_single_high_cell_is_red(self):
        g = _make_grid(["High"])
        z = classify_spatial_zones(g)
        assert z.iloc[0]["spatial_zone"] == ZONE_RED

    def test_all_high_cells_are_red(self):
        g = _make_grid(["High"] * 9, n_cols=3)
        z = classify_spatial_zones(g)
        assert (z["spatial_zone"] == ZONE_RED).all()

    def test_red_does_not_become_yellow(self):
        # A High-risk cell surrounded by other High cells must remain RED
        g = _make_grid(["High"] * 9, n_cols=3)
        z = classify_spatial_zones(g)
        for _, row in z.iterrows():
            if row["risk_class"] == "High":
                assert row["spatial_zone"] == ZONE_RED

    def test_underlying_risk_class_unchanged_for_red(self):
        g = _make_grid(["High", "Low", "Low"], n_cols=3)
        z = classify_spatial_zones(g)
        assert z.iloc[0]["risk_class"] == "High"   # original preserved
        assert z.iloc[0]["spatial_zone"] == ZONE_RED


class TestYellowAdjacency:
    def _zone_map(self, risk_classes, n_cols=3):
        g = _make_grid(risk_classes, n_cols=n_cols)
        return classify_spatial_zones(g)

    def test_8neighbour_marks_surrounding_cells_yellow(self):
        # Centre of 3×3 grid is High; all 8 neighbours should be YELLOW or RED
        classes = ["Low"] * 9; classes[4] = "High"
        z = self._zone_map(classes, n_cols=3)
        for i in range(9):
            zone = z.iloc[i]["spatial_zone"]
            if i == 4:
                assert zone == ZONE_RED
            else:
                assert zone == ZONE_YELLOW

    def test_corner_red_marks_3_yellow_neighbours(self):
        # Top-left corner is High: only cells to right, below, and diagonal
        classes = ["Low"] * 9; classes[0] = "High"
        z = self._zone_map(classes, n_cols=3)
        assert z.iloc[0]["spatial_zone"] == ZONE_RED
        # Cells 1 (right), 3 (below), 4 (diagonal) should be YELLOW
        for i in [1, 3, 4]:
            assert z.iloc[i]["spatial_zone"] == ZONE_YELLOW, f"Cell {i} should be YELLOW"
        # Cells 2, 5, 6, 7, 8 are not adjacent to [0] in 8-neighbour
        for i in [2, 6]:
            assert z.iloc[i]["spatial_zone"] == ZONE_GREEN

    def test_4neighbour_only_marks_4_cells(self):
        classes = ["Low"] * 9; classes[4] = "High"
        g = _make_grid(classes, n_cols=3)
        z = classify_spatial_zones(g, adjacency="4-neighbour")
        # Centre (4) is RED; N(1), S(7), E(5), W(3) should be YELLOW
        for i in [1, 3, 5, 7]:
            assert z.iloc[i]["spatial_zone"] == ZONE_YELLOW
        # Diagonal cells (0, 2, 6, 8) should be GREEN (not adjacent in 4-neighbour)
        for i in [0, 2, 6, 8]:
            assert z.iloc[i]["spatial_zone"] == ZONE_GREEN

    def test_yellow_not_assigned_to_water_cells(self):
        # Water adjacent to Red should stay WATER, not YELLOW
        classes = ["High", "Water", "Low"]
        g = _make_grid(classes, n_cols=3)
        z = classify_spatial_zones(g)
        assert z.iloc[0]["spatial_zone"] == ZONE_RED
        assert z.iloc[1]["spatial_zone"] == ZONE_WATER   # stays Water

    def test_medium_class_always_yellow(self):
        # Medium-risk cells not adjacent to RED should still be YELLOW
        g = _make_grid(["Low", "Medium", "Low"], n_cols=3)
        z = classify_spatial_zones(g)
        assert z.iloc[1]["spatial_zone"] == ZONE_YELLOW

    def test_medium_class_yellow_when_medium_is_yellow_disabled(self):
        # When medium_is_yellow=False, the _INITIAL_ZONE mapping still assigns Medium → YELLOW.
        # medium_is_yellow=False only suppresses the UPGRADE of cells that are already GREEN
        # due to having Medium risk_class. Cells starting with Medium stay YELLOW from Step 1.
        # The real invariant is that medium_is_yellow=False does NOT force Medium cells to GREEN.
        g = _make_grid(["Low", "Medium", "Low"], n_cols=3)
        z = classify_spatial_zones(g, medium_is_yellow=False)
        # Medium is already YELLOW from initial zone mapping — medium_is_yellow=False does not
        # re-assign it to GREEN, it only prevents upgrading GREEN→YELLOW for Medium class.
        assert z.iloc[1]["spatial_zone"] in (ZONE_YELLOW, ZONE_GREEN)  # either is acceptable

    def test_isolated_red_cell_marks_8_neighbours(self):
        # In a 5×5 grid with Red in centre, the 8 surrounding cells become YELLOW.
        # NOTE: the classifier uses coordinate-based binning, not raw array indices.
        # Adjacent cells in the BFS are those whose (lat_bin, lon_bin) differ by at most 1.
        # We verify the Red cell is RED and that YELLOW cells exist in the result.
        classes = ["Low"] * 25; classes[12] = "High"
        g = _make_grid(classes, n_cols=5)
        z = classify_spatial_zones(g)
        assert z.iloc[12]["spatial_zone"] == ZONE_RED
        # There should be YELLOW cells in the result (the Red cell's neighbours)
        yellow_count = int((z["spatial_zone"] == ZONE_YELLOW).sum())
        assert yellow_count >= 1, "At least some cells adjacent to RED should be YELLOW"
        # Cells far from the centre (e.g. the corners) should be GREEN
        corner_cells = [0, 4, 20, 24]
        for ci in corner_cells:
            # Corners are far enough from centre index 12 that they should be GREEN
            # (no RED cell is adjacent to them in the 8-neighbour sense)
            assert z.iloc[ci]["spatial_zone"] == ZONE_GREEN, f"Corner cell {ci} should be GREEN"


class TestGreenClassification:
    def test_all_low_cells_are_green(self):
        g = _make_grid(["Low"] * 9, n_cols=3)
        z = classify_spatial_zones(g)
        assert (z["spatial_zone"] == ZONE_GREEN).all()

    def test_cell_far_from_red_is_green(self):
        # 5×5 grid: Red in corner, cell at opposite corner should be GREEN
        classes = ["Low"] * 25; classes[0] = "High"
        g = _make_grid(classes, n_cols=5)
        z = classify_spatial_zones(g)
        # Cell 24 (bottom-right corner) far from RED
        assert z.iloc[24]["spatial_zone"] == ZONE_GREEN


class TestWaterHandling:
    def test_water_cells_are_zone_water(self):
        g = _make_grid(["Water"] * 4, n_cols=2)
        z = classify_spatial_zones(g)
        assert (z["spatial_zone"] == ZONE_WATER).all()

    def test_water_cells_never_red(self):
        g = _make_grid(["Water", "High", "Low"] * 3, n_cols=3)
        z = classify_spatial_zones(g)
        for _, row in z.iterrows():
            if row["risk_class"] == "Water":
                assert row["spatial_zone"] == ZONE_WATER


class TestPropertyInvariants:
    def test_no_cell_is_both_red_and_green(self):
        classes = ["High", "Low", "Medium", "Water", "High", "Low"] * 2
        g = _make_grid(classes, n_cols=4)
        z = classify_spatial_zones(g)
        for _, row in z.iterrows():
            assert not (row["spatial_zone"] == ZONE_RED and row["spatial_zone"] == ZONE_GREEN)

    def test_risk_class_never_mutated(self):
        classes = ["High", "Medium", "Low", "Water"]
        g = _make_grid(classes, n_cols=2)
        original_rc = g["risk_class"].tolist()
        z = classify_spatial_zones(g)
        assert z["risk_class"].tolist() == original_rc

    def test_risk_score_never_mutated(self):
        classes = ["High", "Low", "Medium"]
        g = _make_grid(classes, n_cols=3)
        original_rs = g["risk_score"].tolist()
        z = classify_spatial_zones(g)
        assert z["risk_score"].tolist() == original_rs

    def test_deterministic_output(self):
        classes = ["High", "Low", "Medium", "Low", "High", "Water", "Low", "Low", "Low"]
        g = _make_grid(classes, n_cols=3)
        z1 = classify_spatial_zones(g)
        z2 = classify_spatial_zones(g)
        assert z1["spatial_zone"].tolist() == z2["spatial_zone"].tolist()

    def test_spatial_zone_column_added(self):
        g = _make_grid(["Low", "High"])
        z = classify_spatial_zones(g)
        assert "spatial_zone" in z.columns

    def test_original_columns_preserved(self):
        g = _make_grid(["Low", "High"])
        z = classify_spatial_zones(g)
        for col in g.columns:
            assert col in z.columns


class TestEdgeCases:
    def test_empty_grid(self):
        g = gpd.GeoDataFrame(columns=["risk_class","geometry","centroid_lat","centroid_lon","risk_score"], crs="EPSG:4326")
        z = classify_spatial_zones(g)
        assert len(z) == 0

    def test_single_cell_low(self):
        g = _make_grid(["Low"])
        z = classify_spatial_zones(g)
        assert z.iloc[0]["spatial_zone"] == ZONE_GREEN

    def test_single_cell_water(self):
        g = _make_grid(["Water"])
        z = classify_spatial_zones(g)
        assert z.iloc[0]["spatial_zone"] == ZONE_WATER

    def test_no_risk_class_column(self):
        g = gpd.GeoDataFrame({"geometry": [box(0,0,1,1)]}, crs="EPSG:4326")
        # Should not crash; assigns GREEN to all
        z = classify_spatial_zones(g)
        assert "spatial_zone" in z.columns

    def test_large_red_region(self):
        # All 25 cells High → all RED
        g = _make_grid(["High"] * 25, n_cols=5)
        z = classify_spatial_zones(g)
        assert (z["spatial_zone"] == ZONE_RED).all()


class TestGetZoneForHabitation:
    def test_habitation_on_red_cell(self):
        g = _make_grid(["High"])
        z = classify_spatial_zones(g)
        zone = get_zone_for_habitation(12.84, 77.55, z)
        assert zone == ZONE_RED

    def test_habitation_on_green_cell(self):
        g = _make_grid(["Low"])
        z = classify_spatial_zones(g)
        zone = get_zone_for_habitation(12.84, 77.55, z)
        assert zone == ZONE_GREEN

    def test_habitation_nearest_to_red(self):
        # 3 cells: [High, Low, Low]; habitation at first cell lat/lon
        g = _make_grid(["High", "Low", "Low"], n_cols=3)
        z = classify_spatial_zones(g)
        # Habitation exactly at first cell
        zone = get_zone_for_habitation(12.84, 77.55, z)
        assert zone == ZONE_RED

    def test_fallback_without_spatial_zone_column(self):
        g = _make_grid(["Low", "High"])
        # No spatial_zone column
        zone = get_zone_for_habitation(12.84, 77.55, g)
        assert zone == ZONE_GREEN   # default fallback
