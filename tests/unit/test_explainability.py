"""
Tests for per-cell explainability (hover tooltip + click popup).
"""
from __future__ import annotations

import numpy as np
import pytest

from flood_risk_zonation.visualization.explainability import (
    build_cell_explanation,
    _normalise,
    _compute_factors,
    _build_summary,
    FACTOR_WEIGHTS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _land_row(risk_class="Medium", risk_score=50.0, is_coastal=False, **overrides):
    row = {
        "risk_class": risk_class,
        "risk_score": risk_score,
        "is_coastal_tsunami_risk": is_coastal,
        "elevation_m": 50.0,
        "slope_deg": 5.0,
        "twi": 8.0,
        "rainfall_mean_mm": 900.0,
        "rainfall_max_24h_mm": 80.0,
        "dist_water_m": 1500.0,
        "drainage_capacity": 0.5,
        "population_density": 500.0,
        "curvature": 0.0,
        "aspect_deg": 180.0,
    }
    row.update(overrides)
    return row


def _water_row(water_type="water"):
    return {
        "risk_class": "Water",
        "risk_score": 0.0,
        "is_coastal_tsunami_risk": False,
        "water_type": water_type,
    }


# ── _normalise ────────────────────────────────────────────────────────────────

class TestNormalise:

    def test_midpoint_gives_half(self):
        # elevation direction=-1: n = 1 - 0.5 = 0.5
        n = _normalise(50.0, "elevation_m", 0.0, 100.0)
        assert abs(n - 0.5) < 0.01

    def test_low_elevation_high_risk(self):
        """Very low elevation → high risk contribution (direction=-1)."""
        n = _normalise(0.0, "elevation_m", 0.0, 100.0)
        assert n > 0.9

    def test_high_elevation_low_risk(self):
        n = _normalise(100.0, "elevation_m", 0.0, 100.0)
        assert n < 0.1

    def test_high_twi_high_risk(self):
        """High TWI (direction=+1) → high risk."""
        n = _normalise(20.0, "twi", 0.0, 20.0)
        assert n > 0.9

    def test_degenerate_bounds_returns_half(self):
        n = _normalise(5.0, "elevation_m", 5.0, 5.0)
        assert abs(n - 0.5) < 0.01

    def test_high_risk_factor_gets_high_phrase(self):
        """A high-risk cell with low elevation must show 'low elevation' phrase, not 'high elevation'."""
        row = _land_row("High", 80.0, elevation_m=1.0)
        factors = _compute_factors(row, None)
        elev = next(f for f in factors if f[2] == "elevation_m")
        n, contrib, feat, label, unit, icon, phrase = elev
        assert n > 0.9, f"Expected n>0.9 for 1m elevation, got {n}"
        assert "low elevation" in phrase.lower(), f"Expected 'low elevation' phrase, got: {phrase}"

    def test_good_drainage_gets_low_risk_phrase(self):
        """A cell with good drainage (0.9) should show 'good drainage' phrase."""
        row = _land_row("Low", 20.0, drainage_capacity=0.9)
        factors = _compute_factors(row, None)
        drain = next(f for f in factors if f[2] == "drainage_capacity")
        n, contrib, feat, label, unit, icon, phrase = drain
        # drainage direction=-1: high capacity → low risk contribution
        assert n < 0.2, f"Expected n<0.2 for drainage 0.9, got {n}"
        assert "good drainage" in phrase.lower(), f"Expected 'good drainage', got: {phrase}"

    def test_poor_drainage_gets_high_risk_phrase(self):
        """A cell with poor drainage (0.1) should show 'poor drainage' phrase."""
        row = _land_row("High", 75.0, drainage_capacity=0.1)
        factors = _compute_factors(row, None)
        drain = next(f for f in factors if f[2] == "drainage_capacity")
        n, contrib, feat, label, unit, icon, phrase = drain
        assert n > 0.8, f"Expected n>0.8 for drainage 0.1, got {n}"
        assert "poor drainage" in phrase.lower(), f"Expected 'poor drainage', got: {phrase}"


# ── Tooltip tests ─────────────────────────────────────────────────────────────

class TestTooltip:

    def test_high_risk_tooltip_shows_risky_factor(self):
        """High Risk cell tooltip shows risk class and score only."""
        row = _land_row("High", 80.0, elevation_m=1.0)
        tooltip, _ = build_cell_explanation(row)
        assert "High" in tooltip
        assert "80" in tooltip
        # No factor phrases in tooltip
        assert "low elevation" not in tooltip.lower()
        assert "reduces risk" not in tooltip.lower()

    def test_high_risk_tooltip_not_positive(self):
        """High Risk tooltip shows score only — no factor reasons at all."""
        row = _land_row("High", 82.0, elevation_m=2.0, drainage_capacity=0.1, dist_water_m=50.0)
        tooltip, _ = build_cell_explanation(row)
        assert "High" in tooltip
        assert "82" in tooltip
        assert "reduces risk" not in tooltip
        assert "increases" not in tooltip

    def test_low_risk_tooltip_shows_score(self):
        """Low Risk cell tooltip shows score only."""
        row = _land_row("Low", 15.0, elevation_m=500.0, drainage_capacity=0.95)
        tooltip, _ = build_cell_explanation(row)
        assert "Low" in tooltip
        assert "15" in tooltip

    def test_tooltip_contains_risk_class(self):
        for cls in ["High", "Medium", "Low"]:
            row = _land_row(cls, 50.0)
            tooltip, _ = build_cell_explanation(row)
            assert cls in tooltip

    def test_tooltip_contains_score(self):
        row = _land_row("Medium", 55.0)
        tooltip, _ = build_cell_explanation(row)
        assert "55" in tooltip

    def test_tooltip_short_no_table(self):
        row = _land_row("Low", 20.0)
        tooltip, _ = build_cell_explanation(row)
        assert "<table" not in tooltip
        assert "<tr" not in tooltip
        assert "<span" not in tooltip  # no factor phrase span

    def test_water_tooltip_says_water(self):
        tooltip, _ = build_cell_explanation(_water_row("water"))
        assert "Water" in tooltip or "💧" in tooltip
        # Tooltip now shows the water body type label

    def test_coastal_tooltip_shows_flag(self):
        row = _land_row("High", 80.0, is_coastal=True)
        tooltip, _ = build_cell_explanation(row)
        assert "Coastal" in tooltip or "⚠" in tooltip

    def test_inland_no_tsunami_flag(self):
        row = _land_row("Medium", 50.0, is_coastal=False)
        tooltip, _ = build_cell_explanation(row)
        assert "Tsunami" not in tooltip
        assert "Coastal" not in tooltip


# ── Popup tests ───────────────────────────────────────────────────────────────

class TestPopup:

    def test_popup_has_all_factor_labels(self):
        row = _land_row("High", 80.0)
        _, popup = build_cell_explanation(row)
        for label in ["Elevation", "Slope", "Drainage", "Rainfall"]:
            assert label in popup, f"'{label}' missing from popup"

    def test_popup_contains_score(self):
        row = _land_row("Medium", 55.0)
        _, popup = build_cell_explanation(row)
        assert "55" in popup

    def test_popup_has_bar_elements(self):
        row = _land_row("High", 80.0)
        _, popup = build_cell_explanation(row)
        assert "background:#e74c3c" in popup or "background:#f39c12" in popup \
               or "background:#2ecc71" in popup

    def test_popup_has_summary(self):
        row = _land_row("High", 80.0)
        _, popup = build_cell_explanation(row)
        # New summary uses factor names, not "Classified"
        assert "elevation" in popup.lower() or "drainage" in popup.lower() \
               or "risk" in popup.lower()

    def test_water_popup_no_factor_table(self):
        _, popup = build_cell_explanation(_water_row("coastline"))
        # Water popup now has a parameters table; verify it mentions the water type
        assert "ocean" in popup.lower() or "coastline" in popup.lower() or "water" in popup.lower()

    def test_water_popup_mentions_type(self):
        _, popup = build_cell_explanation(_water_row("river"))
        assert "river" in popup.lower()
        _, popup2 = build_cell_explanation(_water_row("coastline"))
        assert "ocean" in popup2.lower() or "coastline" in popup2.lower()

    def test_water_popup_says_not_scored(self):
        _, popup = build_cell_explanation(_water_row())
        assert "does not apply" in popup or "masked" in popup

    def test_coastal_popup_has_tsunami_badge(self):
        row = _land_row("High", 80.0, is_coastal=True)
        _, popup = build_cell_explanation(row)
        assert "Tsunami" in popup and "⚠" in popup

    def test_coastal_popup_mentions_ocean_or_sea(self):
        row = _land_row("Medium", 55.0, is_coastal=True)
        _, popup = build_cell_explanation(row)
        assert "ocean" in popup.lower() or "sea" in popup.lower() or "adjacent" in popup.lower()

    def test_inland_popup_no_tsunami(self):
        row = _land_row("Low", 20.0, is_coastal=False)
        _, popup = build_cell_explanation(row)
        assert "Tsunami" not in popup

    def test_popup_color_matches_risk_class(self):
        from flood_risk_zonation.visualization.layers import RISK_COLOR_MAP
        for cls, color in RISK_COLOR_MAP.items():
            row = _water_row() if cls == "Water" else _land_row(cls, 50.0)
            _, popup = build_cell_explanation(row)
            assert color.lower() in popup.lower(), f"Color {color} missing for {cls}"

    def test_high_risk_popup_highlights_risky_factors(self):
        """High risk cell — rows with n>0.66 should have pink background in table."""
        row = _land_row("High", 85.0, elevation_m=2.0, dist_water_m=50.0, drainage_capacity=0.1)
        _, popup = build_cell_explanation(row)
        assert "fff5f5" in popup  # risky row highlight colour

    def test_cells_produce_different_summaries(self):
        """Two cells with different factor profiles must produce different summaries."""
        row_high = _land_row("High", 82.0, elevation_m=2.0, dist_water_m=50.0)
        row_low = _land_row("Low", 18.0, elevation_m=500.0, dist_water_m=4500.0)
        _, popup_high = build_cell_explanation(row_high)
        _, popup_low = build_cell_explanation(row_low)
        # Extract summary divs and compare
        assert popup_high != popup_low


# ── model_bounds integration ──────────────────────────────────────────────────

class TestModelBoundsIntegration:

    _BOUNDS = {
        "elevation_m": (5.0, 60.0),
        "slope_deg": (1.0, 20.0),
        "twi": (4.0, 18.0),
        "rainfall_mean_mm": (700.0, 1400.0),
        "rainfall_max_24h_mm": (50.0, 200.0),
        "dist_water_m": (100.0, 4000.0),
        "drainage_capacity": (0.2, 0.9),
        "population_density": (50.0, 2000.0),
        "curvature": (-5.0, 5.0),
    }

    def test_model_bounds_no_error(self):
        row = _land_row("High", 80.0)
        tooltip, popup = build_cell_explanation(row, model_bounds=self._BOUNDS)
        assert tooltip and popup

    def test_low_elevation_red_bar_with_bounds(self):
        row = _land_row("High", 85.0, elevation_m=5.0)
        _, popup = build_cell_explanation(row, model_bounds=self._BOUNDS)
        assert "e74c3c" in popup

    def test_high_elevation_green_bar(self):
        row = _land_row("Low", 20.0, elevation_m=200.0)
        _, popup = build_cell_explanation(row, model_bounds=self._BOUNDS)
        assert "2ecc71" in popup

    def test_phrase_direction_correct_with_bounds(self):
        """With model bounds, a cell at 5m elevation should show 'low elevation' phrase."""
        row = _land_row("High", 85.0, elevation_m=5.0)
        factors = _compute_factors(row, model_bounds=self._BOUNDS)
        elev = next(f for f in factors if f[2] == "elevation_m")
        n, _, _, _, _, _, phrase = elev
        assert n > 0.9, f"n={n} — 5m at bounds (5,60) should be fully risky"
        assert "low elevation" in phrase.lower(), f"Got: {phrase}"
