"""
Unit tests for Nepal active_flood_override behavior.

Tests verify:
  T1 -- check_active_flooding accepts active_flood_override parameter
  T2 -- Override only activates when verification FAILS (CHECK_FAILED or INSUFFICIENT_EVIDENCE)
  T3 -- Override does NOT activate when verification succeeds with NO_ACTIVE_FLOODING
  T4 -- Override does NOT activate when ACTIVE_FLOODING is already detected
  T5 -- Override converts CHECK_FAILED to ACTIVE_FLOODING with confidence 0.5
  T6 -- Override converts INSUFFICIENT_EVIDENCE to ACTIVE_FLOODING
  T7 -- Override does NOT affect non-override regions
  T8 -- Override sets appropriate summary message with developer authorization note
"""
from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from flood_risk_zonation.verification.active_flood_check import check_active_flooding, CACHE_DIR
from flood_risk_zonation.verification.models import ActiveFloodVerificationResult


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the active flood cache before each test."""
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    yield
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)


class TestOverrideParameter:
    """T1: check_active_flooding accepts active_flood_override parameter."""

    def test_function_signature_accepts_override(self):
        """check_active_flooding must accept active_flood_override parameter."""
        import inspect
        sig = inspect.signature(check_active_flooding)
        assert "active_flood_override" in sig.parameters, (
            "check_active_flooding must have active_flood_override parameter"
        )

    def test_override_parameter_is_boolean(self):
        """active_flood_override parameter should be bool with default False."""
        import inspect
        sig = inspect.signature(check_active_flooding)
        param = sig.parameters["active_flood_override"]
        assert param.default is False or param.default is None, (
            "active_flood_override should default to False"
        )


class TestOverrideActivatesOnVerificationFailure:
    """T2: Override activates when verification FAILS."""

    @patch("flood_risk_zonation.verification.active_flood_check._fetch_active_flood_evidence")
    def test_override_activates_on_check_failed(self, mock_fetch):
        """Override should convert CHECK_FAILED to ACTIVE_FLOODING."""
        # Simulate a scenario where check would fail
        mock_fetch.side_effect = Exception("Network error")
        
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        # With override, CHECK_FAILED should become ACTIVE_FLOODING
        assert result.status == "ACTIVE_FLOODING", (
            f"Override failed: expected ACTIVE_FLOODING, got {result.status}"
        )
        assert result.confidence == 0.5, (
            f"Override should set confidence to 0.5, got {result.confidence}"
        )

    def test_override_activates_on_insufficient_evidence(self):
        """Override should convert INSUFFICIENT_EVIDENCE to ACTIVE_FLOODING."""
        # Simulate missing API key (INSUFFICIENT_EVIDENCE state)
        with patch.dict(os.environ, {}, clear=True):
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        # With override enabled and INSUFFICIENT_EVIDENCE, should become ACTIVE_FLOODING
        assert result.status == "ACTIVE_FLOODING", (
            f"Override should activate on INSUFFICIENT_EVIDENCE, got {result.status}"
        )


class TestOverrideDoesNotActivateOnSuccess:
    """T3-T4: Override does NOT activate when verification succeeds."""

    @patch("flood_risk_zonation.verification.active_flood_check._fetch_active_flood_evidence")
    def test_override_respects_no_active_flooding(self, mock_fetch):
        """Override must NOT override NO_ACTIVE_FLOODING result."""
        mock_fetch.return_value = []  # No evidence = NO_ACTIVE_FLOODING
        
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        # Override should NOT trigger when verification succeeds
        assert result.status == "NO_ACTIVE_FLOODING", (
            f"Override should NOT override successful NO_ACTIVE_FLOODING, got {result.status}"
        )

    @patch("flood_risk_zonation.verification.active_flood_check._fetch_active_flood_evidence")
    def test_override_respects_active_flooding(self, mock_fetch):
        """Override must NOT change verified ACTIVE_FLOODING result."""
        from flood_risk_zonation.verification.models import FloodEvidence
        
        # Simulate evidence indicating active flooding
        evidence = FloodEvidence(
            source="Test Source",
            title="Current Flooding",
            location="Test",
            timestamp=datetime.now(timezone.utc),
            evidence_text="Active flood now happening",
            indicates_active_flooding=True,
            confidence=0.9,
        )
        mock_fetch.return_value = [evidence]
        
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        # Should remain ACTIVE_FLOODING (verified, not overridden)
        assert result.status == "ACTIVE_FLOODING", (
            f"Result should be ACTIVE_FLOODING from verification, got {result.status}"
        )
        assert result.confidence == 0.9, (
            f"Should use verified confidence (0.9), not override (0.5), got {result.confidence}"
        )


class TestOverrideConversionDetails:
    """T5-T6: Override properly converts failed states."""

    def test_override_check_failed_confidence(self):
        """Override should set confidence to 0.5 when converting CHECK_FAILED."""
        with patch.dict(os.environ, {}, clear=True):  # INSUFFICIENT_EVIDENCE
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        assert result.status == "ACTIVE_FLOODING"
        assert result.confidence == 0.5

    def test_override_summary_includes_developer_note(self):
        """Override summary should mention developer authorization."""
        with patch.dict(os.environ, {}, clear=True):
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        assert "Developer-authorized" in result.summary or "override" in result.summary.lower(), (
            f"Summary should mention developer authorization, got: {result.summary}"
        )

    def test_override_fallback_reason_set(self):
        """Override should set fallback_reason to explain override activation."""
        with patch.dict(os.environ, {}, clear=True):
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        assert result.fallback_reason is not None
        assert "override" in result.fallback_reason.lower() or "verification" in result.fallback_reason.lower()


class TestOverrideDoesNotAffectNonOverrideRegions:
    """T7: Override parameter does NOT affect non-override calls."""

    def test_override_false_respects_no_active_flooding(self):
        """With override=False, NO_ACTIVE_FLOODING should stay NO_ACTIVE_FLOODING."""
        with patch.dict(os.environ, {}, clear=True):
            result = check_active_flooding(
                location_name="Test",
                lat=28.3,
                lon=85.9,
                active_flood_override=False,  # Explicitly disabled
            )
        
        # Should NOT become ACTIVE_FLOODING
        assert result.status == "INSUFFICIENT_EVIDENCE", (
            f"Without override, should be INSUFFICIENT_EVIDENCE, got {result.status}"
        )

    def test_override_false_on_check_failed_stays_failed(self):
        """With override=False, CHECK_FAILED should stay CHECK_FAILED."""
        with patch("flood_risk_zonation.verification.active_flood_check._fetch_active_flood_evidence") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            
            with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
                result = check_active_flooding(
                    location_name="Test",
                    lat=28.3,
                    lon=85.9,
                    active_flood_override=False,  # Explicitly disabled
                )
            
            # Should stay CHECK_FAILED, not become ACTIVE_FLOODING
            assert result.status == "CHECK_FAILED", (
                f"Without override, CHECK_FAILED should remain, got {result.status}"
            )


class TestOverrideSummaryMessages:
    """T8: Override produces appropriate summary messages."""

    def test_override_activated_message(self):
        """Override activation message should be clear and auditable."""
        with patch.dict(os.environ, {}, clear=True):
            result = check_active_flooding(
                location_name="Nepal Test Region",
                lat=28.3,
                lon=85.9,
                active_flood_override=True,
            )
        
        # Message should be clear about authorization
        assert "Developer" in result.summary or "override" in result.summary.lower()

    def test_failed_verification_described(self):
        """Summary should indicate what verification failed."""
        with patch("flood_risk_zonation.verification.active_flood_check._fetch_active_flood_evidence") as mock_fetch:
            mock_fetch.side_effect = Exception("API connection failed")
            
            with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
                result = check_active_flooding(
                    location_name="Test",
                    lat=28.3,
                    lon=85.9,
                    active_flood_override=True,
                )
            
            # Message should reference the failure reason
            assert "verification" in result.summary.lower() or "failed" in result.summary.lower()

