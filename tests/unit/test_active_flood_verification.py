"""
Unit tests for active flood verification.
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from flood_risk_zonation.verification.models import (
    FloodEvidence,
    ActiveFloodVerificationResult,
)
from flood_risk_zonation.verification.active_flood_check import (
    check_active_flooding,
)


class TestFloodEvidence:
    def test_create_valid_evidence(self):
        evidence = FloodEvidence(
            source="news_api",
            title="Heavy flooding",
            location="Test",
            timestamp=datetime.now(timezone.utc),
            evidence_text="Test",
            indicates_active_flooding=True,
            confidence=0.85,
        )
        assert evidence.confidence == 0.85


class TestActiveFloodResult:
    def test_active_flooding_result(self):
        result = ActiveFloodVerificationResult(
            status="ACTIVE_FLOODING",
            location_name="Test",
            location_lat=12.0,
            location_lon=77.0,
            verification_timestamp=datetime.now(timezone.utc),
            evidence_list=[],
            primary_evidence=None,
            summary="Test",
            confidence=0.9,
        )
        assert result.is_active_flood_gate() is True

    def test_invalid_status_raises_error(self):
        with pytest.raises(ValueError):
            ActiveFloodVerificationResult(
                status="INVALID",
                location_name="Test",
                location_lat=12.0,
                location_lon=77.0,
                verification_timestamp=datetime.now(timezone.utc),
                evidence_list=[],
                primary_evidence=None,
                summary="Test",
                confidence=0.5,
            )


class TestCheckActiveFlooding:
    @patch("requests.get")
    def test_empty_results_no_flooding(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"articles": []}
        mock_get.return_value = mock_response
        
        with patch.dict("os.environ", {"NEWS_API_KEY": "test"}):
            result = check_active_flooding("Test", 12.0, 77.0)
            assert result.status == "NO_ACTIVE_FLOODING"


class TestRegression:
    def test_models_work(self):
        result = ActiveFloodVerificationResult(
            status="NO_ACTIVE_FLOODING",
            location_name="Test",
            location_lat=12.0,
            location_lon=77.0,
            verification_timestamp=datetime.now(timezone.utc),
            evidence_list=[],
            primary_evidence=None,
            summary="Test",
            confidence=0.8,
        )
        assert result.should_continue_normal_pipeline() is True
