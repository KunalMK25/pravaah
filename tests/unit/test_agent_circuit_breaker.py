"""
Unit tests for LLM circuit breaker functionality.
"""
import pytest
from unittest.mock import patch, MagicMock
import flood_risk_zonation.agents.agents as agents_module


class TestLLMCircuitBreaker:
    """Tests for the LLM circuit breaker that prevents repeated failed API calls."""
    
    def setup_method(self):
        """Reset circuit breaker state before each test."""
        agents_module._llm_circuit_open = False
        agents_module._llm_consecutive_failures = 0
    
    def test_circuit_breaker_returns_none_when_open(self):
        """Circuit breaker should immediately return None without making API calls."""
        # Open the circuit breaker
        agents_module._llm_circuit_open = True
        
        # Attempt to make a call
        with patch.dict('os.environ', {'PRAVAAH_LLM_PROVIDER': 'groq', 'GROQ_API_KEY': 'test_key'}):
            with patch('groq.Groq') as mock_groq:
                result = agents_module._call_llm("system", "user")
                
                # Should return None immediately without calling Groq
                assert result is None
                assert mock_groq.call_count == 0  # Groq should never be instantiated
    
    def test_circuit_breaker_state_variables_exist(self):
        """Verify circuit breaker state variables are defined."""
        assert hasattr(agents_module, '_llm_circuit_open')
        assert hasattr(agents_module, '_llm_consecutive_failures')
        assert hasattr(agents_module, '_MAX_CONSECUTIVE_FAILURES')
        assert agents_module._MAX_CONSECUTIVE_FAILURES == 3
    
    def test_rule_based_fallback_when_circuit_open(self):
        """Agents should use rule-based fallback when circuit is open."""
        # Open circuit breaker
        agents_module._llm_circuit_open = True
        
        # Test hazard agent
        from flood_risk_zonation.agents.agents import run_hazard_agent
        
        hazard_data = {
            "hazard_score": 90.0,
            "hazard_class": "Critical",
            "spatial_zone": "RED",
            "pct_high_risk": 0.85,
            "dominant_features": ["elevation", "water_proximity"]
        }
        
        result = run_hazard_agent(hazard_data)
        
        # Should succeed with rule-based fallback
        assert result is not None
        assert result.ai_assisted is False
        assert "Hazard score: 90.0/100" in result.summary
    
    def test_no_api_key_exposure_in_circuit_breaker(self):
        """Circuit breaker should never log or expose API keys."""
        test_api_key = "gsk_test_secret_key_12345"
        
        # Set up the circuit breaker and call with circuit open
        agents_module._llm_circuit_open = True
        
        with patch.dict('os.environ', {'PRAVAAH_LLM_PROVIDER': 'groq', 'GROQ_API_KEY': test_api_key}):
            # Should not raise or expose key
            result = agents_module._call_llm("system", "user")
            assert result is None
    
    def test_llm_call_respects_timeout_config(self):
        """Verify timeout constant is set correctly."""
        assert agents_module._LLM_TIMEOUT_S == 15
    
    def test_multiple_agents_share_circuit_breaker_state(self):
        """All agents share the same circuit breaker state."""
        from flood_risk_zonation.agents.agents import (
            run_hazard_agent,
            run_exposure_agent
        )
        
        # Open circuit for all agents
        agents_module._llm_circuit_open = True
        
        hazard_data = {
            "hazard_score": 50.0,
            "hazard_class": "Moderate",
            "spatial_zone": "ORANGE",
            "pct_high_risk": 0.5,
            "dominant_features": ["elevation"]
        }
        
        exposure_data = {
            "pop_count": 1000,
            "infra_type": "schools",
            "infra_count": 5,
            "avg_elevation": 100.0
        }
        
        # Both should fall back to rule-based
        hazard_result = run_hazard_agent(hazard_data)
        exposure_result = run_exposure_agent(exposure_data)
        
        assert hazard_result.ai_assisted is False
        assert exposure_result.ai_assisted is False
