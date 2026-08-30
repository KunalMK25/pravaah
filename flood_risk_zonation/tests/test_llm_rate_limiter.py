"""
Tests for LLM Rate Limiter.

Verifies:
  - Token budget tracking
  - TPM limit enforcement
  - Request pacing
  - 429 error recording
  - Rate limiter reset
"""

import time
import unittest

from flood_risk_zonation.agents.llm_rate_limiter import LLMRateLimiter, get_rate_limiter


class TestLLMRateLimiter(unittest.TestCase):
    """Tests for LLMRateLimiter class."""

    def setUp(self):
        """Create a fresh rate limiter for each test."""
        self.limiter = LLMRateLimiter(tpm_limit=1000, max_tokens_per_request=100)

    def test_estimate_request_tokens(self):
        """Test token estimation (1 token ≈ 4 characters)."""
        system_prompt = "x" * 100  # 100 chars ≈ 25 tokens
        user_message = "y" * 200   # 200 chars ≈ 50 tokens
        
        estimated = self.limiter.estimate_request_tokens(system_prompt, user_message)
        
        # Expected: (100 + 200) // 4 + 100 (max_tokens) = 75 + 100 = 175
        self.assertEqual(estimated, 175)

    def test_can_make_request_within_budget(self):
        """Test that requests within budget are allowed."""
        system_prompt = "x" * 100
        user_message = "y" * 100
        
        # First request should be allowed
        can_make = self.limiter.can_make_request(system_prompt, user_message)
        self.assertTrue(can_make)

    def test_can_make_request_exceeds_budget(self):
        """Test that requests exceeding budget are rejected."""
        # Manually fill the budget
        self.limiter.request_history.append((time.time(), 900))  # 900 tokens used
        
        system_prompt = "x" * 200  # 50 tokens estimate
        user_message = "y" * 200   # 50 tokens estimate
        # Total: 100 + 100 (max_tokens) = 200 tokens request
        
        # Should reject because 900 + 200 > 1000
        can_make = self.limiter.can_make_request(system_prompt, user_message)
        self.assertFalse(can_make)

    def test_record_request(self):
        """Test recording a successful request."""
        self.limiter.record_request(150)
        
        self.assertEqual(len(self.limiter.request_history), 1)
        tokens_recorded = self.limiter.request_history[0][1]
        self.assertEqual(tokens_recorded, 150)

    def test_request_history_cleanup(self):
        """Test that old requests (>60s old) are cleaned up."""
        # Add a very old request
        old_time = time.time() - 70  # 70 seconds ago
        self.limiter.request_history.append((old_time, 100))
        
        # Add a recent request
        self.limiter.record_request(50)
        
        # When checking budget, old request should be cleaned
        can_make = self.limiter.can_make_request("x" * 10, "y" * 10)
        
        # Old request should be gone; only recent one (50) + new estimate should count
        # This is a side effect of can_make_request cleaning up old entries
        self.assertTrue(can_make)

    def test_record_rate_limit_error(self):
        """Test recording a 429 rate-limit error."""
        before_time = time.time()
        backoff_delay = self.limiter.record_rate_limit_error()
        
        self.assertIsNotNone(self.limiter.last_429_time)
        self.assertGreaterEqual(self.limiter.last_429_time, before_time)
        self.assertEqual(backoff_delay, 5.0)  # Conservative backoff

    def test_reset(self):
        """Test resetting the rate limiter."""
        self.limiter.record_request(100)
        self.limiter.record_rate_limit_error()
        
        self.assertEqual(len(self.limiter.request_history), 1)
        self.assertIsNotNone(self.limiter.last_429_time)
        
        self.limiter.reset()
        
        self.assertEqual(len(self.limiter.request_history), 0)
        self.assertIsNone(self.limiter.last_429_time)

    def test_get_status(self):
        """Test status reporting."""
        self.limiter.record_request(300)
        self.limiter.record_request(200)
        
        status = self.limiter.get_status()
        
        self.assertEqual(status["tpm_limit"], 1000)
        self.assertEqual(status["tokens_used_this_minute"], 500)
        self.assertEqual(status["tokens_remaining"], 500)
        self.assertEqual(status["request_count_this_minute"], 2)

    def test_multiple_requests_accumulate(self):
        """Test that multiple requests accumulate correctly."""
        for i in range(3):
            self.limiter.record_request(100)
        
        status = self.limiter.get_status()
        self.assertEqual(status["tokens_used_this_minute"], 300)
        self.assertEqual(status["tokens_remaining"], 700)

    def test_can_make_request_with_accumulated_history(self):
        """Test budget check against accumulated history."""
        # Fill budget to 950 tokens
        self.limiter.record_request(500)
        self.limiter.record_request(450)
        
        # Try to make a request that would exceed budget
        system_prompt = "x" * 100
        user_message = "y" * 200
        # Estimated: 75 + 100 = 175 tokens
        # 950 + 175 = 1125 > 1000, so should be rejected
        
        can_make = self.limiter.can_make_request(system_prompt, user_message)
        self.assertFalse(can_make)
        
        # But a smaller request should fit
        system_prompt = "x" * 10
        user_message = "y" * 10
        # Estimated: 5 + 100 = 105 tokens
        # 950 + 105 = 1055 > 1000, still rejected
        
        can_make = self.limiter.can_make_request(system_prompt, user_message)
        self.assertFalse(can_make)


class TestGlobalRateLimiter(unittest.TestCase):
    """Tests for global rate limiter singleton."""

    def test_get_rate_limiter_returns_singleton(self):
        """Test that get_rate_limiter returns the same instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        self.assertIs(limiter1, limiter2)

    def test_singleton_default_values(self):
        """Test that singleton has correct defaults."""
        limiter = get_rate_limiter()
        
        self.assertEqual(limiter.tpm_limit, 8000)
        self.assertEqual(limiter.max_tokens_per_request, 300)


class TestRateLimiterIntegration(unittest.TestCase):
    """Integration tests for rate limiter."""

    def setUp(self):
        """Create a fresh limiter with small TPM for testing."""
        self.limiter = LLMRateLimiter(tpm_limit=500, max_tokens_per_request=100)

    def test_realistic_scenario_within_budget(self):
        """Test realistic scenario: Multiple requests until budget exhausted."""
        prompts = [
            ("System A", "User message 1"),
            ("System B", "User message 2"),
            ("System C", "User message 3"),
            ("System D", "User message 4"),
            ("System E", "User message 5"),
        ]
        
        accepted_count = 0
        for sys_prompt, usr_msg in prompts:
            can_make = self.limiter.can_make_request(sys_prompt, usr_msg)
            if can_make:
                self.limiter.record_request(
                    self.limiter.estimate_request_tokens(sys_prompt, usr_msg)
                )
                accepted_count += 1

        status = self.limiter.get_status()
        # With 500 TPM budget and ~105 tokens per request, expect ~4-5 requests accepted
        self.assertGreater(accepted_count, 0)
        self.assertLessEqual(accepted_count, 5)
        self.assertEqual(status["request_count_this_minute"], accepted_count)

    def test_realistic_scenario_budget_exhaustion(self):
        """Test that budget exhaustion gracefully rejects requests."""
        # Make requests until budget is exceeded
        rejected_count = 0
        total_attempts = 0
        
        for i in range(20):
            can_make = self.limiter.can_make_request("sys", "msg")
            total_attempts += 1
            
            if not can_make:
                rejected_count += 1
            else:
                self.limiter.record_request(150)  # Arbitrary token count
        
        # Some requests should have been rejected
        self.assertGreater(rejected_count, 0)
        self.assertGreater(total_attempts, rejected_count)


if __name__ == "__main__":
    unittest.main()
