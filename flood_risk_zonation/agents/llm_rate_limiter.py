"""
LLM Rate Limiter for Groq TPM Budget Management

Provides centralized request-level rate limiting to keep token consumption
within Groq's 8,000 TPM limit during analysis runs.

Mechanism:
  1. Token Budget Awareness: Before sending a request, check if estimated tokens
     fit within the remaining TPM budget.
  2. Request Pacing: Space out consecutive requests with small delays to allow
     TPM to regenerate over time.
  3. Graceful Degradation: If budget exhausted, use deterministic fallback.

Usage:
    rate_limiter = get_rate_limiter()
    
    # Before making an LLM request:
    if not rate_limiter.can_make_request(system_prompt, user_message):
        logger.warning("TPM budget exhausted; using fallback.")
        return fallback_result
    
    # Make the request
    result = groq_api_call(...)
    
    # After successful request:
    rate_limiter.record_request(estimated_tokens)
    
    # If 429 error:
    backoff_delay = rate_limiter.record_rate_limit_error()
    time.sleep(backoff_delay)
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class LLMRateLimiter:
    """
    Centralized LLM request rate limiter with token budget awareness.

    Attributes:
        tpm_limit: Maximum tokens per minute (Groq default: 8000).
        max_tokens_per_request: Maximum output tokens per LLM call (default: 300).
        request_history: List of (timestamp, tokens_consumed) tuples for the last 60 seconds.
        last_429_time: Timestamp of the last HTTP 429 rate-limit error.
    """

    def __init__(self, tpm_limit: int = 8000, max_tokens_per_request: int = 300):
        """
        Initialize the rate limiter.

        Args:
            tpm_limit: Tokens per minute allowed by Groq (default: 8000).
            max_tokens_per_request: Maximum output tokens per request (default: 300).
        """
        self.tpm_limit = tpm_limit
        self.max_tokens_per_request = max_tokens_per_request
        self.request_history = []  # List of (timestamp, tokens) tuples
        self.last_429_time = None

    def estimate_request_tokens(self, system_prompt: str, user_message: str) -> int:
        """
        Estimate the total token count for a request.

        Uses a conservative rule: 1 token ≈ 4 characters.
        Includes system prompt + user message + max output tokens.

        Args:
            system_prompt: System prompt (role description).
            user_message: User message (analysis data).

        Returns:
            Estimated total tokens for this request (input + output).
        """
        # Total characters
        total_chars = len(system_prompt) + len(user_message)

        # Estimate input tokens: ~1 token per 4 characters
        estimated_input_tokens = total_chars // 4

        # Add maximum output tokens (hard cap)
        estimated_total = estimated_input_tokens + self.max_tokens_per_request

        return estimated_total

    def can_make_request(self, system_prompt: str, user_message: str) -> bool:
        """
        Check if a request can be made within the TPM budget.

        Cleans up old requests (older than 60 seconds) and sums tokens consumed
        in the current minute window. If the estimated request tokens fit within
        the remaining budget, returns True; otherwise False.

        Args:
            system_prompt: System prompt for the LLM call.
            user_message: User message for the LLM call.

        Returns:
            True if the request can be made within budget; False otherwise.
        """
        estimated_tokens = self.estimate_request_tokens(system_prompt, user_message)

        # Clean up old requests (older than 60 seconds)
        current_time = time.time()
        self.request_history = [
            (ts, tokens)
            for ts, tokens in self.request_history
            if current_time - ts < 60
        ]

        # Sum tokens used in the last minute
        tokens_used_this_minute = sum(tokens for _, tokens in self.request_history)
        tokens_remaining = self.tpm_limit - tokens_used_this_minute

        # Can we fit this request?
        can_fit = estimated_tokens <= tokens_remaining

        if not can_fit:
            logger.warning(
                "LLM request would exceed TPM budget. "
                "Used this minute: %d / %d TPM. "
                "Estimated request: %d tokens. "
                "Using rule-based fallback.",
                tokens_used_this_minute,
                self.tpm_limit,
                estimated_tokens,
            )

        return can_fit

    def record_request(self, tokens_consumed: int) -> None:
        """
        Record a successful LLM request and its token consumption.

        Should be called after every successful Groq API call.

        Args:
            tokens_consumed: Number of tokens consumed by this request
                            (input + output combined).
        """
        self.request_history.append((time.time(), tokens_consumed))
        logger.debug(
            "LLM request recorded: %d tokens. "
            "Request history entries in last 60s: %d.",
            tokens_consumed,
            len(self.request_history),
        )

    def record_rate_limit_error(self) -> float:
        """
        Record a rate-limit error (HTTP 429) and return a suggested backoff delay.

        Should be called when Groq returns 429 Too Many Requests.

        Returns:
            Suggested retry delay in seconds (conservative, not aggressive).
        """
        self.last_429_time = time.time()
        # Conservative backoff: 5 seconds (allows TPM to regenerate)
        backoff_delay = 5.0
        logger.warning(
            "HTTP 429 rate-limit error recorded. "
            "Suggested backoff: %.1f seconds.",
            backoff_delay,
        )
        return backoff_delay

    def reset(self) -> None:
        """
        Reset the rate limiter for a new analysis session.

        Clears all request history and rate-limit error timestamps.
        """
        self.request_history.clear()
        self.last_429_time = None
        logger.info("LLM rate limiter reset for new session.")

    def get_status(self) -> dict:
        """
        Get the current rate limiter status (for debugging/logging).

        Returns:
            Dictionary with current TPM usage and remaining budget.
        """
        current_time = time.time()
        self.request_history = [
            (ts, tokens)
            for ts, tokens in self.request_history
            if current_time - ts < 60
        ]
        tokens_used_this_minute = sum(tokens for _, tokens in self.request_history)
        tokens_remaining = self.tpm_limit - tokens_used_this_minute

        return {
            "tpm_limit": self.tpm_limit,
            "tokens_used_this_minute": tokens_used_this_minute,
            "tokens_remaining": tokens_remaining,
            "request_count_this_minute": len(self.request_history),
            "last_429_time": self.last_429_time,
        }


# Singleton instance
_global_rate_limiter = LLMRateLimiter(tpm_limit=8000, max_tokens_per_request=300)


def get_rate_limiter() -> LLMRateLimiter:
    """
    Return the global LLM rate limiter instance.

    Returns:
        The singleton LLMRateLimiter instance.
    """
    return _global_rate_limiter
