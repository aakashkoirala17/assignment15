"""Reliability Engineering: Retries, Circuit Breaker, and Fallback Provider Chains."""

import asyncio
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from backend.app.config import settings

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Operating normally
    OPEN = "OPEN"          # Failing, fast-rejecting calls
    HALF_OPEN = "HALF_OPEN" # Testing recovery


class CircuitBreaker:
    """Circuit Breaker pattern to protect downstream LLM providers and fail fast."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    async def can_execute(self) -> bool:
        """Check if execution is permitted through the circuit."""
        async with self._lock:
            now = time.time()
            if self.state == CircuitState.OPEN:
                if now - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("CircuitBreaker [%s] transitioned to HALF_OPEN", self.name)
                    return True
                return False
            return True

    async def record_success(self):
        """Record successful call and reset breaker if in HALF_OPEN."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("CircuitBreaker [%s] recovered to CLOSED", self.name)
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    async def record_failure(self):
        """Record failure and trip circuit to OPEN if threshold exceeded."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    "CircuitBreaker [%s] tripped to OPEN (failures=%d)",
                    self.name,
                    self.failure_count,
                )


# Map of circuit breakers by provider
_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider_name: str) -> CircuitBreaker:
    if provider_name not in _breakers:
        _breakers[provider_name] = CircuitBreaker(
            name=provider_name,
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIME,
        )
    return _breakers[provider_name]


async def execute_with_retry_and_fallback(
    primary_provider: str,
    call_fn: Callable[[str], Any],
    fallback_chain: Optional[List[str]] = None,
) -> Any:
    """
    Executes an LLM call with exponential backoff retries and fallback failover.
    If the primary provider trips the circuit breaker or fails after retries,
    the execution fails over gracefully down the fallback chain.
    """
    chain = [primary_provider]
    fallbacks = fallback_chain or settings.FALLBACK_PROVIDERS
    for fb in fallbacks:
        if fb not in chain:
            chain.append(fb)

    last_exception = None
    for provider in chain:
        breaker = get_circuit_breaker(provider)
        if not await breaker.can_execute():
            logger.warning(
                "Provider [%s] circuit is OPEN. Skipping to next fallback.", provider
            )
            continue

        # Execute with exponential backoff (retries: settings.MAX_RETRIES)
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                logger.info(
                    "Attempting execution with provider [%s] (attempt %d/%d)",
                    provider,
                    attempt,
                    settings.MAX_RETRIES,
                )
                result = await call_fn(provider)
                await breaker.record_success()
                return result
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "Provider [%s] attempt %d failed: %s", provider, attempt, exc
                )
                if attempt < settings.MAX_RETRIES:
                    # Exponential backoff with jitter
                    backoff = (settings.RETRY_BACKOFF_FACTOR ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(backoff)
                else:
                    await breaker.record_failure()

    # If all providers in chain failed, raise final error
    raise RuntimeError(
        f"All LLM providers in fallback chain {chain} failed. Last error: {last_exception}"
    )
