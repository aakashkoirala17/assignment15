"""Tests for Reliability Engineering: Retries, Circuit Breaker, Rate Limiting, and Caching."""

import asyncio
import pytest
from backend.app.core.cache import ResponseCache
from backend.app.core.rate_limiter import TokenBucketRateLimiter
from backend.app.core.reliability import (
    CircuitBreaker,
    CircuitState,
    execute_with_retry_and_fallback,
)


@pytest.mark.asyncio
async def test_token_bucket_rate_limiter():
    """Verify rate limiter permits requests within capacity and throttles bursts."""
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst_capacity=2)
    client = "test_client_1"

    # First requests should succeed
    allowed1, headers1 = await limiter.acquire(client)
    assert allowed1 is True

    # Exhaust capacity
    for _ in range(100):
        await limiter.acquire(client)

    # Subsequent request should be throttled
    throttled, headers_throttled = await limiter.acquire(client)
    assert throttled is False
    assert "Retry-After" in headers_throttled


def test_response_caching():
    """Verify exact prompt/response caching."""
    cache = ResponseCache(ttl_seconds=10)
    req = {"prompt": "Hello world", "temperature": 0.7}
    resp = {"response": "Hi there!"}

    # Initial get is a miss
    assert cache.get(req) is None

    # Set cache
    cache.set(req, resp)

    # Second get is a hit
    cached_val = cache.get(req)
    assert cached_val == resp
    assert cache.hits >= 1


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    """Verify circuit breaker trips from CLOSED to OPEN after failure threshold."""
    breaker = CircuitBreaker("test_provider", failure_threshold=2, recovery_timeout=0.2)
    assert breaker.state == CircuitState.CLOSED
    assert await breaker.can_execute() is True

    # Record failures
    await breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED
    await breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert await breaker.can_execute() is False

    # Wait for recovery timeout
    await asyncio.sleep(0.3)
    assert await breaker.can_execute() is True
    assert breaker.state == CircuitState.HALF_OPEN

    # Recovery on success
    await breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_fallback_failover_chain():
    """Verify that when primary provider fails, failover executes secondary provider."""
    call_log = []

    async def failing_primary(prov: str):
        call_log.append(prov)
        if prov == "unstable_primary":
            raise ConnectionError("Primary provider is down")
        return f"Recovered by {prov}"

    res = await execute_with_retry_and_fallback(
        primary_provider="unstable_primary",
        call_fn=failing_primary,
        fallback_chain=["backup_provider"],
    )

    assert "Recovered by backup_provider" in res
    assert "unstable_primary" in call_log
    assert "backup_provider" in call_log
