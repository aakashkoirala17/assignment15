"""Token Bucket Rate Limiter for Performance & Reliability."""

import asyncio
import time
from typing import Dict, Tuple
from fastapi import HTTPException, Request, status
from backend.app.config import settings


class TokenBucketRateLimiter:
    """In-memory thread-safe Token Bucket Rate Limiter."""

    def __init__(self, requests_per_minute: int = 60, burst_capacity: int = 10):
        self.rate = requests_per_minute / 60.0  # Tokens per second
        self.capacity = float(burst_capacity + requests_per_minute)
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, client_id: str = "global") -> Tuple[bool, Dict[str, str]]:
        """Attempt to consume 1 token for client_id. Returns (allowed, headers)."""
        async with self._lock:
            now = time.time()
            if client_id not in self.tokens:
                self.tokens[client_id] = self.capacity
                self.last_update[client_id] = now

            # Replenish tokens based on elapsed time
            elapsed = now - self.last_update[client_id]
            self.tokens[client_id] = min(
                self.capacity, self.tokens[client_id] + elapsed * self.rate
            )
            self.last_update[client_id] = now

            headers = {
                "X-RateLimit-Limit": str(int(self.capacity)),
                "X-RateLimit-Remaining": str(max(0, int(self.tokens[client_id]))),
                "X-RateLimit-Reset": str(int(now + 60)),
            }

            if self.tokens[client_id] >= 1.0:
                self.tokens[client_id] -= 1.0
                headers["X-RateLimit-Remaining"] = str(int(self.tokens[client_id]))
                return True, headers

            retry_after = (1.0 - self.tokens[client_id]) / self.rate
            headers["Retry-After"] = str(int(retry_after) + 1)
            return False, headers


rate_limiter = TokenBucketRateLimiter(
    requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    burst_capacity=settings.RATE_LIMIT_BURST,
)


async def rate_limit_dependency(request: Request):
    """FastAPI route dependency to enforce rate limiting."""
    client_ip = request.client.host if request.client else "unknown_client"
    allowed, headers = await rate_limiter.acquire(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please throttle your requests.",
            headers=headers,
        )
