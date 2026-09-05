"""Prompt and Response Caching Engine (In-Memory LRU + Optional Redis)."""

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional

from backend.app.config import settings

logger = logging.getLogger(__name__)


class ResponseCache:
    """High-performance TTL response cache to reduce latency and API cost."""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0
        self._redis_client = None

        if settings.REDIS_URL:
            try:
                import redis

                self._redis_client = redis.Redis.from_url(
                    settings.REDIS_URL, decode_responses=True
                )
                self._redis_client.ping()
                logger.info("Connected to Redis cache at %s", settings.REDIS_URL)
            except Exception as e:
                logger.warning(
                    "Redis connection failed (%s); falling back to in-memory cache", e
                )
                self._redis_client = None

    def _generate_key(self, request_data: Dict[str, Any]) -> str:
        """Create a deterministic MD5 hash key from normalized request parameters."""
        normalized = json.dumps(request_data, sort_keys=True, default=str)
        return "cache:llm:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if not expired."""
        if not settings.ENABLE_CACHE:
            return None

        key = self._generate_key(request_data)

        # 1. Try Redis
        if self._redis_client:
            try:
                val = self._redis_client.get(key)
                if val:
                    self.hits += 1
                    return json.loads(val)
            except Exception as e:
                logger.debug("Redis get error: %s", e)

        # 2. In-memory cache fallback
        entry = self._memory_cache.get(key)
        if entry:
            if time.time() < entry["expires_at"]:
                self.hits += 1
                return entry["data"]
            else:
                # Expired
                del self._memory_cache[key]

        self.misses += 1
        return None

    def set(
        self, request_data: Dict[str, Any], response_data: Dict[str, Any], ttl: Optional[int] = None
    ) -> None:
        """Store response in cache with TTL."""
        if not settings.ENABLE_CACHE:
            return

        key = self._generate_key(request_data)
        ttl = ttl or self.ttl

        # 1. Store in Redis
        if self._redis_client:
            try:
                self._redis_client.setex(
                    key, ttl, json.dumps(response_data, default=str)
                )
            except Exception as e:
                logger.debug("Redis set error: %s", e)

        # 2. Store in memory
        self._memory_cache[key] = {
            "data": response_data,
            "expires_at": time.time() + ttl,
        }

    def clear(self) -> None:
        """Flush cache."""
        self._memory_cache.clear()
        if self._redis_client:
            try:
                self._redis_client.flushdb()
            except Exception as e:
                logger.debug("Redis clear error: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_ratio = (self.hits / total) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_ratio": round(hit_ratio, 3),
            "in_memory_items": len(self._memory_cache),
            "backend": "redis" if self._redis_client else "in-memory",
        }


# Global cache instance
cache = ResponseCache(ttl_seconds=settings.CACHE_TTL_SECONDS)
