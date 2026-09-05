"""Health check and telemetry endpoints."""

import os
import time
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.core.cache import cache
from backend.app.rag.vector_store import vector_store

router = APIRouter(tags=["Health & Monitoring"])

START_TIME = time.time()


@router.get("/healthz", summary="Liveness and Readiness Probe")
async def health_check():
    """Health check endpoint for Docker and Kubernetes orchestration."""
    uptime_seconds = time.time() - START_TIME
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "uptime_seconds": round(uptime_seconds, 1),
        "primary_provider": settings.LLM_PROVIDER,
        "fallback_providers": settings.FALLBACK_PROVIDERS,
        "environment": "development" if settings.DEBUG else "production",
    }


@router.get("/metrics", summary="Performance and Telemetry Metrics")
async def get_metrics():
    """Returns runtime telemetry, cache hit ratios, and vector store stats."""
    return {
        "cache": cache.get_stats(),
        "vector_collections": vector_store.list_collections(),
        "rate_limiting": {
            "requests_per_minute": settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
            "burst_capacity": settings.RATE_LIMIT_BURST,
        },
    }
