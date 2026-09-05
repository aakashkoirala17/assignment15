"""Performance Engineering: Concurrency, Latency, and Throughput Benchmarking."""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")


async def _send_request(
    client: httpx.AsyncClient, base_url: str, request_id: int
) -> float:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": f"Explain the benefits of RAG in modern applied AI systems. Request #{request_id}",
            }
        ],
        "temperature": 0.5,
        "max_tokens": 128,
        "tools_enabled": False,
        "provider": "mock",
    }
    t0 = time.perf_counter()
    resp = await client.post(f"{base_url}/api/v1/chat", json=payload, timeout=30.0)
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000
    return latency_ms


async def run_concurrent_benchmark(
    base_url: str = "http://localhost:8000",
    total_requests: int = 50,
    concurrency: int = 10,
    in_process: bool = False,
):
    """Executes asynchronous concurrent requests against the AI Assistant API."""
    logger.info(
        "Starting concurrent benchmark: %d requests at concurrency %d (in_process=%s)",
        total_requests,
        concurrency,
        in_process,
    )

    if in_process or base_url == "in-process":
        from backend.app.main import app
        transport = httpx.ASGITransport(app=app)
        client_ctx = httpx.AsyncClient(transport=transport, base_url="http://test")
    else:
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        client_ctx = httpx.AsyncClient(limits=limits)

    async with client_ctx as client:
        url_prefix = "" if in_process or base_url == "in-process" else base_url
        sem = asyncio.Semaphore(concurrency)

        async def worker(req_id: int) -> float:
            async with sem:
                return await _send_request(client, url_prefix, req_id)

        start_time = time.perf_counter()
        tasks = [worker(i) for i in range(total_requests)]
        latencies: List[float] = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time

    qps = total_requests / total_time
    logger.info("=== Benchmarking Results ===")
    logger.info("Total Requests: %d", total_requests)
    logger.info("Concurrency: %d", concurrency)
    logger.info("Elapsed Time: %.2f seconds", total_time)
    logger.info("Throughput (QPS): %.2f req/s", qps)
    logger.info("Mean Latency: %.2f ms", np.mean(latencies))
    logger.info("P50 Latency: %.2f ms", np.percentile(latencies, 50))
    logger.info("P95 Latency: %.2f ms", np.percentile(latencies, 95))
    logger.info("P99 Latency: %.2f ms", np.percentile(latencies, 99))

    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "elapsed_seconds": round(total_time, 2),
        "throughput_qps": round(qps, 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent load test")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--in-process", action="store_true", help="Run benchmark in-process using FastAPI ASGI transport")
    args = parser.parse_args()

    asyncio.run(
        run_concurrent_benchmark(
            base_url=args.url,
            total_requests=args.requests,
            concurrency=args.concurrency,
            in_process=getattr(args, "in_process", False),
        )
    )
