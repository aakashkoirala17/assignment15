"""End-to-end API integration tests for FastAPI backend."""

import pytest
from httpx import ASGITransport, AsyncClient
from backend.app.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test /healthz endpoint returns 200 OK and healthy status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Test /metrics endpoint returns cache and rate limit telemetry."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "cache" in data
        assert "rate_limiting" in data


@pytest.mark.asyncio
async def test_chat_endpoint():
    """Test POST /api/v1/chat completion."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "messages": [{"role": "user", "content": "Ping"}],
            "temperature": 0.5,
            "provider": "mock",
        }
        resp = await client.post("/api/v1/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0


@pytest.mark.asyncio
async def test_structured_output_endpoint():
    """Test POST /api/v1/chat/structured returns valid JSON schema."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "text": "Product launch went exceptionally well. Sales increased by 30%.",
            "schema_type": "analysis_report",
            "provider": "mock",
        }
        resp = await client.post("/api/v1/chat/structured", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "summary" in data["data"]
        assert "sentiment" in data["data"]


@pytest.mark.asyncio
async def test_tools_list_endpoint():
    """Test GET /api/v1/tools/list returns registered tools."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/tools/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tools"] >= 4


@pytest.mark.asyncio
async def test_rag_upload_and_query_endpoints():
    """Test document upload and vector query flow."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        files = {"file": ("test.txt", b"Antigravity AI Assistant architecture uses ChromaDB and FastAPI.")}
        upload_resp = await client.post(
            "/api/v1/rag/upload",
            files=files,
            data={"collection_name": "api_test_collection"},
        )
        assert upload_resp.status_code == 200
        assert upload_resp.json()["chunks_created"] >= 1

        query_payload = {
            "query": "What architecture is used?",
            "collection_name": "api_test_collection",
            "generate_answer": True,
            "provider": "mock",
        }
        query_resp = await client.post("/api/v1/rag/query", json=query_payload)
        assert query_resp.status_code == 200
        assert len(query_resp.json()["retrieved_chunks"]) >= 1
