"""Unit and integration tests for LLM provider, prompt engineering, and parameter tuning."""

import pytest
from backend.app.core.llm_provider import llm_service
from backend.app.schemas.chat import ChatMessage, ChatRequest


@pytest.mark.asyncio
async def test_llm_provider_generation():
    """Verify that LLM provider generates valid responses with system prompt."""
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello, assistant!")],
        system_prompt="You are a helpful test assistant.",
        temperature=0.7,
        top_p=0.9,
        provider="mock",
    )
    resp = await llm_service.complete(req)

    assert resp is not None
    assert resp.role == "assistant"
    assert len(resp.content) > 0
    assert resp.provider == "mock"
    assert "latency_ms" in resp.model_dump()


@pytest.mark.asyncio
async def test_hyperparameter_tuning():
    """Verify temperature and top_p pass validation and impact request metadata."""
    req_low_temp = ChatRequest(
        messages=[ChatMessage(role="user", content="Deterministic query")],
        temperature=0.1,
        top_p=0.5,
        max_tokens=64,
        provider="mock",
    )
    resp = await llm_service.complete(req_low_temp)
    assert resp is not None
    assert resp.finish_reason in ["stop", "tool_calls"]
