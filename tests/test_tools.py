"""Tests for Tool Calling, function schemas, and execution."""

import json
import pytest
from backend.app.tools.builtins import calculate, get_current_time, weather_lookup, web_search
from backend.app.tools.registry import tool_registry


def test_calculate_tool():
    """Verify safe AST calculation tool."""
    assert calculate("12 * 8 + 4") == "100"
    assert calculate("sqrt(144)") == "12.0"
    assert calculate("2 ** 5") == "32"
    # Unsafe code should be blocked safely
    res = calculate("__import__('os').system('ls')")
    assert "error" in res.lower() or "not permitted" in res.lower()


def test_weather_lookup_tool():
    """Verify weather lookup tool."""
    res = weather_lookup("London")
    assert "London" in res
    assert "Condition:" in res


def test_get_current_time_tool():
    """Verify current time tool."""
    res = get_current_time()
    assert "Current date and time:" in res
    assert "UTC" in res


def test_web_search_tool():
    """Verify web search tool."""
    res = web_search("vLLM architecture")
    assert "vllm" in res.lower() or "search result" in res.lower()


@pytest.mark.asyncio
async def test_tool_registry_execution():
    """Verify asynchronous execution via tool registry dispatcher."""
    res = await tool_registry.execute("calculate", {"expression": "50 / 2"})
    assert res == "25.0"

    schemas = tool_registry.list_openai_tools()
    tool_names = [t["function"]["name"] for t in schemas]
    assert "calculate" in tool_names
    assert "weather_lookup" in tool_names
    assert "web_search" in tool_names
    assert "get_current_time" in tool_names
