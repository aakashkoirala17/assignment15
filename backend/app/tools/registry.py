"""Tool Registry and Function Calling Dispatcher."""

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional
from backend.app.tools.builtins import calculate, get_current_time, weather_lookup, web_search

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Represents a callable tool with metadata and JSON schema."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters_schema: Dict[str, Any],
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters_schema = parameters_schema

    def to_openai_schema(self) -> Dict[str, Any]:
        """Format as OpenAI tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    """Central registry of tools available for LLM function calling."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
    ):
        """Decorator to register a function as a tool."""

        def decorator(fn: Callable[..., Any]):
            self._tools[name] = ToolDefinition(
                name=name,
                description=description,
                func=fn,
                parameters_schema=parameters_schema,
            )
            return fn

        return decorator

    def _register_default_tools(self):
        """Register the core built-in tools."""
        self._tools["calculate"] = ToolDefinition(
            name="calculate",
            description="Perform mathematical calculations and scientific evaluations safely. Example: 'sqrt(144) + 25 * 4'",
            func=calculate,
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '12 * 8 + 4'",
                    }
                },
                "required": ["expression"],
            },
        )

        self._tools["web_search"] = ToolDefinition(
            name="web_search",
            description="Search the web for real-time information, technical docs, and latest news.",
            func=web_search,
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or question",
                    }
                },
                "required": ["query"],
            },
        )

        self._tools["weather_lookup"] = ToolDefinition(
            name="weather_lookup",
            description="Get current weather conditions and temperature for any city.",
            func=weather_lookup,
            parameters_schema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'London', 'Tokyo', 'San Francisco'",
                    }
                },
                "required": ["city"],
            },
        )

        self._tools["get_current_time"] = ToolDefinition(
            name="get_current_time",
            description="Get current UTC date and time.",
            func=get_current_time,
            parameters_schema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name (default 'UTC')",
                    }
                },
                "required": [],
            },
        )

    def list_openai_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools formatted for OpenAI / vLLM function calling."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    async def execute(self, name: str, arguments: Any) -> str:
        """Execute a tool by name with arguments (dict or json string)."""
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found."

        if isinstance(arguments, str):
            try:
                args_dict = json.loads(arguments)
            except Exception:
                args_dict = {"query": arguments} if "query" in str(inspect.signature(tool.func)) else {}
        elif isinstance(arguments, dict):
            args_dict = arguments
        else:
            args_dict = {}

        try:
            if inspect.iscoroutinefunction(tool.func):
                result = await tool.func(**args_dict)
            else:
                result = tool.func(**args_dict)
            return str(result)
        except Exception as e:
            logger.error("Error executing tool %s: %s", name, e)
            return f"Error executing tool '{name}': {str(e)}"


tool_registry = ToolRegistry()
