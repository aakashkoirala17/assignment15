"""Tools API endpoints for external tool inspection and execution."""

import logging
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.tools.registry import tool_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["External Tools"])


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


@router.get(
    "/list",
    summary="List All Registered Tools and Function Schemas",
)
async def list_tools():
    """Returns OpenAI/JSON-schema definitions of all registered tools."""
    tools = tool_registry.list_openai_tools()
    return {"total_tools": len(tools), "tools": tools}


@router.post(
    "/execute",
    summary="Directly Execute an External Tool",
)
async def execute_tool(request: ToolExecutionRequest):
    """Executes a registered tool directly with provided arguments."""
    tool = tool_registry.get_tool(request.tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{request.tool_name}' not found.",
        )
    output = await tool_registry.execute(request.tool_name, request.arguments)
    return {
        "tool_name": request.tool_name,
        "arguments": request.arguments,
        "output": output,
    }
