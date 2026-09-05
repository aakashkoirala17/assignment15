"""Chat and structured output schemas."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Function/Tool call specification."""

    id: str
    type: str = "function"
    function: Dict[str, Any]  # name, arguments (JSON string or dict)


class ChatMessage(BaseModel):
    """Message object adhering to standard chat formatting."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Standard request payload for chat completion."""

    messages: List[ChatMessage]
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider: 'gemini', 'openai', 'anthropic', 'vllm', or 'mock'",
    )
    model: Optional[str] = Field(
        default=None, description="Specific model identifier"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt to prepend"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    max_tokens: int = Field(
        default=1024, ge=1, le=8192, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Stream response via SSE")
    tools_enabled: bool = Field(
        default=True, description="Enable function/tool calling"
    )
    force_json: bool = Field(
        default=False, description="Enforce JSON structured response"
    )


class ChatResponse(BaseModel):
    """Standard response payload from LLM."""

    id: str
    role: str = "assistant"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    provider: str
    model: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    latency_ms: float = 0.0
    cached: bool = False


# Structured Output Specifications
class KeyInsight(BaseModel):
    title: str = Field(description="Insight headline")
    description: str = Field(description="Detailed explanation")
    impact_level: Literal["low", "medium", "high", "critical"] = Field(
        default="medium"
    )


class AnalysisReport(BaseModel):
    """Strict JSON schema for structured analysis report."""

    summary: str = Field(description="Executive summary of the analysis")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment"
    )
    key_insights: List[KeyInsight] = Field(
        description="List of key findings or insights"
    )
    recommended_actions: List[str] = Field(
        description="Actionable recommendations"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence rating between 0 and 1"
    )


class ActionItem(BaseModel):
    task: str
    assignee: Optional[str] = "Unassigned"
    priority: Literal["low", "medium", "high"] = "medium"


class MeetingNotesExtraction(BaseModel):
    """Strict JSON schema for structured meeting notes extraction."""

    topic: str
    attendees: List[str] = Field(default_factory=list)
    key_decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)


class StructuredOutputRequest(BaseModel):
    """Request for guaranteed valid JSON schema outputs."""

    text: str = Field(description="Source text or user instruction")
    schema_type: Literal["analysis_report", "meeting_notes", "custom"] = Field(
        default="analysis_report"
    )
    custom_json_schema: Optional[Dict[str, Any]] = None
    provider: Optional[str] = None
    temperature: float = 0.2
