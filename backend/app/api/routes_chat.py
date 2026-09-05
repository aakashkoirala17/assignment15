"""Chat and structured output API endpoints."""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from backend.app.core.llm_provider import llm_service, repair_and_extract_json
from backend.app.core.rate_limiter import rate_limit_dependency
from backend.app.schemas.chat import (
    AnalysisReport,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    MeetingNotesExtraction,
    StructuredOutputRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Standard Chat Completion with Tool Calling & Fallback",
    dependencies=[Depends(rate_limit_dependency)],
)
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """
    Executes a chat completion query against configured LLM provider.
    Supports system prompts, temperature/top_p parameter tuning, tool calling, and caching.
    """
    try:
        response = await llm_service.complete(request, auto_execute_tools=True)
        return response
    except Exception as e:
        logger.error("Chat completion error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}",
        )


@router.post(
    "/stream",
    summary="Stream Chat Completion using Server-Sent Events (SSE)",
    dependencies=[Depends(rate_limit_dependency)],
)
async def chat_stream(request: ChatRequest):
    """Streams chat completion response tokens as Server-Sent Events."""
    request.stream = True
    provider = llm_service.get_provider(request.provider)

    async def event_generator():
        try:
            async for token in provider.generate_stream(request):
                payload = json.dumps({"token": token})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_payload = json.dumps({"error": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post(
    "/structured",
    summary="Generate Valid JSON Response Adhering to Strict Schema",
    dependencies=[Depends(rate_limit_dependency)],
)
async def structured_output(request: StructuredOutputRequest):
    """
    Ensure the model generates valid JSON responses adhering
    to Pydantic schemas (e.g. AnalysisReport, MeetingNotesExtraction).
    """
    # Select target schema
    if request.schema_type == "analysis_report":
        schema_model = AnalysisReport
    elif request.schema_type == "meeting_notes":
        schema_model = MeetingNotesExtraction
    else:
        schema_model = None

    schema_repr = (
        json.dumps(schema_model.model_json_schema(), indent=2)
        if schema_model
        else json.dumps(request.custom_json_schema or {}, indent=2)
    )

    system_instruction = (
        "You are an expert data analysis engine. You MUST respond with ONLY a valid JSON object. "
        "Do NOT include any explanatory text before or after the JSON. "
        f"Your output MUST conform strictly to this JSON schema:\n{schema_repr}"
    )

    chat_req = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=f"Please analyze the following text and return structured JSON:\n\n{request.text}",
            )
        ],
        system_prompt=system_instruction,
        provider=request.provider,
        temperature=request.temperature,
        force_json=True,
        tools_enabled=False,
    )

    try:
        response = await llm_service.complete(chat_req)
        parsed_json = repair_and_extract_json(response.content)

        # Validate with Pydantic schema if applicable
        validated_data = None
        if schema_model:
            try:
                validated_obj = schema_model.model_validate(parsed_json)
                validated_data = validated_obj.model_dump()
            except Exception as val_err:
                logger.warning("Pydantic validation warning: %s; returning repaired JSON", val_err)
                validated_data = parsed_json
        else:
            validated_data = parsed_json

        return {
            "status": "success",
            "schema_type": request.schema_type,
            "data": validated_data,
            "provider": response.provider,
            "latency_ms": response.latency_ms,
        }
    except Exception as e:
        logger.error("Structured output error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Structured output generation failed: {str(e)}",
        )
