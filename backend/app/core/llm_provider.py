"""Unified Multi-Provider LLM Engine with Prompt Engineering, Tool Calling & Structured Outputs."""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.app.config import settings
from backend.app.core.cache import cache
from backend.app.core.reliability import execute_with_retry_and_fallback
from backend.app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ToolCall,
)
from backend.app.tools.registry import tool_registry

logger = logging.getLogger(__name__)


def repair_and_extract_json(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor and repair engine.
    Handles Markdown ```json code blocks, trailing commas, and partial structures.
    """
    cleaned = raw_text.strip()

    # 1. Extract from markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # 2. Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Find outermost curly braces
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = cleaned[start_idx : end_idx + 1]
        try:
            return json.loads(json_candidate)
        except Exception:
            # Clean trailing commas: ,} -> } and ,] -> ]
            fixed = re.sub(r",\s*([\}\]])", r"\1", json_candidate)
            try:
                return json.loads(fixed)
            except Exception:
                pass

    # 4. Fallback structured dict
    return {
        "summary": raw_text[:200],
        "sentiment": "neutral",
        "key_insights": [{"title": "Extracted Output", "description": raw_text, "impact_level": "medium"}],
        "recommended_actions": ["Review raw model output"],
        "confidence_score": 0.85,
    }


class BaseLLMProvider:
    """Abstract interface for LLM provider implementations."""

    name: str

    async def generate(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def generate_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError


class MockLLMProvider(BaseLLMProvider):
    """
    High-fidelity offline provider for testing, fallback, and zero-cost demonstration.
    Accurately emulates prompt tuning, tool calling, and Pydantic structured output.
    """

    name = "mock"

    async def generate(self, request: ChatRequest) -> ChatResponse:
        start_time = time.time()
        last_msg = request.messages[-1].content if request.messages else ""
        system_prompt = request.system_prompt or "You are an AI Assistant."

        # 1. Check for Tool Calling triggers if tools enabled
        tool_calls: Optional[List[ToolCall]] = None
        response_content = ""

        if request.tools_enabled:
            # Check for calculator pattern
            calc_match = re.search(r"calculate|math|\d+\s*[\+\-\*\/\^]\s*\d+|sqrt\(", last_msg, re.IGNORECASE)
            # Check for weather pattern
            weather_match = re.search(r"weather in ([a-zA-Z\s]+)|what.*weather.*for ([a-zA-Z\s]+)", last_msg, re.IGNORECASE)
            # Check for search pattern
            search_match = re.search(r"search for (.+)|who is (.+)|what is (.+)", last_msg, re.IGNORECASE)

            if calc_match:
                expr = re.sub(r"[^\d\+\-\*\/\^\(\)\.\s]|calculate|what is", "", last_msg).strip()
                if expr:
                    call_id = f"call_{uuid.uuid4().hex[:8]}"
                    tool_calls = [
                        ToolCall(
                            id=call_id,
                            function={"name": "calculate", "arguments": json.dumps({"expression": expr})},
                        )
                    ]
            elif weather_match:
                city = (weather_match.group(1) or weather_match.group(2) or "San Francisco").strip()
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_calls = [
                    ToolCall(
                        id=call_id,
                        function={"name": "weather_lookup", "arguments": json.dumps({"city": city})},
                    )
                ]
            elif "time" in last_msg.lower() or "date" in last_msg.lower():
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_calls = [
                    ToolCall(
                        id=call_id,
                        function={"name": "get_current_time", "arguments": "{}"},
                    )
                ]
            elif search_match and len(last_msg) > 10:
                q = (search_match.group(1) or search_match.group(2) or search_match.group(3) or last_msg).strip()
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                tool_calls = [
                    ToolCall(
                        id=call_id,
                        function={"name": "web_search", "arguments": json.dumps({"query": q})},
                    )
                ]

        # 2. Check for Structured Output requirement
        if request.force_json:
            structured_data = {
                "summary": f"Structured evaluation of: '{last_msg[:80]}...'",
                "sentiment": "positive" if "great" in last_msg.lower() or "good" in last_msg.lower() else "neutral",
                "key_insights": [
                    {
                        "title": "System Prompt & Parameter Alignment",
                        "description": f"Generated with temperature={request.temperature}, top_p={request.top_p}.",
                        "impact_level": "high",
                    },
                    {
                        "title": "Robust Architecture Validation",
                        "description": "Output adheres strictly to Pydantic JSON schema specifications.",
                        "impact_level": "medium",
                    },
                ],
                "recommended_actions": [
                    "Proceed with production deployment.",
                    "Verify latency and cache hit ratios under load.",
                ],
                "confidence_score": 0.96,
            }
            response_content = json.dumps(structured_data, indent=2)
        elif not tool_calls:
            # Standard conversational response
            response_content = (
                f"Assistant Response [Prompt: '{system_prompt[:50]}...', Temp: {request.temperature}]:\n\n"
                f"I have received your request: \"{last_msg}\".\n\n"
                "The Applied AI & Systems Engineering pipeline is operational. RAG vector embeddings, "
                "concurrency handling, prompt caching, and fault tolerance mechanisms are active."
            )

        latency = (time.time() - start_time) * 1000
        return ChatResponse(
            id=f"mock-{uuid.uuid4().hex[:8]}",
            role="assistant",
            content=response_content,
            tool_calls=tool_calls,
            provider="mock",
            model=request.model or "mock-assistant-v1",
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={"prompt_tokens": len(last_msg) // 4 + 10, "completion_tokens": len(response_content) // 4, "total_tokens": (len(last_msg) + len(response_content)) // 4},
            latency_ms=round(latency, 2),
        )

    async def generate_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        resp = await self.generate(request)
        words = resp.content.split(" ")
        for word in words:
            await asyncio.sleep(0.02)
            yield word + " "


class VLLMProvider(BaseLLMProvider):
    """Local OpenAI-compatible vLLM serving engine (e.g. Llama 3 / Mistral)."""

    name = "vllm"

    def __init__(self, base_url: str = settings.VLLM_BASE_URL):
        self.base_url = base_url
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY")

    async def generate(self, request: ChatRequest) -> ChatResponse:
        start_time = time.time()
        messages_payload = []
        if request.system_prompt:
            messages_payload.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            m = {"role": msg.role, "content": msg.content or ""}
            if msg.name:
                m["name"] = msg.name
            messages_payload.append(m)

        tools = tool_registry.list_openai_tools() if request.tools_enabled else None
        model_name = request.model or settings.VLLM_MODEL_NAME

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages_payload,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if request.force_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type="function",
                    function={
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                )
                for tc in choice.message.tool_calls
            ]

        latency = (time.time() - start_time) * 1000
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }

        return ChatResponse(
            id=response.id,
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            provider="vllm",
            model=model_name,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            latency_ms=round(latency, 2),
        )

    async def generate_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        messages_payload = []
        if request.system_prompt:
            messages_payload.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            messages_payload.append({"role": msg.role, "content": msg.content or ""})

        stream = await self.client.chat.completions.create(
            model=request.model or settings.VLLM_MODEL_NAME,
            messages=messages_payload,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GeminiProvider(BaseLLMProvider):
    """Google Gemini Provider using google-generativeai SDK."""

    name = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        if self.api_key:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        import google.generativeai as genai

        start_time = time.time()
        model_name = request.model or "gemini-1.5-flash"
        system_instruction = request.system_prompt or "You are an intelligent AI assistant."

        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_tokens,
            response_mime_type="application/json" if request.force_json else "text/plain",
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=generation_config,
        )

        # Convert messages to Gemini format
        history = []
        for msg in request.messages[:-1]:
            role = "user" if msg.role == "user" else "model"
            history.append({"role": role, "parts": [msg.content or ""]})

        chat = model.start_chat(history=history)
        last_prompt = request.messages[-1].content if request.messages else ""

        # Run in thread pool to prevent blocking asyncio loop
        response = await asyncio.to_thread(chat.send_message, last_prompt)
        content = response.text or ""
        latency = (time.time() - start_time) * 1000

        return ChatResponse(
            id=f"gemini-{uuid.uuid4().hex[:8]}",
            role="assistant",
            content=content,
            provider="gemini",
            model=model_name,
            latency_ms=round(latency, 2),
        )

    async def generate_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        resp = await self.generate(request)
        for chunk in resp.content.split(" "):
            await asyncio.sleep(0.02)
            yield chunk + " "


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Provider using official async client."""

    name = "openai"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        if self.api_key:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        start_time = time.time()
        messages_payload = []
        if request.system_prompt:
            messages_payload.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            messages_payload.append({"role": msg.role, "content": msg.content or ""})

        tools = tool_registry.list_openai_tools() if request.tools_enabled else None
        model_name = request.model or "gpt-4o"

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages_payload,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if request.force_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function={"name": tc.function.name, "arguments": tc.function.arguments},
                )
                for tc in choice.message.tool_calls
            ]

        latency = (time.time() - start_time) * 1000
        return ChatResponse(
            id=response.id,
            role="assistant",
            content=choice.message.content or "",
            tool_calls=tool_calls,
            provider="openai",
            model=model_name,
            latency_ms=round(latency, 2),
        )

    async def generate_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        resp = await self.generate(request)
        for chunk in resp.content.split(" "):
            await asyncio.sleep(0.02)
            yield chunk + " "


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider using AsyncAnthropic client."""

    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.client = None
        if self.api_key:
            try:
                import anthropic

                self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except Exception as e:
                logger.warning("Anthropic client initialization warning: %s", e)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if not self.api_key or not self.client:
            raise ValueError("ANTHROPIC_API_KEY is not configured or client failed to initialize.")

        start_time = time.time()
        model_name = request.model or "claude-3-5-sonnet-20240620"
        system_instruction = request.system_prompt or "You are an intelligent AI assistant."

        messages_payload = []
        for msg in request.messages:
            role = "user" if msg.role == "user" else "assistant"
            messages_payload.append({"role": role, "content": msg.content or ""})

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "system": system_instruction,
            "messages": messages_payload,
            "temperature": min(1.0, max(0.0, request.temperature)),
            "max_tokens": request.max_tokens,
        }

        response = await self.client.messages.create(**kwargs)
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        latency = (time.time() - start_time) * 1000
        return ChatResponse(
            id=f"anthropic-{response.id}",
            role="assistant",
            content=content,
            provider="anthropic",
            model=model_name,
            latency_ms=round(latency, 2),
        )

    async def generate_stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        resp = await self.generate(request)
        for chunk in resp.content.split(" "):
            await asyncio.sleep(0.02)
            yield chunk + " "


class LLMService:
    """Master LLM orchestration service managing caching, retries, and provider failover."""

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {
            "mock": MockLLMProvider(),
            "vllm": VLLMProvider(),
        }
        if settings.GEMINI_API_KEY:
            try:
                self.providers["gemini"] = GeminiProvider()
            except Exception as e:
                logger.warning("Failed to initialize GeminiProvider: %s", e)
        if settings.OPENAI_API_KEY:
            try:
                self.providers["openai"] = OpenAIProvider()
            except Exception as e:
                logger.warning("Failed to initialize OpenAIProvider: %s", e)
        if settings.ANTHROPIC_API_KEY:
            try:
                self.providers["anthropic"] = AnthropicProvider()
            except Exception as e:
                logger.warning("Failed to initialize AnthropicProvider: %s", e)

    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        """Get provider by name or active default, defaulting safely to mock if key missing."""
        prov_name = (name or settings.LLM_PROVIDER).lower()
        if prov_name in self.providers:
            return self.providers[prov_name]
        return self.providers["mock"]

    async def complete(
        self, request: ChatRequest, auto_execute_tools: bool = True
    ) -> ChatResponse:
        """
        Executes chat completion with caching, exponential backoff retries,
        fallback failover, and optional tool calling execution loop.
        """
        req_dict = request.model_dump()

        # 1. Prompt/Response Caching check (if not forcing fresh or streaming)
        if not request.stream:
            cached_res = cache.get(req_dict)
            if cached_res:
                logger.info("Serving LLM response from cache")
                resp = ChatResponse(**cached_res)
                resp.cached = True
                return resp

        # 2. Resilient Execution with Fallback Chain
        primary = request.provider or settings.LLM_PROVIDER
        if primary not in self.providers:
            primary = "mock"

        async def _call_provider(provider_name: str) -> ChatResponse:
            provider = self.get_provider(provider_name)
            return await provider.generate(request)

        response = await execute_with_retry_and_fallback(
            primary_provider=primary,
            call_fn=_call_provider,
        )

        # 3. Autonomous Tool Calling Resolution
        if auto_execute_tools and response.tool_calls:
            logger.info("Executing %d returned tool calls", len(response.tool_calls))
            for tc in response.tool_calls:
                t_name = tc.function.get("name")
                t_args = tc.function.get("arguments")
                tool_output = await tool_registry.execute(t_name, t_args)

                # Synthesize final response with tool result
                response.content = (
                    f"Tool Execution [{t_name}]: {tool_output}\n\n"
                    f"Synthesized Result: The operation completed successfully with output: {tool_output}"
                )

        # 4. Save to Cache
        if not request.stream:
            cache.set(req_dict, response.model_dump())

        return response


llm_service = LLMService()
