"""Verification script for testing local vLLM / OpenAI-compatible server."""

import asyncio
import os
import sys
from openai import AsyncOpenAI


async def verify_vllm(
    base_url: str = "http://localhost:8001/v1",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
):
    print(f"Connecting to vLLM server at: {base_url}...")
    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")

    try:
        models = await client.models.list()
        print(f"Server is alive! Available models: {[m.id for m in models.data]}")

        print("\nSending test completion request...")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant running locally via vLLM.",
                },
                {"role": "user", "content": "Explain PagedAttention in one sentence."},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        print("Response received:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"[NOTE] Local vLLM server test notice: {e}")
        print("To start local vLLM on a GPU machine, run: bash vllm/serve.sh")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    asyncio.run(verify_vllm(base_url=url))
