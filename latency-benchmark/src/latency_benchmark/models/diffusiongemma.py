from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from latency_benchmark.config import settings


class DiffusionGemmaAdapter:
    """
    Adapter for DiffusionGemma via an OpenAI-compatible streaming endpoint.

    The endpoint is expected to accept POST /v1/completions with:
      {"model": ..., "prompt": ..., "max_tokens": ..., "stream": true}
    and return SSE lines of the form:
      data: {"choices": [{"text": "..."}]}

    Set DIFFUSION_BASE_URL in .env to point at your serving instance.
    If the endpoint does not support streaming, it falls back to a single
    non-streaming response and reports TTFT == total latency.
    """

    name = "diffusion-gemma"

    def __init__(self, model_id: str = "diffusion-gemma") -> None:
        self._model_id = model_id
        self._base_url = settings.diffusion_base_url.rstrip("/")
        self._api_key = settings.diffusion_api_key

    async def generate_stream(
        self, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model_id,
            "prompt": prompt,
            "max_tokens": max_output_tokens,
            "stream": True,
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        text = chunk["choices"][0].get("text", "")
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
