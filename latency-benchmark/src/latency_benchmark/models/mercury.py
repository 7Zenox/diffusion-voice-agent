from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from latency_benchmark.config import settings


class MercuryAdapter:
    """
    Adapter for Mercury (Inception Labs) via its OpenAI-compatible streaming API.
    Endpoint: https://api.inception.ai/v1/chat/completions
    Docs: https://docs.inception.ai
    """

    name = "mercury-diffusion"

    def __init__(self, model_id: str = "mercury-2") -> None:
        self._model_id = model_id
        self._base_url = settings.mercury_base_url.rstrip("/")
        self._api_key = settings.mercury_api_key or settings.diffusion_api_key

    async def generate_stream(
        self, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        payload = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
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
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
