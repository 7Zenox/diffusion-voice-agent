from __future__ import annotations

from typing import AsyncIterator

from google import genai
from google.genai import types

from latency_benchmark.config import settings


class GeminiAutoregressiveAdapter:
    """Adapter for the Gemini autoregressive text model using the google-genai SDK."""

    name = "gemini-autoregressive"

    def __init__(self, model_id: str = "gemini-2.0-flash-lite") -> None:
        self._model_id = model_id
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def generate_stream(
        self, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[str]:
        config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.0,
        )
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._model_id,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                yield chunk.text
