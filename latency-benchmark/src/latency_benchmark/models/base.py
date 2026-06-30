from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    name: str

    async def generate_stream(
        self, prompt: str, max_output_tokens: int
    ) -> AsyncIterator[str]: ...
