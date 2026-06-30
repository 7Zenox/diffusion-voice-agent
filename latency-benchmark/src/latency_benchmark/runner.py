from __future__ import annotations

import time
from typing import AsyncIterator

from latency_benchmark.models.base import ModelAdapter
from latency_benchmark.schemas import PromptRecord, RequestResult


async def _timed_generate(
    adapter: ModelAdapter, prompt: str, max_output_tokens: int
) -> RequestResult:
    """Run one generation and capture timing + output."""
    output_parts: list[str] = []
    first_token_at: float | None = None
    started_at = time.perf_counter()

    try:
        stream: AsyncIterator[str] = adapter.generate_stream(prompt, max_output_tokens)
        async for chunk in stream:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            output_parts.append(chunk)

        completed_at = time.perf_counter()
        full_output = "".join(output_parts)
        output_chars = len(full_output)
        output_tokens_est = output_chars // 4

        ttft_ms = (first_token_at - started_at) * 1000 if first_token_at else None
        total_ms = (completed_at - started_at) * 1000

        return RequestResult(
            model=adapter.name,
            prompt_id="",
            run_id=0,
            ttft_ms=ttft_ms,
            total_latency_ms=total_ms,
            output_chars=output_chars,
            output_tokens_est=output_tokens_est,
            status="ok",
            error=None,
        )

    except Exception as exc:
        completed_at = time.perf_counter()
        return RequestResult(
            model=adapter.name,
            prompt_id="",
            run_id=0,
            ttft_ms=None,
            total_latency_ms=(completed_at - started_at) * 1000,
            output_chars=0,
            output_tokens_est=0,
            status="error",
            error=str(exc),
        )


async def run_prompt(
    adapter: ModelAdapter,
    prompt: PromptRecord,
    run_id: int,
    max_output_tokens: int,
) -> RequestResult:
    result = await _timed_generate(adapter, prompt.text, max_output_tokens)
    result.prompt_id = prompt.id
    result.run_id = run_id
    return result


async def warmup(adapter: ModelAdapter, max_output_tokens: int, runs: int = 3) -> None:
    probe = PromptRecord(id="warmup", category="warmup", text="Say hi.")
    for _ in range(runs):
        await _timed_generate(adapter, probe.text, max_output_tokens)
