from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class PromptRecord(BaseModel):
    id: str
    category: str
    text: str


class ModelConfig(BaseModel):
    name: str
    provider: str
    model_id: str
    max_output_tokens: int = 128


class RequestResult(BaseModel):
    model: str
    prompt_id: str
    run_id: int
    ttft_ms: Optional[float]
    total_latency_ms: Optional[float]
    output_chars: int
    output_tokens_est: int
    status: str  # "ok" | "error"
    error: Optional[str]


class AggregateStats(BaseModel):
    model: str
    n: int
    mean_ttft_ms: Optional[float]
    median_ttft_ms: Optional[float]
    p95_ttft_ms: Optional[float]
    mean_total_ms: float
    median_total_ms: float
    p95_total_ms: float
    mean_output_chars: float
    error_rate: float
