from __future__ import annotations

import statistics
from typing import Optional

from latency_benchmark.schemas import AggregateStats, RequestResult


def _percentile(data: list[float], p: float) -> float:
    """Simple percentile without numpy."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def compute_stats(model_name: str, results: list[RequestResult]) -> AggregateStats:
    ok = [r for r in results if r.status == "ok"]
    total_ms_list = [r.total_latency_ms for r in ok if r.total_latency_ms is not None]
    ttft_list: list[float] = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    chars_list = [float(r.output_chars) for r in ok]

    def safe_stat(fn, data: list[float]) -> Optional[float]:
        return fn(data) if data else None

    return AggregateStats(
        model=model_name,
        n=len(results),
        mean_ttft_ms=safe_stat(statistics.mean, ttft_list),
        median_ttft_ms=safe_stat(statistics.median, ttft_list),
        p95_ttft_ms=_percentile(ttft_list, 95) if ttft_list else None,
        mean_total_ms=statistics.mean(total_ms_list) if total_ms_list else 0.0,
        median_total_ms=statistics.median(total_ms_list) if total_ms_list else 0.0,
        p95_total_ms=_percentile(total_ms_list, 95) if total_ms_list else 0.0,
        mean_output_chars=statistics.mean(chars_list) if chars_list else 0.0,
        error_rate=(len(results) - len(ok)) / max(len(results), 1),
    )
