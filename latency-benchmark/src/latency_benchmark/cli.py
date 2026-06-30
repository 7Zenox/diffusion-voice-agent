from __future__ import annotations

import asyncio
import csv
import datetime
import json
import pathlib
from typing import Annotated, Optional

import orjson
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from latency_benchmark.config import settings
from latency_benchmark.metrics import compute_stats
from latency_benchmark.models.mercury import MercuryAdapter
from latency_benchmark.models.gemini_autoregressive import GeminiAutoregressiveAdapter
from latency_benchmark.runner import run_prompt, warmup
from latency_benchmark.schemas import AggregateStats, PromptRecord, RequestResult

app = typer.Typer(help="Latency benchmark: autoregressive vs diffusion text models.")
console = Console()

RESULTS_RAW = pathlib.Path("results/raw")
RESULTS_REPORTS = pathlib.Path("results/reports")


def _load_prompts(path: pathlib.Path) -> list[PromptRecord]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(PromptRecord(**json.loads(line)))
    return records


def _build_adapters(model_arg: str):
    adapters = []
    if model_arg in ("all", "gemini"):
        adapters.append(GeminiAutoregressiveAdapter())
    if model_arg in ("all", "mercury"):
        adapters.append(MercuryAdapter())
    if not adapters:
        console.print(f"[red]Unknown model '{model_arg}'. Choose: all, gemini, mercury.[/red]")
        raise typer.Exit(1)
    return adapters


async def _run_benchmark(
    model_arg: str,
    prompts_path: pathlib.Path,
    repeats: int,
    warmup_runs: int,
    max_output_tokens: int,
) -> None:
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(prompts_path)
    adapters = _build_adapters(model_arg)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for adapter in adapters:
        console.rule(f"[bold]{adapter.name}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Warming up {adapter.name} ({warmup_runs} runs)…", total=None)
            await warmup(adapter, max_output_tokens, warmup_runs)
            progress.update(task, description=f"Warmup done. Running {len(prompts)} prompts × {repeats}…")

            out_path = RESULTS_RAW / f"{adapter.name}_{ts}.jsonl"
            results: list[RequestResult] = []

            total_runs = len(prompts) * repeats
            run_task = progress.add_task("Benchmarking…", total=total_runs)

            with out_path.open("wb") as fh:
                for prompt in prompts:
                    for run_id in range(repeats):
                        result = await run_prompt(adapter, prompt, run_id, max_output_tokens)
                        results.append(result)
                        fh.write(orjson.dumps(result.model_dump()) + b"\n")
                        progress.advance(run_task)

        stats = compute_stats(adapter.name, results)
        _print_stats(stats)
        console.print(f"[dim]Raw results → {out_path}[/dim]\n")


def _print_stats(stats: AggregateStats) -> None:
    table = Table(title=f"Results: {stats.model}  (n={stats.n})")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    def fmt(v) -> str:
        return f"{v:.1f} ms" if v is not None else "n/a"

    table.add_row("Mean TTFT", fmt(stats.mean_ttft_ms))
    table.add_row("Median TTFT", fmt(stats.median_ttft_ms))
    table.add_row("p95 TTFT", fmt(stats.p95_ttft_ms))
    table.add_row("Mean total latency", fmt(stats.mean_total_ms))
    table.add_row("Median total latency", fmt(stats.median_total_ms))
    table.add_row("p95 total latency", fmt(stats.p95_total_ms))
    table.add_row("Mean output chars", f"{stats.mean_output_chars:.0f}")
    table.add_row("Error rate", f"{stats.error_rate:.1%}")
    console.print(table)


@app.command()
def run(
    model: Annotated[str, typer.Option(help="all | gemini | diffusion")] = "all",
    prompts: Annotated[pathlib.Path, typer.Option(help="Path to JSONL prompts")] = pathlib.Path("prompts/simple_prompts.jsonl"),
    repeats: Annotated[int, typer.Option(help="Runs per prompt")] = settings.repeats,
    warmup_count: Annotated[int, typer.Option("--warmup", help="Warmup runs")] = settings.warmup_runs,
    max_tokens: Annotated[int, typer.Option(help="Max output tokens")] = settings.max_output_tokens,
) -> None:
    """Execute the latency benchmark."""
    asyncio.run(_run_benchmark(model, prompts, repeats, warmup_count, max_tokens))


@app.command()
def summarize(
    input_dir: Annotated[pathlib.Path, typer.Option("--input")] = RESULTS_RAW,
    output: Annotated[Optional[pathlib.Path], typer.Option()] = None,
) -> None:
    """Aggregate raw JSONL results and write a markdown + CSV report."""
    RESULTS_REPORTS.mkdir(parents=True, exist_ok=True)

    by_model: dict[str, list[RequestResult]] = {}
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        with jsonl_file.open("rb") as f:
            for line in f:
                r = RequestResult(**orjson.loads(line))
                by_model.setdefault(r.model, []).append(r)

    if not by_model:
        console.print("[yellow]No result files found.[/yellow]")
        raise typer.Exit(0)

    all_stats: list[AggregateStats] = []
    for model_name, results in by_model.items():
        stats = compute_stats(model_name, results)
        all_stats.append(stats)
        _print_stats(stats)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output or (RESULTS_REPORTS / f"summary_{ts}.md")
    csv_path = RESULTS_REPORTS / f"summary_{ts}.csv"

    _write_markdown(report_path, all_stats)
    _write_csv(csv_path, all_stats)
    console.print(f"\n[green]Report written to {report_path}[/green]")
    console.print(f"[green]CSV written to {csv_path}[/green]")


def _write_markdown(path: pathlib.Path, all_stats: list[AggregateStats]) -> None:
    lines = [
        "# Latency Benchmark Report\n",
        f"Generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n",
        "## Summary\n\n",
        "| Model | Median TTFT (ms) | Median Total (ms) | p95 Total (ms) | Mean Output Chars | Error Rate |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for s in all_stats:
        lines.append(
            f"| {s.model} "
            f"| {s.median_ttft_ms:.1f if s.median_ttft_ms else 'n/a'} "
            f"| {s.median_total_ms:.1f} "
            f"| {s.p95_total_ms:.1f} "
            f"| {s.mean_output_chars:.0f} "
            f"| {s.error_rate:.1%} |\n"
        )

    if len(all_stats) == 2:
        a, b = all_stats[0], all_stats[1]
        lines += [
            "\n## Interpretation\n\n",
            f"In this serving environment, **{a.model}** had median TTFT of "
            f"{a.median_ttft_ms:.1f if a.median_ttft_ms else 'n/a'} ms and median total latency of "
            f"{a.median_total_ms:.1f} ms, while **{b.model}** had median TTFT of "
            f"{b.median_ttft_ms:.1f if b.median_ttft_ms else 'n/a'} ms and median total latency of "
            f"{b.median_total_ms:.1f} ms on short constrained prompts.\n",
        ]

    path.write_text("".join(lines))


def _write_csv(path: pathlib.Path, all_stats: list[AggregateStats]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AggregateStats.model_fields.keys()))
        writer.writeheader()
        for s in all_stats:
            writer.writerow(s.model_dump())


@app.command()
def check() -> None:
    """Validate environment and credentials."""
    ok = True

    if settings.gemini_api_key:
        console.print("[green]✓[/green] GEMINI_API_KEY is set")
    else:
        console.print("[yellow]✗[/yellow] GEMINI_API_KEY not set (gemini model will fail)")
        ok = False

    mercury_key = settings.mercury_api_key or settings.diffusion_api_key
    if mercury_key:
        console.print("[green]✓[/green] Mercury API key is set")
    else:
        console.print("[yellow]✗[/yellow] Mercury API key not set (mercury model will fail)")
        ok = False

    if not ok:
        console.print("\n[dim]Create a .env file with the required variables.[/dim]")
        raise typer.Exit(1)

    console.print("\n[green]Environment looks good.[/green]")


def main() -> None:
    app()
