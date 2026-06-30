# Diffusion vs. Autoregressive Latency Benchmark Plan

## Goal

Implement a minimal benchmark in Python 3.13 using `uv` to compare latency between one autoregressive text model and one diffusion text model on short, simple prompts. The benchmark should measure user-facing latency rather than throughput and should produce clean logs suitable for later analysis.[cite:44][cite:9]

## Scope

This experiment is intentionally narrow. It measures only:

- Time to first token (TTFT) for streaming responses.[cite:44]
- Total completion latency for the full response.[cite:44]
- Output length so latency results can be normalized or sanity-checked for fairness.[cite:9]

The study does not attempt to measure voice latency, throughput under concurrency, tool calling, or end-to-end agent quality. DiffusionGemma is specifically interesting here because published guidance notes that it can have higher TTFT but lower total completion time than autoregressive models in some settings.[cite:9][cite:40]

## Recommendation on LiveKit

LiveKit is **not required** for this experiment. The benchmark is text-only, and the only timing of interest is model request latency and streaming behavior, which can be measured directly from Python HTTP or SDK calls without a real-time media layer.[cite:49][cite:26]

LiveKit only becomes useful if the project later expands into streaming audio, turn detection, multi-participant sessions, or telephony-style routing. For this benchmark, adding LiveKit would increase setup complexity without improving the latency measurement itself.[cite:26][cite:36]

## Model choices

Use one autoregressive model that is accessible with the existing Gemini API key and one diffusion model that is callable through an API endpoint or a local/hosted inference runtime.

Suggested setup:

| Role | Candidate | Why |
|---|---|---|
| Autoregressive baseline | Gemini text model via Gemini API.[cite:49][cite:46] | Already available and easy to benchmark immediately. |
| Diffusion model | DiffusionGemma endpoint or runtime.[cite:38][cite:39] | Designed for lower end-to-end latency on short generation workloads, with known TTFT vs total latency tradeoff.[cite:9][cite:39] |

To keep the experiment clean, fix one baseline and one diffusion model only. Avoid comparing many models in the first pass because that adds variance and makes interpretation harder.

## Fairness rules

The benchmark should be controlled tightly so the comparison is interpretable:

- Use the same prompt set for both models.
- Keep prompts short and response constraints strict.
- Set low temperature or deterministic decoding where supported.
- Use the same maximum output length cap for both models.
- Run with concurrency = 1 so the measurement reflects single-user latency rather than batch throughput.[cite:45][cite:40]
- Record output length because shorter answers can make a model appear faster unfairly.[cite:9]

Important caveat: vendor-reported speedups for DiffusionGemma often depend on specific hardware and serving stacks, so the final writeup should frame results as latency in the chosen serving environment rather than a universal claim about the model family.[cite:39][cite:45]

## Prompt set

Use a small, repeatable prompt suite of short tasks. The first version should contain 20 to 25 prompts across 4 simple categories:

1. One-sentence summaries.
2. One-sentence factual explanations.
3. Polite rewrites.
4. Small extraction tasks.

Examples:

- "Summarize this in one sentence: Python uses indentation to define code blocks."
- "Answer in one sentence: What is recursion?"
- "Rewrite politely: send me the file now."
- "Extract the city name only: I moved from Pune to Bengaluru last year."

All prompts should specify a short response format so the benchmark stays focused on latency for simple responses instead of long-form generation.[cite:49][cite:9]

## Metrics

Each request should log the following fields:

| Metric | Meaning |
|---|---|
| `started_at` | Timestamp right before the request is sent. |
| `first_token_at` | Timestamp when the first streamed token/chunk is received. |
| `completed_at` | Timestamp when the full response finishes. |
| `ttft_ms` | `first_token_at - started_at` in milliseconds.[cite:44] |
| `total_latency_ms` | `completed_at - started_at` in milliseconds.[cite:44] |
| `output_chars` | Final response character count. |
| `output_tokens_est` | Optional estimated token count. |
| `status` | Success or failure. |
| `error` | Exception or API error, if any. |

Report at least mean, median, and p95 for TTFT and total latency after collection.

## Project structure

Use a simple `uv` project layout:

```text
latency-benchmark/
├── pyproject.toml
├── .python-version
├── .env
├── README.md
├── prompts/
│   └── simple_prompts.jsonl
├── src/
│   └── latency_benchmark/
│       ├── __init__.py
│       ├── config.py
│       ├── models/
│       │   ├── base.py
│       │   ├── gemini_autoregressive.py
│       │   └── diffusiongemma.py
│       ├── runner.py
│       ├── metrics.py
│       ├── schemas.py
│       └── cli.py
└── results/
    ├── raw/
    └── reports/
```

## Environment setup

### Python and uv

Use Python 3.13 with `uv`.

```bash
uv init latency-benchmark
cd latency-benchmark
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
```

Add dependencies as needed. A typical starting set is:

```bash
uv add httpx pydantic pydantic-settings typer rich orjson python-dotenv
```

If a provider SDK is more convenient than raw HTTP, add it only for that provider. Keep the common benchmark path provider-agnostic.

### Environment variables

Store secrets in `.env`:

```bash
GEMINI_API_KEY=...
DIFFUSION_API_KEY=...
DIFFUSION_BASE_URL=...
```

If the diffusion path is local instead of hosted, replace API credentials with the local endpoint URL.

## Implementation steps

### Step 1: Define the benchmark schema

Create Pydantic models for:

- Prompt record.
- Model configuration.
- Request result.
- Aggregate summary.

This keeps output structured and makes later analysis easier.

### Step 2: Create a common model adapter interface

Define a small interface such as:

```python
class ModelAdapter(Protocol):
    name: str
    async def generate_stream(self, prompt: str, max_output_tokens: int) -> AsyncIterator[str]: ...
```

The benchmark runner should not know whether the model is Gemini or DiffusionGemma. It should only consume streamed chunks and timestamp them.

### Step 3: Implement the Gemini adapter

Implement one adapter for the autoregressive baseline using the Gemini API text generation path.[cite:49][cite:46]

Requirements:

- Support streaming if available.
- Timestamp first chunk arrival.
- Concatenate final output.
- Capture request failures cleanly.

### Step 4: Implement the diffusion adapter

Implement one adapter for DiffusionGemma using whichever access path is simplest: hosted endpoint first, local runtime only if setup remains light.[cite:38][cite:39]

Requirements:

- Match the same interface as the Gemini adapter.
- Expose a comparable short-output generation setting.
- Record any differences in output format or streaming semantics in notes.

### Step 5: Build the runner

The runner should:

1. Load prompts from JSONL.
2. For each model, iterate through all prompts.
3. Repeat each prompt `n` times, such as 20 runs.
4. Measure TTFT and total latency.
5. Write one JSONL row per request into `results/raw/`.

Use `time.perf_counter()` for high-resolution timing.

### Step 6: Add a warmup phase

Before recording results, send 3 to 5 warmup requests per model. This helps reduce noise from cold starts, first connection setup, and just-in-time initialization effects.

Warmup results should not be included in the final metrics.

### Step 7: Summarize the results

After raw collection, compute:

- Mean TTFT.
- Median TTFT.
- p95 TTFT.
- Mean total latency.
- Median total latency.
- p95 total latency.
- Mean output length.
- Error rate.

Write a small markdown summary in `results/reports/summary.md` and a CSV/JSON export for plotting later.

## CLI design

Use `typer` for a small CLI:

```bash
uv run benchmark run --model all --prompts prompts/simple_prompts.jsonl --repeats 20
uv run benchmark summarize --input results/raw/
```

Recommended commands:

- `benchmark run` — execute benchmark.
- `benchmark summarize` — aggregate raw outputs.
- `benchmark check` — validate environment and credentials.

## Logging format

Prefer JSONL for raw request logs. One row per request keeps the benchmark append-friendly and easy to analyze.

Example shape:

```json
{
  "model": "gemini-autoregressive",
  "prompt_id": "summary_01",
  "run_id": 7,
  "ttft_ms": 412.8,
  "total_latency_ms": 1387.1,
  "output_chars": 84,
  "status": "ok",
  "error": null
}
```

## Analysis plan

The first analysis should stay simple.

Primary comparison:

- Compare median TTFT between models.
- Compare median total latency between models.
- Compare p95 total latency between models.

Secondary checks:

- Compare output lengths.
- Inspect failures and outliers.
- Check whether diffusion shows the expected pattern of worse TTFT but better total completion latency.[cite:9][cite:40]

A clear result sentence would look like this:

> In this serving setup, the diffusion model showed higher first-token delay but lower full-response latency on short constrained prompts.[cite:9][cite:40]

## Risks and controls

Main sources of noise:

- Network jitter.
- API-side queueing.
- Variable output lengths.
- Server cold starts.
- Different streaming semantics across providers.

Controls:

- Repeat each prompt many times.
- Use warmups.
- Fix concurrency to 1.[cite:45]
- Keep prompts short and outputs constrained.
- Log failures rather than dropping them.

## Definition of done

The implementation is complete when all of the following are true:

- A `uv` project runs on Python 3.13.
- One Gemini autoregressive adapter works end to end.[cite:49]
- One DiffusionGemma adapter works end to end.[cite:38]
- Raw JSONL benchmark logs are written successfully.
- A summary command computes mean, median, and p95 metrics.
- The final markdown report clearly states whether diffusion improved total latency, TTFT, both, or neither in the tested environment.

## Final recommendation

Do **not** use LiveKit for this experiment. It is unnecessary for a text-only latency benchmark and would add infrastructure without improving the measurement quality.[cite:26][cite:36]

The simplest successful version is: Python 3.13, `uv`, two model adapters, one prompt file, one benchmark runner, JSONL logs, and one summarizer. That is enough to produce a clean experiment quickly and leaves room to expand into voice later if the latency results are promising.
