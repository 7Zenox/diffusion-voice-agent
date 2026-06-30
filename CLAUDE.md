# diffusion-voice-agent

## What this project is

A two-part project exploring diffusion language models:

1. **Latency benchmark** — compare time-to-first-token (TTFT) and total latency between a Gemini autoregressive model and Mercury (a diffusion LLM) on short prompts.
2. **Denoising audio explorer** — run DiffusionGemma locally, intercept every intermediate denoising step, TTS each one with pyttsx3, and save as MP3 so you can *hear* the model go from noise to coherent text. This is the fun part.

The original plan is in `docs/plan.md`.

---

## Current state

### Done
- Full `latency-benchmark/` uv project built and working (Python 3.13)
- Gemini autoregressive adapter (`gemini-2.0-flash-lite`) — streams via google-genai SDK
- Mercury diffusion adapter (`mercury-2`) — streams via `https://api.inceptionlabs.ai/v1/chat/completions`
- CLI: `uv run benchmark run/summarize/check`
- `scripts/denoising_audio.py` — streams Mercury, TTS each SSE chunk cumulatively, saves per-chunk MP3s + `final.mp3`
- Mercury confirmed working: `api.inceptionlabs.ai` is the correct base URL (not `api.inception.ai`)
- Mercury model is `mercury-2` (not `mercury-coder-small`)
- 10M free Mercury tokens — keep `max_output_tokens=32`, `repeats=5` to conserve

### The next thing to build (requires 48GB RAM)
The fun experiment: hook into **real DiffusionGemma denoising steps** via `transformers` and TTS each one.

The model to use: `google/diffusiongemma-26B-A4B-it` (51GB in BF16, needs ~48GB+ RAM).

The key hook is `TextDiffusionStreamer.put_draft()` — transformers calls this at every denoising step (up to 48 steps) with the full current token sequence. By subclassing it you get the intermediate noisy text at each step and can TTS it.

The goal: hear the model go from gibberish → partially coherent → final answer across ~48 steps.

---

## Hardware context

- **This machine (M3 Pro, 18GB)**: too small for BF16 DiffusionGemma. Only runs Mercury via API and Gemini via API.
- **Target machine (48GB RAM)**: can load `google/diffusiongemma-26B-A4B-it` in BF16 (~51GB but macOS can spill to swap; with MPS the active 3.8B params fit in 18GB, full weights need ~50GB so plan for slow load). Consider loading with `torch_dtype=torch.bfloat16, device_map="auto"` — it will use both RAM and MPS.

Actually, 48GB unified memory on Apple Silicon should handle this — the model is 51GB on disk but MoE means only 3.8B params are active per forward pass. Expect the first load to be slow (~5-10 min) but inference to be reasonable.

---

## Key technical facts

### DiffusionGemma denoising steps
- Up to 48 denoising steps per canvas block
- Temperature schedule: 0.8 → 0.4 linear decay
- Adaptive stopping when entropy < 0.005
- `TextDiffusionStreamer.put_draft(token_ids)` is called at each step with the full current canvas
- `TextDiffusionStreamer.put(token_ids)` is called when a canvas block is confirmed (final)
- Subclass `TextDiffusionStreamer` and override `put_draft` to intercept steps

### Mercury streaming
- Mercury sends only 3 SSE chunks for a short answer (not 48 denoising steps)
- It does NOT expose intermediate denoising steps via API — only the final output in chunks
- TTFT ≈ total latency (single-shot output arriving in a few large chunks)

### Why not LiveKit
LiveKit is not used in this project. It's for real-time audio rooms. The benchmark is text-only latency measurement and the denoising explorer just saves MP3 files locally.

---

## Project structure

```
diffusion-voice-agent/
├── CLAUDE.md                  ← you are here
├── docs/plan.md               ← original benchmark plan
└── latency-benchmark/         ← uv project (Python 3.13)
    ├── .env                   ← secrets (not committed)
    ├── .env.example           ← template
    ├── pyproject.toml
    ├── prompts/
    │   └── simple_prompts.jsonl   ← 23 prompts across 4 categories
    ├── src/latency_benchmark/
    │   ├── config.py              ← pydantic-settings, reads .env
    │   ├── schemas.py             ← Pydantic models
    │   ├── runner.py              ← timing logic, perf_counter
    │   ├── metrics.py             ← mean/median/p95 (no numpy)
    │   ├── cli.py                 ← typer CLI: run/summarize/check
    │   └── models/
    │       ├── base.py                    ← ModelAdapter Protocol
    │       ├── gemini_autoregressive.py   ← Gemini streaming adapter
    │       ├── mercury.py                 ← Mercury streaming adapter
    │       └── diffusiongemma.py          ← stub (not yet implemented for local)
    ├── scripts/
    │   ├── denoising_audio.py     ← Mercury chunk → pyttsx3 → MP3
    │   └── audio_out/             ← generated MP3s (chunk_000..N + final.mp3)
    └── results/
        ├── raw/                   ← JSONL benchmark output
        └── reports/               ← markdown + CSV summaries
```

---

## Environment variables (create `latency-benchmark/.env`)

```
GEMINI_API_KEY=<gemini key>
MERCURY_API_KEY=<mercury key from inceptionlabs>
MERCURY_BASE_URL=https://api.inceptionlabs.ai
```

---

## How to run

```bash
cd latency-benchmark

# Install deps
uv sync

# Check credentials
uv run benchmark check

# Run latency benchmark (both models)
uv run benchmark run --model all --repeats 5

# Run benchmark (gemini only)
uv run benchmark run --model gemini

# Summarize results
uv run benchmark summarize --input results/raw/

# Mercury denoising audio explorer (streams Mercury, TTS each chunk, saves MP3)
uv run python scripts/denoising_audio.py "Your prompt here"
afplay scripts/audio_out/final.mp3
```

---

## The task for the 48GB machine: build the DiffusionGemma denoising audio script

Create `scripts/diffusiongemma_denoising_audio.py` that:

1. Loads `google/diffusiongemma-26B-A4B-it` via transformers:
```python
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor, TextDiffusionStreamer

model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    "google/diffusiongemma-26B-A4B-it",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")
```

2. Subclasses `TextDiffusionStreamer` and overrides `put_draft` to:
   - Decode the token ids to text
   - TTS the text with pyttsx3 (save to temp AIFF → convert to MP3 via ffmpeg)
   - Save as `scripts/audio_out/dg_step_{step_idx:03d}.mp3`
   - Print the step text to console

3. Runs generation with the custom streamer:
```python
streamer = DenoiseAudioStreamer(tokenizer=processor.tokenizer)
model.generate(**input_ids, max_new_tokens=80, streamer=streamer)
```

4. Concatenates all step MP3s into `scripts/audio_out/dg_final.mp3` using ffmpeg

The result: an MP3 where you hear the model's output evolve across up to 48 denoising steps from noise to final text.

---

## Dependencies already in pyproject.toml

- `google-genai` — Gemini SDK
- `httpx` — Mercury HTTP streaming
- `pydantic`, `pydantic-settings` — schemas and config
- `typer`, `rich` — CLI
- `orjson` — fast JSON
- `python-dotenv` — .env loading
- `pyttsx3` — local TTS (uses macOS NSSpeechSynthesizer)
- `transformers>=4.52` — required for `DiffusionGemmaForBlockDiffusion` and `TextDiffusionStreamer`
- `torch` — required by transformers
- `accelerate` — required for `device_map="auto"`

ffmpeg must be installed: `brew install ffmpeg`
