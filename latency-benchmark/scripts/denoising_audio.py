"""
Stream Mercury's SSE chunks, TTS each one with pyttsx3, and save:
  - scripts/audio_out/chunk_000.mp3, chunk_001.mp3, ... (one per SSE chunk)
  - scripts/audio_out/final.mp3                         (all chunks concatenated)

Usage:
    uv run python scripts/denoising_audio.py "Your prompt here"
    uv run python scripts/denoising_audio.py  # uses default prompt
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

import subprocess

import httpx
import pyttsx3
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR = pathlib.Path(__file__).parent / "audio_out"
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_PROMPT = (
    "Explain in two sentences why the sky is blue."
)

MERCURY_BASE_URL = "https://api.inceptionlabs.ai"
MERCURY_MODEL = "mercury-2"

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    from dotenv import load_dotenv
    import os
    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
    key = os.getenv("MERCURY_API_KEY") or os.getenv("DIFFUSION_API_KEY", "")
    if not key:
        console.print("[red]MERCURY_API_KEY not set in .env[/red]")
        sys.exit(1)
    return key


def _tts_to_aiff(engine: pyttsx3.Engine, text: str, path: pathlib.Path) -> None:
    """Synthesize text to an AIFF file using pyttsx3 (macOS native engine)."""
    engine.save_to_file(text, str(path))
    engine.runAndWait()


def _aiff_to_mp3(aiff_path: pathlib.Path, mp3_path: pathlib.Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), "-b:a", "64k", str(mp3_path)],
        check=True, capture_output=True,
    )


def _stream_mercury(prompt: str, api_key: str):
    """Yield raw SSE text chunks from Mercury as they arrive."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": MERCURY_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 80,
        "temperature": 0.0,
        "stream": True,
    }

    with httpx.Client(timeout=60.0) as client:
        with client.stream(
            "POST",
            f"{MERCURY_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
    api_key = _load_api_key()

    console.print(Panel(f"[bold]Prompt:[/bold] {prompt}", title="Mercury Denoising Audio"))
    console.print(f"Output directory: [cyan]{OUT_DIR}[/cyan]\n")

    engine = pyttsx3.init()
    # Slightly slower rate so each chunk is clearer
    engine.setProperty("rate", 160)

    chunk_mp3s: list[pathlib.Path] = []
    full_text_parts: list[str] = []

    console.print("[bold yellow]Streaming Mercury...[/bold yellow]")

    for idx, chunk_text in enumerate(_stream_mercury(prompt, api_key)):
        full_text_parts.append(chunk_text)
        cumulative = "".join(full_text_parts)

        console.print(f"\n[dim]── chunk {idx:03d} ──[/dim]")
        console.print(f"  [green]+[/green] new text : [italic]{chunk_text!r}[/italic]")
        console.print(f"  [blue]∑[/blue] cumulative: {cumulative}")

        # TTS the cumulative text so far (like hearing the output evolve)
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
            aiff_path = pathlib.Path(tmp.name)

        t0 = time.perf_counter()
        _tts_to_aiff(engine, cumulative, aiff_path)
        mp3_path = OUT_DIR / f"chunk_{idx:03d}.mp3"
        _aiff_to_mp3(aiff_path, mp3_path)
        aiff_path.unlink(missing_ok=True)
        elapsed = (time.perf_counter() - t0) * 1000

        chunk_mp3s.append(mp3_path)
        console.print(f"  [magenta]🔊[/magenta] saved  : {mp3_path.name}  ({elapsed:.0f} ms TTS)")

    if not chunk_mp3s:
        console.print("[red]No chunks received from Mercury.[/red]")
        return

    # Concatenate all chunk MP3s into final.mp3 via ffmpeg concat
    console.print("\n[bold]Concatenating chunks → final.mp3...[/bold]")
    final_path = OUT_DIR / "final.mp3"
    list_file = OUT_DIR / "_concat.txt"
    silence_path = OUT_DIR / "_silence.mp3"

    # Generate a short silence clip once
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
         "-t", "0.4", "-b:a", "64k", str(silence_path)],
        check=True, capture_output=True,
    )

    with list_file.open("w") as f:
        for mp3 in chunk_mp3s:
            f.write(f"file '{mp3.resolve()}'\n")
            f.write(f"file '{silence_path.resolve()}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-b:a", "64k", str(final_path)],
        check=True, capture_output=True,
    )
    list_file.unlink(missing_ok=True)
    silence_path.unlink(missing_ok=True)

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Chunks : {len(chunk_mp3s)} files in {OUT_DIR}/")
    console.print(f"  Final  : {final_path}")
    console.print(f"\n[dim]Play with: afplay {final_path}[/dim]")


if __name__ == "__main__":
    main()
