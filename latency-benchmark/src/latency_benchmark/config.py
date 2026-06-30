from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    diffusion_api_key: str = ""
    diffusion_base_url: str = ""

    mercury_api_key: str = ""
    mercury_base_url: str = "https://api.inceptionlabs.ai"

    # Benchmark defaults — kept small to conserve the 10M free token budget
    repeats: int = 5
    warmup_runs: int = 1
    max_output_tokens: int = 32


settings = Settings()
