"""Application settings loaded from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when an environment variable contains an invalid setting."""


def _integer_setting(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer; received {value!r}.") from exc


@dataclass(frozen=True)
class AppConfig:
    model_repo_id: str
    model_filename: str
    model_path: Path
    n_threads: int
    n_ctx: int
    host: str
    port: int
    hf_token: str | None

    @classmethod
    def from_env(cls) -> AppConfig:
        load_dotenv()

        n_threads = _integer_setting("N_THREADS", 4)
        if n_threads != 4:
            raise ConfigurationError("N_THREADS must be exactly 4 for the supported CPU configuration.")

        n_ctx = _integer_setting("N_CTX", 4096)
        if n_ctx <= 0:
            raise ConfigurationError("N_CTX must be a positive integer.")

        port = _integer_setting("PORT", 7860)
        if not 1 <= port <= 65535:
            raise ConfigurationError("PORT must be between 1 and 65535.")

        filename = os.getenv("MODEL_FILENAME", "Qwen3-8B-Q4_K_M.gguf")
        model_path = Path(os.getenv("MODEL_PATH", str(Path("models") / filename))).expanduser()

        return cls(
            model_repo_id=os.getenv("MODEL_REPO_ID", "Qwen/Qwen3-8B-GGUF"),
            model_filename=filename,
            model_path=model_path,
            n_threads=n_threads,
            n_ctx=n_ctx,
            host=os.getenv("HOST", "127.0.0.1"),
            port=port,
            hf_token=os.getenv("HF_TOKEN") or None,
        )
