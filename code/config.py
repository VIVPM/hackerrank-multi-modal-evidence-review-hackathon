"""Configuration & secrets. Secrets come from env only (contract 6.2)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # ponytail: dotenv optional, env vars still work without it
    def load_dotenv(*args, **kwargs) -> bool:  # type: ignore[misc]
        return False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "code" / ".cache"

load_dotenv(REPO_ROOT / ".env")

# Friendly alias -> HF model id (env-overridable). --model accepts an alias or a literal id.
MODELS = {
    "cheap": os.getenv("VLM_MODEL_CHEAP", "Qwen/Qwen3-VL-8B-Instruct"),
    "strong": os.getenv("VLM_MODEL_STRONG", "Qwen/Qwen2.5-VL-72B-Instruct"),
}


def resolve_model(name: str) -> str:
    """Map an alias to a model id, or pass through a literal HF model id."""
    return MODELS.get(name, name)


@dataclass
class Settings:
    model: str = "cheap"                 # alias or literal HF model id
    provider: str = field(default_factory=lambda: os.getenv("HF_PROVIDER", "auto"))
    mock: bool = False
    limit: int | None = None
    sleep: float = 1.5                   # seconds between live calls (skipped on cache hit)
    workers: int = 3
    temperature: float = 0.0
    max_tokens: int = 700
    max_image_px: int = 1024             # long-edge resize before encoding
    jpeg_quality: int = 85
    max_retries: int = 4
    # response_format json_schema isn't supported by every HF provider; off by default.
    # We rely on prompt + defensive parsing. Enable with VLM_USE_SCHEMA=1 if your provider supports it.
    use_schema: bool = field(
        default_factory=lambda: os.getenv("VLM_USE_SCHEMA", "").lower() in ("1", "true", "yes"))
    cache_path: Path = CACHE_DIR / "vlm_cache.json"
    hf_token: str | None = field(default_factory=lambda: (os.getenv("HF_TOKEN") or "").strip() or None)

    @property
    def model_id(self) -> str:
        return resolve_model(self.model)
