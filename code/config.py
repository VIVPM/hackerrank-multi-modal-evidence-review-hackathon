# Loads environment-based pipeline configuration and model settings.
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    # Provides a no-op environment loader when python-dotenv is unavailable.
    def load_dotenv(*args, **kwargs) -> bool:
        _ = args, kwargs
        return False

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = REPO_ROOT / "code" / ".cache"

load_dotenv(REPO_ROOT / ".env")

MODELS = {
    "cheap": os.getenv("VLM_MODEL_CHEAP", "Qwen/Qwen3-VL-8B-Instruct"),
    "strong": os.getenv("VLM_MODEL_STRONG", "Qwen/Qwen2.5-VL-72B-Instruct"),
}


# Resolves a model alias or preserves a literal model ID.
def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


# Stores all settings used by the claim-review pipeline.
@dataclass
class Settings:
    model: str = "cheap"
    provider: str = field(default_factory=lambda: os.getenv("HF_PROVIDER", "auto"))
    mock: bool = False
    limit: int | None = None
    sleep: float = 1.5
    workers: int = 3
    temperature: float = 0.0
    max_tokens: int = 700
    max_image_px: int = 1024
    jpeg_quality: int = 85
    max_retries: int = 4
    use_schema: bool = field(
        default_factory=lambda: os.getenv("VLM_USE_SCHEMA", "").lower() in ("1", "true", "yes"))
    cache_path: Path = CACHE_DIR / "vlm_cache.json"
    hf_token: str | None = field(default_factory=lambda: (os.getenv("HF_TOKEN") or "").strip() or None)

    # Returns the resolved provider model ID.
    @property
    def model_id(self) -> str:
        return resolve_model(self.model)
