"""Application configuration — SPEC §13.

Secrets live only in the project-root `.env` file (never in source).

Providers:
- Embedding (build / retrieve): DashScope MultiModalEmbedding → DASHSCOPE_* 
- LLM (answer generation): OpenAI-compatible → LLM_* (DeepSeek / Qwen / …)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = _PROJECT_ROOT / ".env"

# Public DashScope defaults (embedding + optional Qwen LLM).
DEFAULT_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_HTTP_API_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_LLM_MODEL = "qwen-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def load_env(*, override: bool = False) -> Path:
    """Load key/value pairs from the dedicated `.env` file at project root."""
    if not ENV_FILE.is_file():
        raise ValueError(
            f"Env file missing: {ENV_FILE}. "
            "Copy .env.example to .env and set keys "
            "(values are never logged)."
        )
    load_dotenv(dotenv_path=ENV_FILE, override=override)
    return ENV_FILE


def _env(name: str, default: str = "") -> str:
    if ENV_FILE.is_file():
        load_dotenv(dotenv_path=ENV_FILE, override=False)
    return (os.getenv(name) or default).strip()


def require_dashscope_api_key() -> str:
    """DashScope key for multimodal embedding (build / retrieve).

    Raises ValueError if missing/empty. Never includes the key value in the message.
    """
    load_env()
    key = os.getenv("DASHSCOPE_API_KEY")
    if key is None or not str(key).strip():
        raise ValueError(
            "DASHSCOPE_API_KEY is missing or empty in .env. "
            "Multimodal embedding (建库/检索) still requires a DashScope / "
            "Model Studio key — DeepSeek cannot replace it. "
            "(value is never logged)."
        )
    return str(key).strip()


def get_dashscope_compat_base_url() -> str:
    """Legacy Qwen OpenAI-compatible URL (used only if LLM_BASE_URL unset)."""
    raw = _env("DASHSCOPE_COMPAT_BASE_URL")
    return (raw or DEFAULT_COMPAT_BASE_URL).rstrip("/")


def get_dashscope_http_api_url() -> str:
    """Native DashScope HTTP API root for MultiModalEmbedding."""
    raw = _env("DASHSCOPE_HTTP_API_URL")
    if raw:
        return raw.rstrip("/")
    compat = get_dashscope_compat_base_url()
    if "compatible-mode/v1" in compat:
        return compat.replace("compatible-mode/v1", "api/v1").rstrip("/")
    return DEFAULT_HTTP_API_URL


def require_llm_api_key() -> str:
    """LLM key: LLM_API_KEY, else fall back to DASHSCOPE_API_KEY."""
    load_env()
    key = os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if key is None or not str(key).strip():
        raise ValueError(
            "LLM_API_KEY (or DASHSCOPE_API_KEY) is missing or empty in .env. "
            "(value is never logged)."
        )
    return str(key).strip()


def get_llm_base_url() -> str:
    """OpenAI-compatible chat base URL (DeepSeek / DashScope / …)."""
    raw = _env("LLM_BASE_URL")
    if raw:
        return raw.rstrip("/")
    return get_dashscope_compat_base_url()


def get_llm_model() -> str:
    """Chat model id (e.g. deepseek-chat, qwen-flash)."""
    return _env("LLM_MODEL", DEFAULT_LLM_MODEL) or DEFAULT_LLM_MODEL


def get_modelscope_cache_dir() -> Path:
    """Return MODELSCOPE_CACHE_DIR (default: <project>/models)."""
    raw = _env("MODELSCOPE_CACHE_DIR")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()
    return (_PROJECT_ROOT / "models").resolve()
