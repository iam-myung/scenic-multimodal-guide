"""Shared helpers for optional contrast-experiment model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from src.config import get_modelscope_cache_dir


class ExperimentDependencyError(RuntimeError):
    """Raised when requirements-experiment.txt deps are missing."""


class ContrastEmbedder(Protocol):
    name: str

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


def ensure_modelscope_model(model_id: str, *, cache_dir: Path | None = None) -> Path:
    """Download (if needed) and return local model directory under cache."""
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise ExperimentDependencyError(
            "modelscope is not installed. "
            "Install optional deps: pip install -r requirements-experiment.txt"
        ) from exc

    root = Path(cache_dir) if cache_dir is not None else get_modelscope_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(model_id, cache_dir=str(root))
    return Path(local)


def resolve_cached_model_dir(model_id: str, *, cache_dir: Path | None = None) -> Path:
    """Prefer an existing cache hit; otherwise snapshot_download."""
    root = Path(cache_dir) if cache_dir is not None else get_modelscope_cache_dir()
    org, _, name = model_id.partition("/")
    candidates: list[Path] = [root / model_id]
    if org and name:
        candidates.extend(
            [
                root / org / name,
                root / org / name.replace(".", "___"),
                root / org / name.replace(".", "_"),
            ]
        )
    for path in candidates:
        if path.is_dir():
            return path
    return ensure_modelscope_model(model_id, cache_dir=root)
