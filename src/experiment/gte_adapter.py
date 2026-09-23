"""GTE-Qwen2-1.5B-instruct contrast embedder (SentenceTransformer) — reference 19_."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.experiment.adapters_common import (
    ExperimentDependencyError,
    resolve_cached_model_dir,
)

GTE_MODEL_ID = "iic/gte_Qwen2-1.5B-instruct"


class GteQwenEmbedder:
    """Query uses prompt_name='query'; docs plain encode (SPEC §9)."""

    name = "gte-qwen2-1.5b"

    def __init__(self, model_dir: str | Path) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ExperimentDependencyError(
                "sentence_transformers is not installed. "
                "Install optional deps: pip install -r requirements-experiment.txt"
            ) from exc
        self._model = SentenceTransformer(str(model_dir), trust_remote_code=True)
        self._model.max_seq_length = 8192

    @classmethod
    def from_cache(cls, cache_dir: str | Path | None = None) -> GteQwenEmbedder:
        root = Path(cache_dir) if cache_dir is not None else None
        local = resolve_cached_model_dir(GTE_MODEL_ID, cache_dir=root)
        return cls(local)

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode([text], prompt_name="query")[0]
        return list(map(float, vec))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts))
        return [list(map(float, row)) for row in vectors]
