"""BGE-M3 contrast embedder (FlagEmbedding) — reference 19_, isolated from FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.experiment.adapters_common import (
    ExperimentDependencyError,
    resolve_cached_model_dir,
)

BGE_MODEL_ID = "BAAI/bge-m3"


class BgeM3Embedder:
    """Dense-only encoder; similarity is query · doc (SPEC §9 / 19_)."""

    name = "bge-m3"

    def __init__(self, model_dir: str | Path, *, use_fp16: bool = True) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ExperimentDependencyError(
                "FlagEmbedding is not installed. "
                "Install optional deps: pip install -r requirements-experiment.txt"
            ) from exc
        self._model = BGEM3FlagModel(str(model_dir), use_fp16=use_fp16)

    @classmethod
    def from_cache(cls, cache_dir: str | Path | None = None) -> BgeM3Embedder:
        root = Path(cache_dir) if cache_dir is not None else None
        local = resolve_cached_model_dir(BGE_MODEL_ID, cache_dir=root)
        return cls(local)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = self._model.encode(
            list(texts),
            batch_size=12,
            max_length=8192,
        )["dense_vecs"]
        return [list(map(float, row)) for row in out]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)
