"""Full-corpus L2 retrieval with similarity and media selection (SPEC §7.1 / §8.3).

No LLM generation in this module. Query vectors are injectable for unit tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.intent.media_intent import MEDIA_DISTANCE_THRESHOLD
from src.store.faiss_store import search as faiss_search

DEFAULT_TEXT_TOP_K = 3

EmbedFn = Callable[[str], list[float]]

_LEADING_INDEX_PREFIX = re.compile(r"^\d+-")


@dataclass(frozen=True)
class SearchHit:
    id: int
    distance: float
    similarity: float
    metadata: dict[str, Any]


def distance_to_similarity(distance: float) -> float:
    """Map L2 distance to display similarity: closer → higher."""
    return 1.0 / (1.0 + float(distance))


def path_label_affinity(query: str, metadata: dict[str, Any]) -> int:
    """Score how strongly the media path/filename label appears in the query.

    Used to break near-ties among threshold-passing media hits (SPEC §15.1 T2/T4).
    Strip leading ``N-`` from the path stem (e.g. ``1-园区合影`` → label).
    """
    q = str(query or "")
    if not q:
        return 0
    path = str(metadata.get("path") or "")
    if not path:
        return 0
    label = _LEADING_INDEX_PREFIX.sub("", Path(path).stem)
    if label and label in q:
        return len(label)
    return 0


def search_with_details(
    query: str,
    index: Any,
    metadata: list[dict[str, Any]],
    *,
    query_vector: np.ndarray | list[float] | None = None,
    embed_fn: EmbedFn | None = None,
) -> list[SearchHit]:
    """Search the full index (k=ntotal); attach similarity and metadata.

    Provide either ``query_vector`` (preferred in unit tests) or ``embed_fn``.
    Does not call DashScope unless the caller passes a real embed_fn.
    """
    if query_vector is not None:
        vec = np.asarray(query_vector, dtype=np.float32)
    elif embed_fn is not None:
        vec = np.asarray(embed_fn(query), dtype=np.float32)
    else:
        raise ValueError(
            "search_with_details requires query_vector or embed_fn "
            "(refuse silent real-API calls in this layer)"
        )

    raw_hits = faiss_search(index, vec, k=None)
    results: list[SearchHit] = []
    for idx, dist in raw_hits:
        if idx < 0 or idx >= len(metadata):
            continue
        results.append(
            SearchHit(
                id=idx,
                distance=float(dist),
                similarity=distance_to_similarity(dist),
                metadata=metadata[idx],
            )
        )
    return results


def select_text_hits(
    hits: list[SearchHit],
    k: int = DEFAULT_TEXT_TOP_K,
) -> list[SearchHit]:
    """Keep type==text hits in distance order; take Top-k (default 3)."""
    text = [h for h in hits if h.metadata.get("type") == "text"]
    return text[: max(0, int(k))]


def select_media_hit(
    hits: list[SearchHit],
    *,
    media_type: str,
    want: bool,
    threshold: float = MEDIA_DISTANCE_THRESHOLD,
    query: str = "",
) -> SearchHit | None:
    """If intent is true, pick best media hit under threshold.

    Ranking: higher path-label affinity first, then lower L2 distance
    (still requires distance < threshold; no LLM / text parsing).
    """
    if not want:
        return None
    candidates = [
        h
        for h in hits
        if h.metadata.get("type") == media_type and h.distance < float(threshold)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda h: (
            -path_label_affinity(query, h.metadata),
            h.distance,
        ),
    )
