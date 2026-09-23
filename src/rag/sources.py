"""Programmatic source assembly from Top-k text hits (SPEC §11.2)."""

from __future__ import annotations

from src.retrieve.search import SearchHit
from src.types import SourceRef

CONTENT_PREVIEW_MAX = 120


def format_sources(text_hits: list[SearchHit]) -> list[SourceRef]:
    """Build sources[] ONLY from retrieved text hits — never from LLM output."""
    refs: list[SourceRef] = []
    for hit in text_hits:
        meta = hit.metadata
        content = str(meta.get("content", ""))
        preview = content if len(content) <= CONTENT_PREVIEW_MAX else content[:CONTENT_PREVIEW_MAX]
        refs.append(
            SourceRef(
                source=str(meta.get("source", "")),
                chunk_id=int(meta.get("chunk_id", 0)),
                similarity=float(hit.similarity),
                content_preview=preview,
            )
        )
    return refs
