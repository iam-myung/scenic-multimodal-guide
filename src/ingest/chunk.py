"""Fixed-size overlapping text chunking — SPEC CHUNK_SIZE=500, OVERLAP=50."""

from __future__ import annotations

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into fixed-size chunks with overlap; skip whitespace-only pieces."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks
