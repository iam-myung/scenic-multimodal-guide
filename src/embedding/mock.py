"""Deterministic mock embeddings for Key-free smoke builds (not for e2e)."""

from __future__ import annotations

import hashlib
import math

MOCK_EMBEDDING_DIM = 32
MOCK_EMBEDDING_MODEL = "mock-embedding-v1"


def _vector_from_seed(seed: str, dim: int = MOCK_EMBEDDING_DIM) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dim:
        for b in digest:
            # Map byte to (-1, 1)
            values.append((b / 255.0) * 2.0 - 1.0)
            if len(values) >= dim:
                break
        digest = hashlib.sha256(digest).digest()
    # L2-normalize for stable distances
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def get_text_embedding(text: str) -> list[float]:
    return _vector_from_seed(f"text:{text}")


def get_image_embedding(image_base64: str, *, image_format: str = "jpeg") -> list[float]:
    head = image_base64[:64] if not image_base64.startswith("data:") else image_base64[:80]
    return _vector_from_seed(f"image:{image_format}:{head}")


def get_video_embedding(video_url: str) -> list[float]:
    return _vector_from_seed(f"video:{video_url}")
