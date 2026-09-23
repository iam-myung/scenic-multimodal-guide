"""DashScope multimodal embedding adapter (SPEC §6 embedding.mm).

Reference logic from 18_ notebooks; re-implemented here.
Model: tongyi-embedding-vision-plus-2026-03-06 (current Bailian market id).
Video: URL only; multi-frame embeddings are mean-pooled.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

import dashscope

from src.config import get_dashscope_http_api_url, require_dashscope_api_key

# Official Bailian model code (2026-03-06 revision). Legacy alias without
# date suffix may still hit FreeTierOnly when free quota is exhausted.
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus-2026-03-06"


class EmbeddingError(RuntimeError):
    """Raised when DashScope multimodal embedding fails."""


def _mean_vectors(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def _normalize_image_format(image_format: str) -> str:
    fmt = image_format.lower().lstrip(".")
    if fmt == "jpg":
        return "jpeg"
    return fmt


def _to_image_data_uri(image_base64: str, image_format: str = "jpeg") -> str:
    if image_base64.startswith("data:image/"):
        return image_base64
    fmt = _normalize_image_format(image_format)
    return f"data:image/{fmt};base64,{image_base64}"


def _ensure_api_key() -> None:
    # Pay-as-you-go MaaS workspace: point SDK away from public free-tier host.
    dashscope.base_http_api_url = get_dashscope_http_api_url()
    dashscope.api_key = require_dashscope_api_key()


def _call_embedding(input_payload: list[dict[str, str]], *, modality: str) -> list[dict[str, Any]]:
    _ensure_api_key()
    resp = dashscope.MultiModalEmbedding.call(
        model=MULTIMODAL_EMBEDDING_MODEL,
        input=input_payload,
    )
    if resp.status_code != HTTPStatus.OK:
        detail = getattr(resp, "message", "") or getattr(resp, "code", "") or "unknown error"
        raise EmbeddingError(
            f"{modality} embedding failed (status={resp.status_code}): {detail}"
        )
    embeddings = resp.output["embeddings"]
    if not embeddings:
        raise EmbeddingError(f"{modality} embedding failed: empty embeddings in response")
    return embeddings


def get_text_embedding(text: str) -> list[float]:
    """Embed a text string via tongyi-embedding-vision-plus."""
    embeddings = _call_embedding([{"text": text}], modality="text")
    return list(embeddings[0]["embedding"])


def get_image_embedding(image_base64: str, *, image_format: str = "jpeg") -> list[float]:
    """Embed an image from raw base64 or a full data:image/...;base64,... URI."""
    image_data = _to_image_data_uri(image_base64, image_format=image_format)
    embeddings = _call_embedding([{"image": image_data}], modality="image")
    return list(embeddings[0]["embedding"])


def get_video_embedding(video_url: str) -> list[float]:
    """Embed a remote video URL; mean-pool when the API returns multiple frames."""
    embeddings = _call_embedding([{"video": video_url}], modality="video")
    if len(embeddings) > 1:
        vectors = [list(e["embedding"]) for e in embeddings]
        return _mean_vectors(vectors)
    return list(embeddings[0]["embedding"])
