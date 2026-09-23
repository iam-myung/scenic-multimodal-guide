"""Multimodal embedding adapters (DashScope primary path)."""

from src.embedding.mm_dashscope import (
    MULTIMODAL_EMBEDDING_MODEL,
    EmbeddingError,
    get_image_embedding,
    get_text_embedding,
    get_video_embedding,
)

__all__ = [
    "MULTIMODAL_EMBEDDING_MODEL",
    "EmbeddingError",
    "get_image_embedding",
    "get_text_embedding",
    "get_video_embedding",
]
