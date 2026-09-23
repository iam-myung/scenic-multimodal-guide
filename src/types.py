"""Shared domain types — SPEC §10–§11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


@dataclass(frozen=True)
class TextMeta:
    id: int
    type: Literal["text"]
    source: str
    content: str
    chunk_id: int


@dataclass(frozen=True)
class ImageMeta:
    id: int
    type: Literal["image"]
    source: str
    content: str
    path: str


@dataclass(frozen=True)
class VideoMeta:
    id: int
    type: Literal["video"]
    source: str
    content: str
    url: str
    description: str


Metadata = Union[TextMeta, ImageMeta, VideoMeta]


@dataclass(frozen=True)
class IndexManifest:
    embedding_model: str
    dim: int
    chunk_size: int
    chunk_overlap: int
    built_at: str
    counts: dict[str, int]
    faiss_index_path: str
    metadata_path: str


@dataclass(frozen=True)
class SourceRef:
    source: str
    chunk_id: int
    similarity: float
    content_preview: str


@dataclass(frozen=True)
class AnswerDTO:
    answer: str
    sources: list[SourceRef]
    image_ref: Metadata | None = None
    video_ref: Metadata | None = None
