"""Request/response schemas aligned with AnswerDTO (SPEC §21.2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1, le=20)


class SourceRefOut(BaseModel):
    source: str
    chunk_id: int
    similarity: float
    content_preview: str


class ImageRefOut(BaseModel):
    id: int
    type: Literal["image"] = "image"
    source: str
    content: str
    path: str


class VideoRefOut(BaseModel):
    id: int
    type: Literal["video"] = "video"
    source: str
    content: str
    url: str
    description: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRefOut]
    image_ref: ImageRefOut | None = None
    video_ref: VideoRefOut | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class StatusResponse(BaseModel):
    index_ready: bool
    index_dir: str
    hint: str = ""


class BuildResponse(BaseModel):
    ok: bool
    message: str
    counts: dict[str, int] | None = None


class ErrorBody(BaseModel):
    detail: str


def meta_to_image_out(meta: Any) -> ImageRefOut:
    return ImageRefOut(
        id=int(meta.id),
        source=str(meta.source),
        content=str(meta.content),
        path=str(meta.path),
    )


def meta_to_video_out(meta: Any) -> VideoRefOut:
    return VideoRefOut(
        id=int(meta.id),
        source=str(meta.source),
        content=str(meta.content),
        url=str(meta.url),
        description=str(meta.description),
    )
