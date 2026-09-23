"""Fixed RAG ask pipeline — AnswerDTO with retrieval-only sources (SPEC §7.1 / §11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.embedding.mm_dashscope import get_text_embedding
from src.intent.media_intent import detect_media_intent
from src.rag.llm import call_qwen_flash
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from src.rag.sources import format_sources
from src.retrieve.search import (
    DEFAULT_TEXT_TOP_K,
    EmbedFn,
    search_with_details,
    select_media_hit,
    select_text_hits,
)
from src.store.faiss_store import DEFAULT_INDEX_DIR, load_index
from src.types import AnswerDTO, ImageMeta, Metadata, VideoMeta

LlmFn = Callable[[str, str], str]


def _meta_to_image(meta: dict[str, Any]) -> ImageMeta:
    return ImageMeta(
        id=int(meta["id"]),
        type="image",
        source=str(meta.get("source", "")),
        content=str(meta.get("content", "")),
        path=str(meta.get("path", "")),
    )


def _meta_to_video(meta: dict[str, Any]) -> VideoMeta:
    return VideoMeta(
        id=int(meta["id"]),
        type="video",
        source=str(meta.get("source", "")),
        content=str(meta.get("content", "")),
        url=str(meta.get("url", "")),
        description=str(meta.get("description", "")),
    )


def ask(
    query: str,
    k: int = DEFAULT_TEXT_TOP_K,
    *,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    index: Any | None = None,
    metadata: list[dict[str, Any]] | None = None,
    embed_fn: EmbedFn | None = None,
    llm_fn: LlmFn | None = None,
) -> AnswerDTO:
    """Retrieve → intent → LLM answer; sources/media refs never parsed from LLM text."""
    if not str(query).strip():
        raise ValueError("query must be a non-empty string")

    if index is None or metadata is None:
        index, metadata, _manifest = load_index(index_dir)

    embed = embed_fn or get_text_embedding
    llm = llm_fn or call_qwen_flash

    hits = search_with_details(
        query,
        index,
        metadata,
        embed_fn=embed,
    )
    text_hits = select_text_hits(hits, k=k)
    sources = format_sources(text_hits)

    want_image, want_video = detect_media_intent(query)
    image_hit = select_media_hit(
        hits, media_type="image", want=want_image, query=query
    )
    video_hit = select_media_hit(
        hits, media_type="video", want=want_video, query=query
    )

    image_ref: Metadata | None = (
        _meta_to_image(image_hit.metadata) if image_hit is not None else None
    )
    video_ref: Metadata | None = (
        _meta_to_video(video_hit.metadata) if video_hit is not None else None
    )

    user_prompt = build_user_prompt(
        query,
        text_hits,
        image_hit=image_hit,
        video_hit=video_hit,
    )
    answer = llm(SYSTEM_PROMPT, user_prompt)

    # Hard rule: never reassign sources/media from answer text.
    return AnswerDTO(
        answer=answer,
        sources=sources,
        image_ref=image_ref,
        video_ref=video_ref,
    )
