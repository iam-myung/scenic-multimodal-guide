"""Integration tests for rag.ask — mock Embedding + LLM (SPEC §11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.store.faiss_store import save_index
from src.types import AnswerDTO, ImageMeta, IndexManifest, SourceRef, VideoMeta


def _build_mini_index(tmp_path: Path):
    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = [
        {
            "id": 0,
            "type": "text",
            "source": "1-某大型游乐园门票规则.docx",
            "content": "一日票可在购票时选定日期使用。",
            "chunk_id": 0,
        },
        {
            "id": 1,
            "type": "text",
            "source": "3-某大型游乐园游玩攻略清单.docx",
            "content": "游玩攻略建议提前规划路线。",
            "chunk_id": 0,
        },
        {
            "id": 2,
            "type": "image",
            "source": "图片: 2-万圣节.jpeg",
            "content": "[图片] 2-万圣节.jpeg",
            "path": "knowledge_base/images/2-万圣节.jpeg",
        },
        {
            "id": 3,
            "type": "video",
            "source": "视频: 景区演示",
            "content": "[视频] 景区演示",
            "url": "https://example.com/scenic-demo.mp4",
            "description": "景区演示",
        },
    ]
    manifest = IndexManifest(
        embedding_model="test",
        dim=2,
        chunk_size=500,
        chunk_overlap=50,
        built_at="2026-01-01T00:00:00+00:00",
        counts={"text": 2, "image": 1, "video": 1},
        faiss_index_path="scenic_index.faiss",
        metadata_path="scenic_metadata.json",
    )
    index_dir = tmp_path / "index"
    save_index(vectors, metadata, manifest, index_dir)
    return index_dir


def test_ask_sources_from_retrieval_not_llm_text(tmp_path: Path) -> None:
    from src.rag.ask import ask

    index_dir = _build_mini_index(tmp_path)

    def fake_embed(text: str) -> list[float]:
        return [0.0, 0.0]

    def fake_llm(system: str, user: str) -> str:
        assert "景区" in system
        # Deliberately mention a fake source that must NOT enter sources[]
        return (
            "根据知识库，一日票需选定日期。"
            "参考来源：伪造文档.pdf；图片：fake.png"
        )

    dto = ask(
        "一日票怎么用",
        k=2,
        index_dir=index_dir,
        embed_fn=fake_embed,
        llm_fn=fake_llm,
    )

    assert isinstance(dto, AnswerDTO)
    assert "一日票" in dto.answer or "选定日期" in dto.answer
    assert dto.sources
    assert all(isinstance(s, SourceRef) for s in dto.sources)
    sources_joined = " ".join(s.source for s in dto.sources)
    assert "伪造文档.pdf" not in sources_joined
    assert "fake.png" not in sources_joined
    assert any("门票规则" in s.source for s in dto.sources)
    assert dto.image_ref is None
    assert dto.video_ref is None


def test_ask_media_refs_only_from_retrieval_filter(tmp_path: Path) -> None:
    from src.rag.ask import ask

    index_dir = _build_mini_index(tmp_path)

    def fake_embed(text: str) -> list[float]:
        # Near image vector
        return [1.0, 0.0]

    def fake_llm(system: str, user: str) -> str:
        assert "[已检索媒体]" in user
        assert "knowledge_base/images/2-万圣节.jpeg" in user
        return "这是万圣节氛围相关说明。另附视频 http://evil.example/car.mp4"

    dto = ask(
        "万圣节海报长什么样看看图片",
        k=2,
        index_dir=index_dir,
        embed_fn=fake_embed,
        llm_fn=fake_llm,
    )

    assert dto.image_ref is not None
    assert isinstance(dto.image_ref, ImageMeta)
    assert dto.image_ref.path == "knowledge_base/images/2-万圣节.jpeg"
    # No video intent → video_ref stays None despite LLM mentioning a video URL
    assert dto.video_ref is None


def test_ask_video_intent_attaches_video_ref(tmp_path: Path) -> None:
    from src.rag.ask import ask

    index_dir = _build_mini_index(tmp_path)

    def fake_embed(text: str) -> list[float]:
        return [0.0, 1.0]

    def fake_llm(system: str, user: str) -> str:
        assert "[已检索媒体]" in user
        assert "https://example.com/scenic-demo.mp4" in user
        return "请观看演示视频。"

    dto = ask(
        "有没有景区演示视频可以看一下",
        k=2,
        index_dir=index_dir,
        embed_fn=fake_embed,
        llm_fn=fake_llm,
    )
    assert dto.video_ref is not None
    assert isinstance(dto.video_ref, VideoMeta)
    assert dto.video_ref.url == "https://example.com/scenic-demo.mp4"
    assert dto.image_ref is None
