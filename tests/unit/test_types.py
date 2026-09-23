"""Unit tests for src.types — SPEC §10–§11 schemas."""

from __future__ import annotations

from src.types import AnswerDTO, IndexManifest, SourceRef, TextMeta


def test_source_ref_and_answer_dto_fields() -> None:
    ref = SourceRef(
        source="knowledge_base/demo.docx",
        chunk_id=0,
        similarity=0.85,
        content_preview="preview",
    )
    dto = AnswerDTO(answer="hello", sources=[ref], image_ref=None, video_ref=None)
    assert dto.answer == "hello"
    assert len(dto.sources) == 1
    assert dto.sources[0].chunk_id == 0


def test_index_manifest_and_text_meta_fields() -> None:
    meta = TextMeta(
        id=0,
        type="text",
        source="knowledge_base/demo.docx",
        content="chunk text",
        chunk_id=0,
    )
    manifest = IndexManifest(
        embedding_model="tongyi-embedding-vision-plus",
        dim=1024,
        chunk_size=500,
        chunk_overlap=50,
        built_at="2026-01-01T00:00:00Z",
        counts={"text": 1, "image": 0, "video": 0},
        faiss_index_path="data/index/scenic_index.faiss",
        metadata_path="data/index/scenic_metadata.json",
    )
    assert meta.id == 0
    assert manifest.counts["text"] == 1
