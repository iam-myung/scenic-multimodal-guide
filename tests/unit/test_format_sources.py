"""Unit tests for format_sources — sources only from text hits (SPEC §11.2)."""

from __future__ import annotations

from src.retrieve.search import SearchHit
from src.rag.sources import format_sources


def test_format_sources_from_text_hits_only() -> None:
    hits = [
        SearchHit(
            id=0,
            distance=0.0,
            similarity=1.0,
            metadata={
                "id": 0,
                "type": "text",
                "source": "a.docx",
                "content": "门票规则全文" + ("X" * 200),
                "chunk_id": 2,
            },
        ),
        SearchHit(
            id=1,
            distance=0.5,
            similarity=1 / 1.5,
            metadata={
                "id": 1,
                "type": "text",
                "source": "b.docx",
                "content": "攻略短文",
                "chunk_id": 0,
            },
        ),
    ]
    refs = format_sources(hits)
    assert len(refs) == 2
    assert refs[0].source == "a.docx"
    assert refs[0].chunk_id == 2
    assert refs[0].similarity == 1.0
    assert refs[0].content_preview.startswith("门票规则全文")
    assert len(refs[0].content_preview) <= 120
    assert refs[1].source == "b.docx"
    assert refs[1].content_preview == "攻略短文"


def test_format_sources_empty() -> None:
    assert format_sources([]) == []
