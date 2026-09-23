"""Unit tests for ingest.chunk — SPEC CHUNK_SIZE=500, OVERLAP=50."""

from __future__ import annotations

from src.ingest.chunk import CHUNK_OVERLAP, CHUNK_SIZE, split_text


def test_constants_match_spec() -> None:
    assert CHUNK_SIZE == 500
    assert CHUNK_OVERLAP == 50


def test_split_empty_text_returns_empty_list() -> None:
    assert split_text("") == []
    assert split_text("   \n\t  ") == []


def test_split_short_text_single_chunk() -> None:
    text = "short scenic note"
    assert split_text(text) == ["short scenic note"]


def test_split_overlap_boundary() -> None:
    """Length > CHUNK_SIZE: second chunk overlaps previous by CHUNK_OVERLAP."""
    text = "A" * 550
    chunks = split_text(text)
    assert len(chunks) == 2
    assert chunks[0] == "A" * 500
    assert chunks[1] == "A" * 100
    # Overlap: last OVERLAP chars of first == first OVERLAP chars of second
    assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]


def test_split_exact_chunk_size_still_emits_overlap_tail() -> None:
    text = "B" * CHUNK_SIZE
    chunks = split_text(text)
    assert chunks[0] == "B" * CHUNK_SIZE
    assert chunks[1] == "B" * CHUNK_OVERLAP
