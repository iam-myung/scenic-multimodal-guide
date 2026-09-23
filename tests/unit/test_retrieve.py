"""Unit tests for retrieve — injected vectors only, no real Embedding API."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieve.search import (
    DEFAULT_TEXT_TOP_K,
    distance_to_similarity,
    search_with_details,
    select_media_hit,
    select_text_hits,
)
from src.store.faiss_store import save_index
from src.types import IndexManifest


def _tiny_index(tmp_path):
    """3 vectors in 2-D: text near [0,0], image near [1,0], video near [0,1]."""
    vectors = np.array(
        [
            [0.0, 0.0],  # text 0
            [0.1, 0.0],  # text 1
            [0.2, 0.0],  # text 2
            [1.0, 0.0],  # image
            [0.0, 1.0],  # video
        ],
        dtype=np.float32,
    )
    metadata = [
        {
            "id": 0,
            "type": "text",
            "source": "a.docx",
            "content": "门票规则A",
            "chunk_id": 0,
        },
        {
            "id": 1,
            "type": "text",
            "source": "b.docx",
            "content": "攻略B",
            "chunk_id": 0,
        },
        {
            "id": 2,
            "type": "text",
            "source": "c.docx",
            "content": "会员C",
            "chunk_id": 0,
        },
        {
            "id": 3,
            "type": "image",
            "source": "图片: x.jpg",
            "content": "[图片] x.jpg",
            "path": "knowledge_base/images/x.jpg",
        },
        {
            "id": 4,
            "type": "video",
            "source": "视频: demo",
            "content": "[视频] demo",
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
        counts={"text": 3, "image": 1, "video": 1},
        faiss_index_path="scenic_index.faiss",
        metadata_path="scenic_metadata.json",
    )
    index_dir = tmp_path / "index"
    save_index(vectors, metadata, manifest, index_dir)

    from src.store.faiss_store import load_index

    index, meta, _ = load_index(index_dir)
    return index, meta


def test_distance_to_similarity() -> None:
    assert distance_to_similarity(0.0) == 1.0
    assert distance_to_similarity(1.0) == pytest.approx(0.5)
    assert distance_to_similarity(3.0) == pytest.approx(0.25)


def test_search_with_details_full_l2_and_similarity(tmp_path) -> None:
    index, metadata = _tiny_index(tmp_path)
    # Query near origin → text hits should rank first
    query_vec = np.array([0.0, 0.0], dtype=np.float32)

    hits = search_with_details(
        query="ignored when vector injected",
        index=index,
        metadata=metadata,
        query_vector=query_vec,
    )

    assert len(hits) == index.ntotal
    assert hits[0].id == 0
    assert hits[0].metadata["type"] == "text"
    assert hits[0].similarity == pytest.approx(
        distance_to_similarity(hits[0].distance)
    )
    # Full-corpus search: distances non-decreasing
    dists = [h.distance for h in hits]
    assert dists == sorted(dists)


def test_select_text_hits_default_top_k(tmp_path) -> None:
    assert DEFAULT_TEXT_TOP_K == 3
    index, metadata = _tiny_index(tmp_path)
    hits = search_with_details(
        query="q",
        index=index,
        metadata=metadata,
        query_vector=np.array([0.0, 0.0], dtype=np.float32),
    )
    text_hits = select_text_hits(hits, k=DEFAULT_TEXT_TOP_K)
    assert len(text_hits) == 3
    assert all(h.metadata["type"] == "text" for h in text_hits)


def test_select_media_hit_respects_intent_and_threshold(tmp_path) -> None:
    index, metadata = _tiny_index(tmp_path)
    # Query near image vector
    hits = search_with_details(
        query="q",
        index=index,
        metadata=metadata,
        query_vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    image = select_media_hit(hits, media_type="image", want=True, threshold=3.0)
    assert image is not None
    assert image.metadata["path"].endswith("x.jpg")
    assert image.distance < 3.0

    # No intent → no media
    assert (
        select_media_hit(hits, media_type="image", want=False, threshold=3.0) is None
    )

    # Threshold too tight → None
    assert (
        select_media_hit(hits, media_type="image", want=True, threshold=0.0) is None
    )


def _t4_near_tie_hits() -> list:
    """Reproduce e2e T4: both images under threshold; Halloween slightly closer."""
    from src.retrieve.search import SearchHit

    halloween = SearchHit(
        id=0,
        distance=1.639,
        similarity=distance_to_similarity(1.639),
        metadata={
            "id": 0,
            "type": "image",
            "source": "图片: 2-万圣节.jpeg",
            "content": "[图片] 2-万圣节.jpeg",
            "path": "knowledge_base/images/2-万圣节.jpeg",
        },
    )
    wonder = SearchHit(
        id=1,
        distance=1.662,
        similarity=distance_to_similarity(1.662),
        metadata={
            "id": 1,
            "type": "image",
            "source": "图片: 1-园区合影.jpg",
            "content": "[图片] 1-园区合影.jpg",
            "path": "knowledge_base/images/1-园区合影.jpg",
        },
    )
    return [halloween, wonder]


def test_select_media_hit_t4_prefers_path_label_over_near_tie() -> None:
    """SPEC §15.1 T4: query label must beat slightly closer wrong image."""
    hits = _t4_near_tie_hits()
    chosen = select_media_hit(
        hits,
        media_type="image",
        want=True,
        threshold=3.0,
        query="园区合影的海报",
    )
    assert chosen is not None
    assert chosen.metadata["path"] == "knowledge_base/images/1-园区合影.jpg"


def test_select_media_hit_t2_still_picks_halloween() -> None:
    """SPEC §15.1 T2 must remain Halloween when query names it."""
    hits = _t4_near_tie_hits()
    # Invert distances so wonder is closer — label affinity must still win for T2
    from src.retrieve.search import SearchHit

    wonder_closer = [
        SearchHit(
            id=1,
            distance=1.50,
            similarity=distance_to_similarity(1.50),
            metadata=hits[1].metadata,
        ),
        SearchHit(
            id=0,
            distance=1.60,
            similarity=distance_to_similarity(1.60),
            metadata=hits[0].metadata,
        ),
    ]
    chosen = select_media_hit(
        wonder_closer,
        media_type="image",
        want=True,
        threshold=3.0,
        query="最近万圣节的活动海报是什么",
    )
    assert chosen is not None
    assert chosen.metadata["path"] == "knowledge_base/images/2-万圣节.jpeg"


def test_search_with_details_requires_vector_or_embed_fn(tmp_path) -> None:
    index, metadata = _tiny_index(tmp_path)
    with pytest.raises(ValueError):
        search_with_details(query="门票", index=index, metadata=metadata)


def test_search_with_details_uses_injectable_embed_fn(tmp_path) -> None:
    index, metadata = _tiny_index(tmp_path)

    def fake_embed(text: str) -> list[float]:
        assert text == "攻略"
        return [0.05, 0.0]

    hits = search_with_details(
        query="攻略",
        index=index,
        metadata=metadata,
        embed_fn=fake_embed,
    )
    assert hits[0].metadata["type"] == "text"
