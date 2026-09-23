"""Unit tests for §15.1 hard assertion helpers (no real API)."""

from __future__ import annotations

import pytest

from src.types import AnswerDTO, ImageMeta, SourceRef, VideoMeta
from tests.e2e.assert_rules import (
    CaseAssertionError,
    assert_cross_sample_gate,
    assert_e2e_case,
    normalize_source_path,
)


def _dto(**kwargs) -> AnswerDTO:
    base = dict(
        answer="ok",
        sources=[
            SourceRef(
                source="knowledge_base/1-某大型游乐园门票规则.docx",
                chunk_id=0,
                similarity=0.9,
                content_preview="提前48小时",
            )
        ],
        image_ref=None,
        video_ref=None,
    )
    base.update(kwargs)
    return AnswerDTO(**base)  # type: ignore[arg-type]


def test_normalize_source_strips_data_prefix() -> None:
    assert (
        normalize_source_path("data/knowledge_base/1-某大型游乐园门票规则.docx")
        == "knowledge_base/1-某大型游乐园门票规则.docx"
    )
    assert (
        normalize_source_path("knowledge_base/1-某大型游乐园门票规则.docx")
        == "knowledge_base/1-某大型游乐园门票规则.docx"
    )


def test_assert_case_t1_rejects_media_and_wrong_source() -> None:
    case = {
        "id": "T1",
        "query": "q",
        "expected_sources_any_of": ["knowledge_base/1-某大型游乐园门票规则.docx"],
        "expected_image_path": None,
        "expected_video_id": None,
        "require_image_ref": False,
        "require_video_ref": False,
    }
    assert_e2e_case(_dto(), case, video_catalog=[])

    wrong = _dto(
        sources=[
            SourceRef(
                source="knowledge_base/2-某大型游乐园老人票价规定.docx",
                chunk_id=0,
                similarity=0.9,
                content_preview="x",
            )
        ]
    )
    with pytest.raises(CaseAssertionError):
        assert_e2e_case(wrong, case, video_catalog=[])

    with_media = _dto(
        image_ref=ImageMeta(
            id=1,
            type="image",
            source="img",
            content="c",
            path="knowledge_base/images/2-万圣节.jpeg",
        )
    )
    with pytest.raises(CaseAssertionError):
        assert_e2e_case(with_media, case, video_catalog=[])


def test_assert_case_t2_requires_exact_image_path() -> None:
    case = {
        "id": "T2",
        "query": "q",
        "expected_sources_any_of": [],
        "expected_image_path": "knowledge_base/images/2-万圣节.jpeg",
        "expected_video_id": None,
        "require_image_ref": True,
        "require_video_ref": False,
    }
    ok = _dto(
        image_ref=ImageMeta(
            id=1,
            type="image",
            source="图片",
            content="c",
            path="knowledge_base/images/2-万圣节.jpeg",
        )
    )
    assert_e2e_case(ok, case, video_catalog=[])

    bad = _dto(
        image_ref=ImageMeta(
            id=1,
            type="image",
            source="图片",
            content="c",
            path="knowledge_base/images/1-园区合影.jpg",
        )
    )
    with pytest.raises(CaseAssertionError):
        assert_e2e_case(bad, case, video_catalog=[])


def test_assert_case_t3_video_id_and_forbids_car() -> None:
    catalog = [
        {
            "id": "scenic_demo_1",
            "url": "https://example.com/scenic-demo.mp4",
            "description": "景区演示视频",
        }
    ]
    case = {
        "id": "T3",
        "query": "q",
        "expected_sources_any_of": [],
        "expected_image_path": None,
        "expected_video_id": "scenic_demo_1",
        "require_image_ref": False,
        "require_video_ref": True,
    }
    ok = _dto(
        video_ref=VideoMeta(
            id=11,
            type="video",
            source="视频",
            content="c",
            url="https://example.com/scenic-demo.mp4",
            description="景区演示视频",
        )
    )
    assert_e2e_case(ok, case, video_catalog=catalog)

    car = _dto(
        video_ref=VideoMeta(
            id=11,
            type="video",
            source="视频",
            content="c",
            url="https://example.com/car.mp4",
            description="汽车剐蹭",
        )
    )
    with pytest.raises(CaseAssertionError):
        assert_e2e_case(car, case, video_catalog=catalog)


def test_cross_sample_gate_requires_image_and_video_success() -> None:
    assert_cross_sample_gate({"T2": True, "T3": True, "T4": False})
    with pytest.raises(CaseAssertionError):
        assert_cross_sample_gate({"T2": False, "T3": True, "T4": False})
    with pytest.raises(CaseAssertionError):
        assert_cross_sample_gate({"T2": True, "T3": False, "T4": True})


def test_weak_assertion_alone_is_insufficient() -> None:
    case = {
        "id": "T1",
        "query": "q",
        "expected_sources_any_of": ["knowledge_base/1-某大型游乐园门票规则.docx"],
        "expected_image_path": None,
        "expected_video_id": None,
        "require_image_ref": False,
        "require_video_ref": False,
    }
    weak_ok_shape = _dto(
        answer="随便回答",
        sources=[
            SourceRef(
                source="knowledge_base/3-某大型游乐园游玩攻略清单.docx",
                chunk_id=0,
                similarity=0.99,
                content_preview="攻略",
            )
        ],
    )
    with pytest.raises(CaseAssertionError):
        assert_e2e_case(weak_ok_shape, case, video_catalog=[])
