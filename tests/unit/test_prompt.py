"""Unit tests for RAG prompt assembly (SPEC §11.1)."""

from __future__ import annotations

from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from src.retrieve.search import SearchHit, distance_to_similarity


def _hit(source: str, content: str, chunk_id: int = 0, distance: float = 0.5) -> SearchHit:
    return SearchHit(
        id=chunk_id,
        distance=distance,
        similarity=distance_to_similarity(distance),
        metadata={
            "id": chunk_id,
            "type": "text",
            "source": source,
            "content": content,
            "chunk_id": chunk_id,
        },
    )


def test_system_prompt_matches_spec() -> None:
    assert "仅根据给定背景知识回答" in SYSTEM_PROMPT
    assert "不知则说明知识库未覆盖" in SYSTEM_PROMPT
    assert "演示知识不代表官方实时信息" in SYSTEM_PROMPT


def test_build_user_prompt_anti_false_refusal_when_hits_present() -> None:
    """Q5-class: retrieved VIP/会员 chunks must not be denied as uncovered."""
    hits = [
        _hit(
            "knowledge_base/4-某大型游乐园酒店会员制度.docx",
            "某大型游乐园酒店会员制度\n首先，园区配套酒店的尊享服务大致分为三种，一种是尊享卡……",
        ),
        _hit(
            "knowledge_base/4-某大型游乐园酒店会员制度.docx",
            "礼宾服务还能享受酒店联动礼遇与观演指引……",
            chunk_id=1,
            distance=0.7,
        ),
    ]
    prompt = build_user_prompt("某大型游乐园酒店会员有什么权益？", hits)
    assert "尊享卡" in prompt
    assert "[用户问题]" in prompt
    assert "某大型游乐园酒店会员有什么权益？" in prompt
    # Must steer model away from false "未覆盖" when background exists
    assert "不要" in prompt and "知识库未覆盖" in prompt
    assert "来源文件名" in prompt or "文首标题" in prompt
    assert "尊享卡" in prompt  # retrieved content still present
    assert "酒店会员" in prompt


def test_build_user_prompt_still_allows_uncovered_when_no_hits() -> None:
    prompt = build_user_prompt("任意问题", [])
    assert "本次检索未命中文本知识" in prompt
    assert "知识库未覆盖" in prompt


def test_build_user_prompt_includes_selected_image_clue() -> None:
    """Q7/Q8: selected image must appear in LLM user prompt."""
    image_hit = SearchHit(
        id=10,
        distance=1.2,
        similarity=distance_to_similarity(1.2),
        metadata={
            "id": 10,
            "type": "image",
            "source": "图片: 2-万圣节.jpeg",
            "content": "[图片] 2-万圣节.jpeg",
            "path": "knowledge_base/images/2-万圣节.jpeg",
        },
    )
    prompt = build_user_prompt(
        "最近万圣节的活动海报是什么",
        [],
        image_hit=image_hit,
    )
    assert "[已检索媒体]" in prompt
    assert "knowledge_base/images/2-万圣节.jpeg" in prompt
    assert "万圣节" in prompt
    assert "不要声称知识库未覆盖" in prompt or "不要" in prompt


def test_build_user_prompt_includes_selected_video_clue() -> None:
    """Q9: selected video must appear in LLM user prompt."""
    video_hit = SearchHit(
        id=11,
        distance=1.0,
        similarity=distance_to_similarity(1.0),
        metadata={
            "id": 11,
            "type": "video",
            "source": "视频: scenic_demo_1",
            "content": "[视频] scenic_demo_1",
            "url": "https://assets.mixkit.co/videos/8058/8058-720.mp4",
            "description": "景区相关演示视频（技术演示）",
        },
    )
    prompt = build_user_prompt(
        "有没有景区相关的演示视频可以看一下",
        [],
        video_hit=video_hit,
    )
    assert "[已检索媒体]" in prompt
    assert "https://assets.mixkit.co/videos/8058/8058-720.mp4" in prompt
    assert "演示视频" in prompt


def test_build_user_prompt_omits_media_section_when_none() -> None:
    prompt = build_user_prompt("一日票规则", [_hit("a.docx", "改期规则")])
    assert "[已检索媒体]" not in prompt
