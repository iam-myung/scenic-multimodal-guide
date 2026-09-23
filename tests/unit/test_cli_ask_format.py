"""Unit tests for CLI ask display formatting (SPEC §11.2 display rules)."""

from __future__ import annotations

from src.types import AnswerDTO, ImageMeta, SourceRef, VideoMeta


def test_format_ask_output_includes_evidence_block() -> None:
    from src.cli import format_ask_output

    dto = AnswerDTO(
        answer="一日票可提前修改。",
        sources=[
            SourceRef(
                source="knowledge_base/1-某大型游乐园门票规则.docx",
                chunk_id=0,
                similarity=0.8,
                content_preview="提前48小时可免费修改",
            )
        ],
        image_ref=None,
        video_ref=None,
    )
    text = format_ask_output(dto)
    assert "一日票可提前修改。" in text
    assert "[检索证据]" in text
    assert "knowledge_base/1-某大型游乐园门票规则.docx" in text
    assert "[相关图片]" not in text
    assert "[相关视频]" not in text


def test_format_ask_output_includes_optional_media() -> None:
    from src.cli import format_ask_output

    dto = AnswerDTO(
        answer="这是海报。",
        sources=[
            SourceRef(
                source="knowledge_base/3-某大型游乐园游玩攻略清单.docx",
                chunk_id=0,
                similarity=0.5,
                content_preview="攻略",
            )
        ],
        image_ref=ImageMeta(
            id=9,
            type="image",
            source="图片: 2-万圣节.jpeg",
            content="[图片] 2-万圣节.jpeg",
            path="knowledge_base/images/2-万圣节.jpeg",
        ),
        video_ref=VideoMeta(
            id=11,
            type="video",
            source="视频: demo",
            content="[视频] demo",
            url="https://example.com/scenic-demo.mp4",
            description="景区演示视频",
        ),
    )
    text = format_ask_output(dto)
    assert "[相关图片]" in text
    assert "knowledge_base/images/2-万圣节.jpeg" in text
    assert "[相关视频]" in text
    assert "https://example.com/scenic-demo.mp4" in text
