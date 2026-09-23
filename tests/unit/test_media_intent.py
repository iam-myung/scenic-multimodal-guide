"""Unit tests for media intent keywords (SPEC §8.3 / 基座 keywords)."""

from __future__ import annotations

from src.intent.media_intent import (
    IMAGE_KEYWORDS,
    MEDIA_DISTANCE_THRESHOLD,
    VIDEO_KEYWORDS,
    detect_media_intent,
)


def test_keywords_and_threshold_match_spec() -> None:
    assert MEDIA_DISTANCE_THRESHOLD == 3.0
    assert IMAGE_KEYWORDS == ["图片", "海报", "照片", "看看", "长什么样", "图"]
    assert VIDEO_KEYWORDS == ["视频", "录像", "影片", "看一下", "播放"]


def test_detect_media_intent_image_only() -> None:
    want_image, want_video = detect_media_intent("园区海报长什么样")
    assert want_image is True
    assert want_video is False


def test_detect_media_intent_video_only() -> None:
    want_image, want_video = detect_media_intent("有没有演示视频可以看一下")
    assert want_image is False
    assert want_video is True


def test_detect_media_intent_both_and_neither() -> None:
    both = detect_media_intent("看看园区视频播放")
    assert both == (True, True)
    neither = detect_media_intent("门票规则是什么")
    assert neither == (False, False)
