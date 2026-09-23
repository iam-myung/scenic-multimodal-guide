"""Keyword-based media intent detection (SPEC §8.3; 18_ keywords)."""

from __future__ import annotations

IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]
MEDIA_DISTANCE_THRESHOLD = 3.0


def detect_media_intent(query: str) -> tuple[bool, bool]:
    """Return (want_image, want_video) from substring keyword match."""
    query_lower = query.lower()
    want_image = any(kw in query_lower for kw in IMAGE_KEYWORDS)
    want_video = any(kw in query_lower for kw in VIDEO_KEYWORDS)
    return want_image, want_video
