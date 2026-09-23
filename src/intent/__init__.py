"""Media intent package."""

from src.intent.media_intent import (
    IMAGE_KEYWORDS,
    MEDIA_DISTANCE_THRESHOLD,
    VIDEO_KEYWORDS,
    detect_media_intent,
)

__all__ = [
    "IMAGE_KEYWORDS",
    "MEDIA_DISTANCE_THRESHOLD",
    "VIDEO_KEYWORDS",
    "detect_media_intent",
]
