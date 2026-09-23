"""SPEC §15.1 hard assertion helpers for e2e cases (not weak answer/sources-only checks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.types import AnswerDTO, ImageMeta, VideoMeta

_FORBIDDEN_VIDEO_MARKERS = ("car.mp4", "汽车剐蹭", "剐蹭", "/car.")


class CaseAssertionError(AssertionError):
    """Raised when a §15.1 hard assertion fails."""


def normalize_source_path(path: str) -> str:
    """Normalize source/path: unify separators and strip leading data/."""
    p = str(path).replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    if p.startswith("data/"):
        p = p[len("data/") :]
    return p


def _sources_match_any(dto: AnswerDTO, expected_any_of: list[str] | None) -> bool:
    if not expected_any_of:
        return True
    got = [normalize_source_path(s.source) for s in dto.sources]
    for exp in expected_any_of:
        exp_n = normalize_source_path(exp)
        for g in got:
            if g == exp_n or Path(g).name == Path(exp_n).name:
                return True
    return False


def _forbid_car_video(video: VideoMeta) -> None:
    blob = f"{video.url} {video.description}".lower()
    for marker in _FORBIDDEN_VIDEO_MARKERS:
        if marker.lower() in blob or marker in f"{video.url} {video.description}":
            raise CaseAssertionError(
                f"forbidden demo video content detected ({marker!r})"
            )


def assert_e2e_case(
    dto: AnswerDTO,
    case: dict[str, Any],
    *,
    video_catalog: list[dict[str, Any]],
) -> None:
    """Apply SPEC §15.1 hard rules for one case. Raises CaseAssertionError on fail."""
    case_id = case.get("id", "?")

    if not str(dto.answer).strip():
        raise CaseAssertionError(f"{case_id}: answer must be non-empty")
    if not dto.sources:
        raise CaseAssertionError(f"{case_id}: sources[] must be non-empty")

    expected_any = case.get("expected_sources_any_of") or []
    if expected_any and not _sources_match_any(dto, list(expected_any)):
        got = [normalize_source_path(s.source) for s in dto.sources]
        raise CaseAssertionError(
            f"{case_id}: sources {got} miss expected_sources_any_of {expected_any}"
        )

    require_image = bool(case.get("require_image_ref"))
    require_video = bool(case.get("require_video_ref"))
    expected_image = case.get("expected_image_path")
    expected_video_id = case.get("expected_video_id")

    # T1-style: no media intent expected → both refs must be null
    if not require_image and not require_video:
        if dto.image_ref is not None or dto.video_ref is not None:
            raise CaseAssertionError(
                f"{case_id}: image_ref/video_ref must be null when media not required"
            )

    if require_image:
        if dto.image_ref is None or not isinstance(dto.image_ref, ImageMeta):
            raise CaseAssertionError(f"{case_id}: require_image_ref but image_ref is null")
        got_path = normalize_source_path(dto.image_ref.path)
        exp_path = normalize_source_path(str(expected_image or ""))
        if got_path != exp_path:
            raise CaseAssertionError(
                f"{case_id}: image_ref.path {got_path!r} != expected {exp_path!r}"
            )

    if require_video:
        if dto.video_ref is None or not isinstance(dto.video_ref, VideoMeta):
            raise CaseAssertionError(f"{case_id}: require_video_ref but video_ref is null")
        _forbid_car_video(dto.video_ref)
        if not expected_video_id:
            raise CaseAssertionError(f"{case_id}: expected_video_id missing in case")
        match = next(
            (v for v in video_catalog if str(v.get("id")) == str(expected_video_id)),
            None,
        )
        if match is None:
            raise CaseAssertionError(
                f"{case_id}: video_catalog missing id={expected_video_id}"
            )
        if dto.video_ref.url.strip() != str(match["url"]).strip():
            raise CaseAssertionError(
                f"{case_id}: video_ref.url {dto.video_ref.url!r} "
                f"!= catalog url for {expected_video_id}"
            )
        # description should not be car-like; catalog row also checked
        _forbid_car_video(
            VideoMeta(
                id=0,
                type="video",
                source="",
                content="",
                url=str(match["url"]),
                description=str(match.get("description", "")),
            )
        )


def assert_cross_sample_gate(media_success: dict[str, bool]) -> None:
    """At least one image success (T2/T4) and one video success (T3) across T1–T4."""
    image_ok = bool(media_success.get("T2")) or bool(media_success.get("T4"))
    video_ok = bool(media_success.get("T3"))
    if not image_ok:
        raise CaseAssertionError(
            "cross-sample gate failed: need at least one successful image ref (T2 or T4)"
        )
    if not video_ok:
        raise CaseAssertionError(
            "cross-sample gate failed: need at least one successful video ref (T3)"
        )


def case_media_success(case_id: str, dto: AnswerDTO, case: dict[str, Any]) -> bool:
    """Whether this case contributed a successful required media citation."""
    if case_id in {"T2", "T4"} and case.get("require_image_ref"):
        return dto.image_ref is not None
    if case_id == "T3" and case.get("require_video_ref"):
        return dto.video_ref is not None
    return False
