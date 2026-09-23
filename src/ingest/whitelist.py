"""Corpus whitelist loading and validation (SPEC §5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AUTHORIZATIONS = frozenset(
    {
        "owner_original",
        "permission_granted",
        "licensed_public",
        "restricted_local",
    }
)
MODALITIES = frozenset({"text", "image", "video_ref"})

_FORBIDDEN_URL_MARKERS = ("car.mp4",)
_FORBIDDEN_DESC_MARKERS = ("汽车剐蹭", "剐蹭")


class WhitelistError(ValueError):
    """Raised when whitelist or video config violates SPEC rules."""


def load_whitelist(path: str | Path) -> list[dict[str, Any]]:
    """Load corpus_whitelist.json and validate schema fields."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise WhitelistError("corpus_whitelist.json must be a non-empty JSON list")

    entries: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise WhitelistError(f"whitelist[{i}] must be an object")
        for key in ("path", "modality", "source_origin", "authorization", "notes"):
            if key not in row:
                raise WhitelistError(f"whitelist[{i}] missing field: {key}")

        rel = str(row["path"]).replace("\\", "/").strip()
        if not rel or rel.startswith("/") or rel.startswith("data/"):
            raise WhitelistError(
                f"whitelist[{i}].path must be relative to data/ "
                f"(got {row['path']!r})"
            )
        if ".." in Path(rel).parts:
            raise WhitelistError(f"whitelist[{i}].path must not contain '..'")

        auth = str(row["authorization"])
        if auth not in AUTHORIZATIONS:
            raise WhitelistError(
                f"whitelist[{i}].authorization must be one of "
                f"{sorted(AUTHORIZATIONS)}; got {auth!r}"
            )

        modality = str(row["modality"])
        if modality not in MODALITIES:
            raise WhitelistError(
                f"whitelist[{i}].modality must be one of {sorted(MODALITIES)}"
            )

        entries.append(
            {
                "path": rel,
                "modality": modality,
                "source_origin": str(row["source_origin"]),
                "authorization": auth,
                "notes": str(row["notes"]),
            }
        )
    return entries


def validate_whitelist_files_exist(
    entries: list[dict[str, Any]], data_dir: str | Path
) -> None:
    """Ensure every whitelist path exists under data_dir."""
    root = Path(data_dir)
    for row in entries:
        target = (root / row["path"]).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise WhitelistError(
                f"whitelist path escapes data dir: {row['path']}"
            ) from exc
        if not target.is_file():
            raise WhitelistError(
                f"whitelist file missing on disk: {row['path']} "
                f"(expected under {root})"
            )


def assert_no_forbidden_video_content(videos: list[dict[str, Any]]) -> None:
    """Reject car.mp4 / 汽车剐蹭 style demo videos (SPEC §8.2)."""
    for row in videos:
        url = str(row.get("url", "")).lower()
        desc = str(row.get("description", ""))
        for marker in _FORBIDDEN_URL_MARKERS:
            if marker in url:
                raise WhitelistError(
                    f"forbidden video URL marker {marker!r} in id={row.get('id')}"
                )
        for marker in _FORBIDDEN_DESC_MARKERS:
            if marker in desc:
                raise WhitelistError(
                    f"forbidden video description marker {marker!r} "
                    f"in id={row.get('id')}"
                )


def load_video_sources(path: str | Path) -> list[dict[str, Any]]:
    """Load video_sources.json; require id/url/description; ban forbidden demos."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise WhitelistError("video_sources.json must be a JSON list")
    videos: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise WhitelistError(f"video_sources[{i}] must be an object")
        for key in ("id", "url", "description"):
            if key not in row or not str(row[key]).strip():
                raise WhitelistError(f"video_sources[{i}] missing/empty: {key}")
        videos.append(
            {
                "id": str(row["id"]),
                "url": str(row["url"]).strip(),
                "description": str(row["description"]).strip(),
            }
        )
    assert_no_forbidden_video_content(videos)
    return videos
