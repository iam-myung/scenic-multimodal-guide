"""Unit tests for corpus whitelist rules (SPEC §5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingest.whitelist import (
    AUTHORIZATIONS,
    WhitelistError,
    assert_no_forbidden_video_content,
    load_whitelist,
    validate_whitelist_files_exist,
)


def test_project_whitelist_paths_relative_to_data_and_auth_enum() -> None:
    root = Path(__file__).resolve().parents[2]
    entries = load_whitelist(root / "data" / "corpus_whitelist.json")
    assert entries
    for row in entries:
        assert not row["path"].startswith("data/")
        assert ".." not in Path(row["path"]).parts
        assert row["authorization"] in AUTHORIZATIONS
        assert row["modality"] in {"text", "image", "video_ref"}


def test_load_whitelist_rejects_invalid_authorization(tmp_path: Path) -> None:
    path = tmp_path / "wl.json"
    path.write_text(
        json.dumps(
            [
                {
                    "path": "knowledge_base/a.docx",
                    "modality": "text",
                    "source_origin": "x",
                    "authorization": "portfolio_demo",
                    "notes": "bad",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(WhitelistError) as exc:
        load_whitelist(path)
    assert "authorization" in str(exc.value).lower()


def test_validate_whitelist_files_exist_fails_when_missing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    entries = [
        {
            "path": "knowledge_base/missing.docx",
            "modality": "text",
            "source_origin": "x",
            "authorization": "restricted_local",
            "notes": "",
        }
    ]
    with pytest.raises(WhitelistError) as exc:
        validate_whitelist_files_exist(entries, data_dir)
    assert "missing.docx" in str(exc.value)


def test_forbidden_car_video_rejected() -> None:
    with pytest.raises(WhitelistError) as exc:
        assert_no_forbidden_video_content(
            [
                {
                    "id": "bad",
                    "url": "https://cdn.example.com/mp4/car.mp4",
                    "description": "演示",
                }
            ]
        )
    assert "car.mp4" in str(exc.value).lower()

    with pytest.raises(WhitelistError) as exc2:
        assert_no_forbidden_video_content(
            [
                {
                    "id": "bad2",
                    "url": "https://example.com/ok.mp4",
                    "description": "汽车剐蹭事故",
                }
            ]
        )
    assert "剐蹭" in str(exc2.value) or "汽车" in str(exc2.value)
