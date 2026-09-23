"""Integration tests for build_index with mocked embeddings (SPEC §7.2 / §5)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.ingest.pipeline import build_index
from src.store.faiss_store import (
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    load_index,
)


@pytest.fixture
def mini_data_dir(tmp_path: Path) -> Path:
    """Minimal data tree: one docx (copied), one image, video_sources, whitelist."""
    root = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    kb = data / "knowledge_base"
    images = kb / "images"
    images.mkdir(parents=True)

    src_docx = next(
        (root / "data" / "knowledge_base").glob("4-*.docx")
    )
    src_img = next(
        (root / "data" / "knowledge_base" / "images").glob("2-*.jpeg")
    )
    shutil.copy2(src_docx, kb / src_docx.name)
    shutil.copy2(src_img, images / src_img.name)

    (data / "video_sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "scenic_demo_1",
                    "url": "https://example.com/placeholder/scenic-demo.mp4",
                    "description": "景区演示视频占位",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    whitelist = [
        {
            "path": f"knowledge_base/{src_docx.name}",
            "modality": "text",
            "source_origin": "test",
            "authorization": "restricted_local",
            "notes": "mini",
        },
        {
            "path": f"knowledge_base/images/{src_img.name}",
            "modality": "image",
            "source_origin": "test",
            "authorization": "restricted_local",
            "notes": "mini",
        },
        {
            "path": "video_sources.json",
            "modality": "video_ref",
            "source_origin": "test",
            "authorization": "restricted_local",
            "notes": "mini",
        },
    ]
    (data / "corpus_whitelist.json").write_text(
        json.dumps(whitelist, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Extra file NOT on whitelist — must be ignored
    (kb / "secret-not-whitelisted.docx").write_bytes(b"PK fake")
    return data


def test_build_index_mock_writes_triad_and_respects_whitelist(
    mini_data_dir: Path, tmp_path: Path
) -> None:
    index_dir = tmp_path / "index"
    manifest = build_index(
        data_dir=mini_data_dir,
        index_dir=index_dir,
        mock_embedding=True,
    )

    assert (index_dir / INDEX_FILENAME).is_file()
    assert (index_dir / METADATA_FILENAME).is_file()
    assert (index_dir / MANIFEST_FILENAME).is_file()

    index, metadata, loaded = load_index(index_dir)
    assert loaded.dim == manifest.dim
    assert index.ntotal == len(metadata)
    assert manifest.counts["text"] >= 1
    assert manifest.counts["image"] == 1
    assert manifest.counts["video"] == 1

    sources = {m.get("source") for m in metadata}
    assert not any("secret-not-whitelisted" in str(s) for s in sources)

    types = {m["type"] for m in metadata}
    assert types == {"text", "image", "video"}
    video_rows = [m for m in metadata if m["type"] == "video"]
    assert video_rows[0]["url"]
    assert "car.mp4" not in video_rows[0]["url"].lower()
