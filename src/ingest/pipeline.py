"""Whitelist-scoped index build pipeline (SPEC §7.2 / §8.1)."""

from __future__ import annotations

import base64
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.embedding import mm_dashscope
from src.embedding import mock as mock_embedding
from src.ingest.chunk import CHUNK_OVERLAP, CHUNK_SIZE, split_text
from src.ingest.parse import parse_docx
from src.ingest.whitelist import (
    WhitelistError,
    assert_no_forbidden_video_content,
    load_video_sources,
    load_whitelist,
    validate_whitelist_files_exist,
)
from src.store.faiss_store import (
    DEFAULT_INDEX_DIR,
    INDEX_FILENAME,
    METADATA_FILENAME,
    save_index,
)
from src.types import IndexManifest


class _Embedder(Protocol):
    def get_text_embedding(self, text: str) -> list[float]: ...

    def get_image_embedding(
        self, image_base64: str, *, image_format: str = "jpeg"
    ) -> list[float]: ...

    def get_video_embedding(self, video_url: str) -> list[float]: ...


def _resolve_embedder(mock_embedding_flag: bool) -> tuple[_Embedder, str]:
    env_mock = os.getenv("SCENIC_EMBEDDING_MOCK", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if mock_embedding_flag or env_mock:
        return mock_embedding, mock_embedding.MOCK_EMBEDDING_MODEL
    return mm_dashscope, mm_dashscope.MULTIMODAL_EMBEDDING_MODEL


def _image_to_base64(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = path.suffix.lower().lstrip(".") or "jpeg"
    return b64, ext


def _apply_demo_video_override(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    override = os.getenv("SCENIC_DEMO_VIDEO_URL", "").strip()
    if not override:
        return videos
    updated: list[dict[str, Any]] = []
    for row in videos:
        item = dict(row)
        if item.get("id") == "scenic_demo_1":
            item["url"] = override
        updated.append(item)
    assert_no_forbidden_video_content(updated)
    return updated


def build_index(
    *,
    data_dir: str | Path = Path("data"),
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    whitelist_path: str | Path | None = None,
    mock_embedding: bool = False,
) -> IndexManifest:
    """Build FAISS index from whitelist-only corpus; write triad artifacts."""
    data_root = Path(data_dir)
    wl_path = Path(whitelist_path) if whitelist_path else data_root / "corpus_whitelist.json"
    entries = load_whitelist(wl_path)
    validate_whitelist_files_exist(entries, data_root)

    embedder, model_name = _resolve_embedder(mock_embedding)

    metadata_store: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    doc_id = 0
    counts = {"text": 0, "image": 0, "video": 0}

    for row in entries:
        rel = row["path"]
        modality = row["modality"]
        abs_path = data_root / rel

        if modality == "text":
            full_text = parse_docx(abs_path)
            chunks = split_text(full_text)
            # Relative to data/ — aligns with §15.1 expected_sources_any_of paths
            source_rel = rel.replace("\\", "/")
            for chunk_id, chunk in enumerate(chunks):
                vector = embedder.get_text_embedding(chunk)
                metadata_store.append(
                    {
                        "id": doc_id,
                        "type": "text",
                        "source": source_rel,
                        "content": chunk,
                        "chunk_id": chunk_id,
                    }
                )
                vectors.append(vector)
                doc_id += 1
                counts["text"] += 1

        elif modality == "image":
            b64, fmt = _image_to_base64(abs_path)
            vector = embedder.get_image_embedding(b64, image_format=fmt)
            fname = Path(rel).name
            metadata_store.append(
                {
                    "id": doc_id,
                    "type": "image",
                    "source": f"图片: {fname}",
                    "content": f"[图片] {fname}",
                    "path": rel.replace("\\", "/"),
                }
            )
            vectors.append(vector)
            doc_id += 1
            counts["image"] += 1

        elif modality == "video_ref":
            videos = _apply_demo_video_override(load_video_sources(abs_path))
            if not any(v.get("id") == "scenic_demo_1" for v in videos):
                raise WhitelistError(
                    "video_sources.json must include id=scenic_demo_1"
                )
            for video in videos:
                vector = embedder.get_video_embedding(video["url"])
                desc = video["description"]
                metadata_store.append(
                    {
                        "id": doc_id,
                        "type": "video",
                        "source": f"视频: {desc}",
                        "content": f"[视频] {desc}",
                        "url": video["url"],
                        "description": desc,
                        "video_id": video["id"],
                    }
                )
                vectors.append(vector)
                doc_id += 1
                counts["video"] += 1

    if not vectors:
        raise WhitelistError("build_index produced zero vectors; check whitelist corpus")

    dim = len(vectors[0])
    out_dir = Path(index_dir)
    manifest = IndexManifest(
        embedding_model=model_name,
        dim=dim,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        built_at=datetime.now(timezone.utc).isoformat(),
        counts=counts,
        faiss_index_path=str(out_dir / INDEX_FILENAME).replace("\\", "/"),
        metadata_path=str(out_dir / METADATA_FILENAME).replace("\\", "/"),
    )
    save_index(np.asarray(vectors, dtype=np.float32), metadata_store, manifest, out_dir)
    return manifest


def manifest_to_dict(manifest: IndexManifest) -> dict[str, Any]:
    return asdict(manifest)
