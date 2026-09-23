"""Unit tests for store.faiss — IndexFlatL2 + required manifest (SPEC §10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.types import IndexManifest


def _manifest(**overrides) -> IndexManifest:
    base = dict(
        embedding_model="tongyi-embedding-vision-plus",
        dim=4,
        chunk_size=500,
        chunk_overlap=50,
        built_at="2026-09-12T00:00:00Z",
        counts={"text": 2, "image": 1, "video": 0},
        faiss_index_path="data/index/scenic_index.faiss",
        metadata_path="data/index/scenic_metadata.json",
    )
    base.update(overrides)
    return IndexManifest(**base)  # type: ignore[arg-type]


def _fake_vectors(n: int = 3, dim: int = 4) -> np.ndarray:
    # Distinct unit-ish vectors for stable L2 nearest-neighbor checks
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim), dtype=np.float32)


def _meta_rows(n: int = 3) -> list[dict]:
    rows: list[dict] = []
    for i in range(n):
        if i < 2:
            rows.append(
                {
                    "id": i,
                    "type": "text",
                    "source": f"doc.docx",
                    "content": f"chunk-{i}",
                    "chunk_id": i,
                }
            )
        else:
            rows.append(
                {
                    "id": i,
                    "type": "image",
                    "source": "img.jpg",
                    "content": "[image] img.jpg",
                    "path": "knowledge_base/images/img.jpg",
                }
            )
    return rows


def test_save_writes_three_required_files(tmp_path: Path) -> None:
    from src.store.faiss_store import (
        METADATA_FILENAME,
        MANIFEST_FILENAME,
        INDEX_FILENAME,
        save_index,
    )

    vectors = _fake_vectors()
    metadata = _meta_rows()
    manifest = _manifest()

    save_index(vectors, metadata, manifest, index_dir=tmp_path)

    assert (tmp_path / INDEX_FILENAME).is_file()
    assert (tmp_path / METADATA_FILENAME).is_file()
    assert (tmp_path / MANIFEST_FILENAME).is_file()


def test_save_load_roundtrip(tmp_path: Path) -> None:
    from src.store.faiss_store import load_index, save_index, search

    vectors = _fake_vectors()
    metadata = _meta_rows()
    manifest = _manifest()
    save_index(vectors, metadata, manifest, index_dir=tmp_path)

    index, loaded_meta, loaded_manifest = load_index(index_dir=tmp_path)
    assert index.ntotal == 3
    assert loaded_meta == metadata
    assert loaded_manifest.counts == manifest.counts
    assert loaded_manifest.dim == 4


def test_load_missing_manifest_fails(tmp_path: Path) -> None:
    from src.store.faiss_store import (
        INDEX_FILENAME,
        METADATA_FILENAME,
        StoreError,
        load_index,
        save_index,
    )

    save_index(_fake_vectors(), _meta_rows(), _manifest(), index_dir=tmp_path)
    (tmp_path / "scenic_manifest.json").unlink()

    with pytest.raises(StoreError) as exc_info:
        load_index(index_dir=tmp_path)
    assert "manifest" in str(exc_info.value).lower()


def test_load_counts_mismatch_ntotal_fails(tmp_path: Path) -> None:
    from src.store.faiss_store import StoreError, load_index, save_index

    save_index(_fake_vectors(), _meta_rows(), _manifest(), index_dir=tmp_path)
    # Corrupt manifest counts so sum != ntotal
    bad = _manifest(counts={"text": 99, "image": 0, "video": 0})
    import json
    from dataclasses import asdict

    (tmp_path / "scenic_manifest.json").write_text(
        json.dumps(asdict(bad), ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(StoreError) as exc_info:
        load_index(index_dir=tmp_path)
    msg = str(exc_info.value).lower()
    assert "count" in msg or "ntotal" in msg


def test_search_returns_nearest_ids(tmp_path: Path) -> None:
    from src.store.faiss_store import load_index, save_index, search

    vectors = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    save_index(vectors, _meta_rows(), _manifest(), index_dir=tmp_path)
    index, _, _ = load_index(index_dir=tmp_path)

    query = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    hits = search(index, query, k=2)
    assert len(hits) == 2
    assert hits[0][0] == 1  # nearest id
    assert hits[0][1] < hits[1][1]  # L2 distance ascending


def test_store_module_has_no_dashscope_import() -> None:
    import src.store.faiss_store as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "dashscope" not in src.lower()
