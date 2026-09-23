"""FAISS IndexFlatL2 persistence with required scenic manifest (SPEC §10)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.types import IndexManifest

INDEX_FILENAME = "scenic_index.faiss"
METADATA_FILENAME = "scenic_metadata.json"
MANIFEST_FILENAME = "scenic_manifest.json"

DEFAULT_INDEX_DIR = Path("data/index")


class StoreError(ValueError):
    """Raised when index artifacts are missing or inconsistent."""


def _as_float32_matrix(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise StoreError("vectors must be a 2-D array of shape (n, dim)")
    return arr


def _counts_total(counts: dict[str, int]) -> int:
    return int(counts.get("text", 0)) + int(counts.get("image", 0)) + int(
        counts.get("video", 0)
    )


def _validate_before_save(
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    manifest: IndexManifest,
) -> None:
    n, dim = vectors.shape
    if len(metadata) != n:
        raise StoreError(
            f"metadata length ({len(metadata)}) must equal vector count ({n})"
        )
    for i, row in enumerate(metadata):
        if int(row.get("id", -1)) != i:
            raise StoreError(f"metadata[{i}].id must equal {i}")
    if manifest.dim != dim:
        raise StoreError(
            f"manifest.dim ({manifest.dim}) must equal vector dim ({dim})"
        )
    total = _counts_total(manifest.counts)
    if total != n:
        raise StoreError(
            f"manifest counts sum ({total}) must equal vector count / ntotal ({n})"
        )


def save_index(
    vectors: np.ndarray,
    metadata: list[dict[str, Any]],
    manifest: IndexManifest,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> IndexManifest:
    """Build IndexFlatL2 and write faiss + metadata + manifest (all required)."""
    matrix = _as_float32_matrix(vectors)
    _validate_before_save(matrix, metadata, manifest)

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(out_dir / INDEX_FILENAME))
    (out_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_index(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> tuple[faiss.Index, list[dict[str, Any]], IndexManifest]:
    """Load index triad; fail if manifest missing or counts != ntotal."""
    root = Path(index_dir)
    index_path = root / INDEX_FILENAME
    meta_path = root / METADATA_FILENAME
    manifest_path = root / MANIFEST_FILENAME

    if not manifest_path.is_file():
        raise StoreError(
            f"required manifest missing: {manifest_path} "
            "(scenic_manifest.json is mandatory for load)"
        )
    if not index_path.is_file():
        raise StoreError(f"required FAISS index missing: {index_path}")
    if not meta_path.is_file():
        raise StoreError(f"required metadata missing: {meta_path}")

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = IndexManifest(**raw_manifest)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise StoreError("scenic_metadata.json must be a JSON list")

    index = faiss.read_index(str(index_path))
    total = _counts_total(manifest.counts)
    if total != int(index.ntotal):
        raise StoreError(
            f"manifest counts sum ({total}) inconsistent with index.ntotal "
            f"({index.ntotal})"
        )
    if len(metadata) != int(index.ntotal):
        raise StoreError(
            f"metadata length ({len(metadata)}) inconsistent with index.ntotal "
            f"({index.ntotal})"
        )
    return index, metadata, manifest


def search(
    index: faiss.Index,
    query: np.ndarray,
    k: int | None = None,
) -> list[tuple[int, float]]:
    """L2 search; returns (id, distance) ascending by distance.

    When k is None, search the full index (k=ntotal) per SPEC retrieve flow.
    """
    if index.ntotal == 0:
        return []
    top_k = int(index.ntotal if k is None else k)
    top_k = max(0, min(top_k, int(index.ntotal)))
    if top_k == 0:
        return []

    q = np.asarray(query, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    distances, ids = index.search(q, top_k)
    hits: list[tuple[int, float]] = []
    for dist, idx in zip(distances[0].tolist(), ids[0].tolist(), strict=True):
        if idx < 0:
            continue
        hits.append((int(idx), float(dist)))
    return hits
