"""H5 ranking judge: min(sim_pos) > max(sim_neg) (SPEC §15.2).

Uses injected vectors only in this module — no FAISS, no real embedding models.
Similarity matches 19_ dense style: query · document (dot product).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ExperimentCaseVerdict:
    passed: bool
    min_sim_pos: float
    max_sim_neg: float
    sim_pos: tuple[float, ...]
    sim_neg: tuple[float, ...]


def _as_float_vec(vec: Sequence[float]) -> list[float]:
    values = [float(x) for x in vec]
    if not values:
        raise ValueError("vector must be non-empty")
    return values


def dot_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Dense similarity used by BGE-M3 style (q @ d.T)."""
    va = _as_float_vec(a)
    vb = _as_float_vec(b)
    if len(va) != len(vb):
        raise ValueError(
            f"vector dim mismatch: query/doc dims {len(va)} vs {len(vb)}"
        )
    return sum(x * y for x, y in zip(va, vb, strict=True))


def judge_experiment_case(
    *,
    query_vec: Sequence[float],
    positive_vecs: Sequence[Sequence[float]],
    negative_vecs: Sequence[Sequence[float]],
) -> ExperimentCaseVerdict:
    """Pass iff min(sim_pos) > max(sim_neg) for the given fake/real vectors."""
    if not positive_vecs:
        raise ValueError("positive_vecs must be non-empty")
    if not negative_vecs:
        raise ValueError("negative_vecs must be non-empty")

    q = _as_float_vec(query_vec)
    sim_pos = tuple(dot_similarity(q, v) for v in positive_vecs)
    sim_neg = tuple(dot_similarity(q, v) for v in negative_vecs)
    min_pos = min(sim_pos)
    max_neg = max(sim_neg)
    return ExperimentCaseVerdict(
        passed=min_pos > max_neg,
        min_sim_pos=min_pos,
        max_sim_neg=max_neg,
        sim_pos=sim_pos,
        sim_neg=sim_neg,
    )
