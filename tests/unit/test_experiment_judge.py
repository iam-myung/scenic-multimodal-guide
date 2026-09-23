"""Unit tests for experiment judge — fake vectors only (SPEC §15.2 H5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.judge import ExperimentCaseVerdict, judge_experiment_case
from src.experiment.testset import load_testset


ROOT = Path(__file__).resolve().parents[2]
TESTSET_PATH = ROOT / "data" / "experiment" / "testset.json"


def test_testset_json_has_case_a_and_b_per_spec() -> None:
    cases = load_testset(TESTSET_PATH)
    by_id = {c["id"]: c for c in cases}
    assert set(by_id) >= {"A", "B"}

    assert by_id["A"]["query"] == "一日票提前多久可以免费修改入园日期"
    assert by_id["A"]["positives"]
    assert by_id["A"]["negatives"]
    assert any("48小时" in p or "修改" in p for p in by_id["A"]["positives"])

    assert by_id["B"]["query"] == "老人票价如何计算"
    assert by_id["B"]["positives"]
    assert by_id["B"]["negatives"]
    assert any("65" in p or "老人票" in p or "75%" in p for p in by_id["B"]["positives"])


def test_judge_passes_when_min_pos_gt_max_neg() -> None:
    # Fake 2-D vectors: query near positives, far from negatives
    query = [1.0, 0.0]
    positives = [[0.9, 0.1], [0.8, 0.0]]
    negatives = [[0.0, 1.0], [-1.0, 0.0]]

    verdict = judge_experiment_case(
        query_vec=query,
        positive_vecs=positives,
        negative_vecs=negatives,
    )
    assert isinstance(verdict, ExperimentCaseVerdict)
    assert verdict.passed is True
    assert verdict.min_sim_pos > verdict.max_sim_neg


def test_judge_fails_when_neg_closer_than_pos() -> None:
    query = [1.0, 0.0]
    positives = [[0.0, 1.0]]  # orthogonal → low sim
    negatives = [[1.0, 0.0]]  # identical → high sim

    verdict = judge_experiment_case(
        query_vec=query,
        positive_vecs=positives,
        negative_vecs=negatives,
    )
    assert verdict.passed is False
    assert verdict.min_sim_pos <= verdict.max_sim_neg


def test_judge_forall_cases_with_handcrafted_vectors() -> None:
    """∀ query: min(sim_pos) > max(sim_neg) — exercised with fake vectors only."""
    cases = load_testset(TESTSET_PATH)
    # Assign orthogonal-ish fake embeddings by case id (no model, no FAISS)
    fake_space = {
        "A": {
            "query": [1.0, 0.0, 0.0],
            "pos": [[0.95, 0.05, 0.0]],
            "neg": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        "B": {
            "query": [0.0, 1.0, 0.0],
            "pos": [[0.05, 0.95, 0.0], [0.0, 0.9, 0.1]],
            "neg": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        },
    }
    for case in cases:
        space = fake_space[case["id"]]
        # One fake vec per positive/negative text (pad if more texts than vecs)
        pos_vecs = [
            space["pos"][i % len(space["pos"])] for i in range(len(case["positives"]))
        ]
        neg_vecs = [
            space["neg"][i % len(space["neg"])] for i in range(len(case["negatives"]))
        ]
        verdict = judge_experiment_case(
            query_vec=space["query"],
            positive_vecs=pos_vecs,
            negative_vecs=neg_vecs,
        )
        assert verdict.passed, f"case {case['id']} should pass with fake vectors"


def test_judge_rejects_empty_pos_or_neg() -> None:
    with pytest.raises(ValueError):
        judge_experiment_case(query_vec=[1.0], positive_vecs=[], negative_vecs=[[1.0]])
    with pytest.raises(ValueError):
        judge_experiment_case(query_vec=[1.0], positive_vecs=[[1.0]], negative_vecs=[])
