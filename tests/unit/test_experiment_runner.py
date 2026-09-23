"""Unit tests for contrast experiment runner — mocked embedders only."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import main
from src.experiment.runner import run_experiment


ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "data" / "experiment" / "testset.json"


class _FakePassEmbedder:
    """Odd embed_documents calls = positives; even = negatives."""

    name = "fake-pass"

    def __init__(self) -> None:
        self._doc_calls = 0

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts):
        self._doc_calls += 1
        if self._doc_calls % 2 == 1:
            return [[0.95, 0.05] for _ in texts]
        return [[0.05, 0.95] for _ in texts]


class _FakeFailEmbedder:
    name = "fake-fail"

    def __init__(self) -> None:
        self._doc_calls = 0

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts):
        self._doc_calls += 1
        if self._doc_calls % 2 == 1:
            return [[0.05, 0.95] for _ in texts]
        return [[0.95, 0.05] for _ in texts]


def test_run_experiment_all_with_injected_embedders_passes() -> None:
    report = run_experiment(
        model="all",
        testset_path=TESTSET,
        embedders={"bge": _FakePassEmbedder(), "gte": _FakePassEmbedder()},
    )
    assert report.h5_passed is True
    assert len(report.models) == 2
    assert all(m.passed for m in report.models)
    payload = report.to_dict()
    assert payload["h5_passed"] is True


def test_run_experiment_h5_fails_if_one_model_fails() -> None:
    report = run_experiment(
        model="all",
        testset_path=TESTSET,
        embedders={"bge": _FakePassEmbedder(), "gte": _FakeFailEmbedder()},
    )
    assert report.h5_passed is False


def test_cli_experiment_help_lists_model_all() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["experiment", "--help"])
    assert exc.value.code == 0


def test_get_modelscope_cache_dir_default_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src import config

    monkeypatch.delenv("MODELSCOPE_CACHE_DIR", raising=False)
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    default = config.get_modelscope_cache_dir()
    assert default.name == "models"

    monkeypatch.setenv("MODELSCOPE_CACHE_DIR", str(tmp_path / "custom_models"))
    assert config.get_modelscope_cache_dir() == (tmp_path / "custom_models").resolve()
