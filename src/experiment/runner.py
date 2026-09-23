"""Run contrast experiment over testset with BGE and/or GTE (SPEC §15.2 H5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.config import get_modelscope_cache_dir
from src.experiment.adapters_common import ContrastEmbedder
from src.experiment.bge_adapter import BgeM3Embedder
from src.experiment.gte_adapter import GteQwenEmbedder
from src.experiment.judge import ExperimentCaseVerdict, judge_experiment_case
from src.experiment.testset import DEFAULT_TESTSET_PATH, load_testset

SUPPORTED_MODELS = ("bge", "gte", "all")


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    query: str
    passed: bool
    min_sim_pos: float
    max_sim_neg: float


@dataclass(frozen=True)
class ModelResult:
    model: str
    passed: bool
    cases: tuple[CaseResult, ...]


@dataclass(frozen=True)
class ExperimentReport:
    h5_passed: bool
    models: tuple[ModelResult, ...]
    testset_path: str
    cache_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "h5_passed": self.h5_passed,
            "testset_path": self.testset_path,
            "cache_dir": self.cache_dir,
            "models": [
                {
                    "model": m.model,
                    "passed": m.passed,
                    "cases": [asdict(c) for c in m.cases],
                }
                for m in self.models
            ],
        }


def _normalize_model_choice(model: str) -> tuple[str, ...]:
    key = str(model).strip().lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(
            f"unsupported --model {model!r}; expected one of {SUPPORTED_MODELS}"
        )
    if key == "all":
        return ("bge", "gte")
    return (key,)


def _load_embedder(name: str, cache_dir: Path) -> ContrastEmbedder:
    if name == "bge":
        return BgeM3Embedder.from_cache(cache_dir)
    if name == "gte":
        return GteQwenEmbedder.from_cache(cache_dir)
    raise ValueError(f"unknown embedder: {name}")


def _eval_case(embedder: ContrastEmbedder, case: dict[str, Any]) -> CaseResult:
    query_vec = embedder.embed_query(case["query"])
    pos_vecs = embedder.embed_documents(case["positives"])
    neg_vecs = embedder.embed_documents(case["negatives"])
    verdict: ExperimentCaseVerdict = judge_experiment_case(
        query_vec=query_vec,
        positive_vecs=pos_vecs,
        negative_vecs=neg_vecs,
    )
    return CaseResult(
        case_id=str(case["id"]),
        query=str(case["query"]),
        passed=verdict.passed,
        min_sim_pos=verdict.min_sim_pos,
        max_sim_neg=verdict.max_sim_neg,
    )


def run_experiment(
    model: str = "all",
    *,
    testset_path: str | Path = DEFAULT_TESTSET_PATH,
    cache_dir: str | Path | None = None,
    embedders: dict[str, ContrastEmbedder] | None = None,
) -> ExperimentReport:
    """Evaluate testset with selected model(s). Does not touch main FAISS index.

    ``embedders`` may inject mocks for unit tests (keys: bge / gte).
    """
    chosen = _normalize_model_choice(model)
    cases = load_testset(testset_path)
    cache = Path(cache_dir) if cache_dir is not None else get_modelscope_cache_dir()

    model_results: list[ModelResult] = []
    for name in chosen:
        if embedders is not None and name in embedders:
            embedder = embedders[name]
        else:
            embedder = _load_embedder(name, cache)
        case_results = tuple(_eval_case(embedder, case) for case in cases)
        model_results.append(
            ModelResult(
                model=embedder.name if hasattr(embedder, "name") else name,
                passed=all(c.passed for c in case_results),
                cases=case_results,
            )
        )

    # H5: every selected model must pass every case.
    # When model=all, both BGE and GTE must pass.
    h5_passed = all(m.passed for m in model_results) and bool(model_results)
    return ExperimentReport(
        h5_passed=h5_passed,
        models=tuple(model_results),
        testset_path=str(Path(testset_path)),
        cache_dir=str(cache),
    )


def iter_missing_deps() -> Iterable[str]:
    """Return missing optional import names (for CLI diagnostics)."""
    missing: list[str] = []
    for mod, label in (
        ("FlagEmbedding", "FlagEmbedding"),
        ("sentence_transformers", "sentence_transformers"),
        ("modelscope", "modelscope"),
    ):
        try:
            __import__(mod)
        except ImportError:
            missing.append(label)
    return missing
