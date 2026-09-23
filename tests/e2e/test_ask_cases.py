"""E2E T1–T4 hard assertions (SPEC §15.1). Skipped unless RUN_E2E=1."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.ingest.pipeline import build_index
from src.ingest.whitelist import load_video_sources
from src.rag.ask import ask
from tests.e2e.assert_rules import (
    assert_cross_sample_gate,
    assert_e2e_case,
    case_media_success,
)

pytestmark = pytest.mark.e2e

CASES_PATH = Path(__file__).with_name("cases.json")
ROOT = Path(__file__).resolve().parents[2]


def _require_run_e2e() -> None:
    if os.getenv("RUN_E2E", "").strip() != "1":
        pytest.skip("e2e skipped unless RUN_E2E=1")


@pytest.fixture(scope="module")
def e2e_ready() -> Path:
    _require_run_e2e()
    data_dir = ROOT / "data"
    index_dir = data_dir / "index"
    # Rebuild so embeddings/LLM path match current corpus + Key
    build_index(data_dir=data_dir, index_dir=index_dir, mock_embedding=False)
    return index_dir


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) >= 4
    return raw


@pytest.fixture(scope="module")
def video_catalog() -> list[dict]:
    videos = load_video_sources(ROOT / "data" / "video_sources.json")
    override = os.getenv("SCENIC_DEMO_VIDEO_URL", "").strip()
    if not override:
        return videos
    updated: list[dict] = []
    for row in videos:
        item = dict(row)
        if item.get("id") == "scenic_demo_1":
            item["url"] = override
        updated.append(item)
    return updated


def test_cases_json_matches_spec_t1_t4(cases: list[dict]) -> None:
    """Fixture shape check — runs even without RUN_E2E so cases.json is gated in CI."""
    by_id = {c["id"]: c for c in cases}
    assert set(by_id) >= {"T1", "T2", "T3", "T4"}

    t1 = by_id["T1"]
    assert "一日票" in t1["query"] and "修改" in t1["query"]
    assert "knowledge_base/1-某大型游乐园门票规则.docx" in t1["expected_sources_any_of"]
    assert t1["require_image_ref"] is False and t1["require_video_ref"] is False

    t2 = by_id["T2"]
    assert t2["require_image_ref"] is True
    assert t2["expected_image_path"] == "knowledge_base/images/2-万圣节.jpeg"

    t3 = by_id["T3"]
    assert t3["require_video_ref"] is True
    assert t3["expected_video_id"] == "scenic_demo_1"

    t4 = by_id["T4"]
    assert t4["require_image_ref"] is True
    assert t4["expected_image_path"] == "knowledge_base/images/1-园区合影.jpg"


def test_e2e_t1_t4_hard_assertions(
    e2e_ready: Path,
    cases: list[dict],
    video_catalog: list[dict],
) -> None:
    media_flags: dict[str, bool] = {}
    for case in cases:
        if case["id"] not in {"T1", "T2", "T3", "T4"}:
            continue
        dto = ask(case["query"], index_dir=e2e_ready)
        assert_e2e_case(dto, case, video_catalog=video_catalog)
        media_flags[case["id"]] = case_media_success(case["id"], dto, case)

    assert_cross_sample_gate(media_flags)
