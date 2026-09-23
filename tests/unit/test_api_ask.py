"""Unit tests for optional Web API (SPEC §21) — mock ask(), no real network."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.api.app import app
from src.store.faiss_store import StoreError
from src.types import AnswerDTO, ImageMeta, SourceRef, VideoMeta


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ask_empty_query_400(client: TestClient) -> None:
    res = client.post("/api/ask", json={"query": "   "})
    assert res.status_code == 400
    assert "query" in res.json()["detail"].lower()
    assert "sk-" not in res.text.lower()


def test_ask_success_serializes_dto_fields(client: TestClient) -> None:
    dto = AnswerDTO(
        answer="演示回答",
        sources=[
            SourceRef(
                source="knowledge_base/1.docx",
                chunk_id=0,
                similarity=0.91,
                content_preview="预览片段",
            )
        ],
        image_ref=ImageMeta(
            id=1,
            type="image",
            source="img",
            content="海报",
            path="knowledge_base/images/2-万圣节.jpeg",
        ),
        video_ref=VideoMeta(
            id=2,
            type="video",
            source="scenic_demo_1",
            content="demo",
            url="https://example.com/demo.mp4",
            description="景区演示",
        ),
    )
    with patch("src.api.app.ask", return_value=dto) as mocked:
        res = client.post("/api/ask", json={"query": "万圣节海报"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "演示回答"
    assert body["sources"][0]["source"] == "knowledge_base/1.docx"
    assert body["sources"][0]["content_preview"] == "预览片段"
    assert body["image_ref"]["path"] == "knowledge_base/images/2-万圣节.jpeg"
    assert body["video_ref"]["url"] == "https://example.com/demo.mp4"
    mocked.assert_called_once()
    assert "sk-" not in res.text


def test_ask_missing_index_maps_to_build_hint(client: TestClient) -> None:
    with patch(
        "src.api.app.ask",
        side_effect=StoreError("required manifest missing: data/index/scenic_manifest.json"),
    ):
        res = client.post("/api/ask", json={"query": "一日票改期"})
    assert res.status_code == 503
    detail = res.json()["detail"].lower()
    assert "索引" in res.json()["detail"] or "index" in detail
    assert "建库" in res.json()["detail"] or "build" in detail


def test_status_reports_index_flag(client: TestClient) -> None:
    with patch("src.api.app._index_ready", return_value=False):
        res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert body["index_ready"] is False
    assert "hint" in body


def test_build_maps_freetier_error(client: TestClient) -> None:
    with patch(
        "src.api.app.build_index",
        side_effect=RuntimeError(
            'AllocationQuota.FreeTierOnly: Free quota exhausted. disable the "use free tier only" mode'
        ),
    ):
        res = client.post("/api/build")
    assert res.status_code == 502
    assert "免费额度" in res.json()["detail"]


def test_build_maps_missing_dashscope_key(client: TestClient) -> None:
    with patch(
        "src.api.app.build_index",
        side_effect=ValueError(
            "DASHSCOPE_API_KEY is missing or empty in .env. "
            "Multimodal embedding still requires a DashScope key — DeepSeek cannot replace it."
        ),
    ):
        res = client.post("/api/build")
    assert res.status_code == 502
    detail = res.json()["detail"]
    assert "DASHSCOPE_API_KEY" in detail or "百炼" in detail
    assert "DeepSeek" in detail
    assert "sk-" not in detail.lower()
