"""Integration tests for DashScope multimodal embedding adapter (SPEC §6 / §7.2).

DashScope MUST be mocked — no real network.
"""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any

import pytest


MODEL = "tongyi-embedding-vision-plus"


def _mean_rows(rows: list[list[float]]) -> list[float]:
    n = len(rows)
    dim = len(rows[0])
    return [sum(row[i] for row in rows) / n for i in range(dim)]


def _ok_resp(embeddings: list[list[float]], *, message: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        status_code=HTTPStatus.OK,
        message=message,
        output={"embeddings": [{"embedding": e} for e in embeddings]},
    )


def _fail_resp(message: str, status_code: int = 400) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status_code,
        message=message,
        output=None,
    )


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = "sk-test-secret-key-do-not-leak"
    monkeypatch.setenv("DASHSCOPE_API_KEY", key)
    from src import config

    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    return key


def test_get_text_embedding_calls_dashscope_and_returns_vector(
    mocker: Any, api_key: str
) -> None:
    from src.embedding import mm_dashscope as mm

    expected = [0.1, 0.2, 0.3]
    call = mocker.patch(
        "src.embedding.mm_dashscope.dashscope.MultiModalEmbedding.call",
        return_value=_ok_resp([expected]),
    )

    result = mm.get_text_embedding("景区门票规则")

    assert result == expected
    call.assert_called_once()
    kwargs = call.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["input"] == [{"text": "景区门票规则"}]


def test_get_image_embedding_sends_data_uri_base64(
    mocker: Any, api_key: str
) -> None:
    from src.embedding import mm_dashscope as mm

    expected = [0.4, 0.5]
    call = mocker.patch(
        "src.embedding.mm_dashscope.dashscope.MultiModalEmbedding.call",
        return_value=_ok_resp([expected]),
    )

    result = mm.get_image_embedding("abc123BASE64", image_format="jpg")

    assert result == expected
    kwargs = call.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["input"] == [
        {"image": "data:image/jpeg;base64,abc123BASE64"}
    ]


def test_get_image_embedding_accepts_existing_data_uri(
    mocker: Any, api_key: str
) -> None:
    from src.embedding import mm_dashscope as mm

    data_uri = "data:image/png;base64,ZZZ"
    call = mocker.patch(
        "src.embedding.mm_dashscope.dashscope.MultiModalEmbedding.call",
        return_value=_ok_resp([[1.0, 2.0]]),
    )

    mm.get_image_embedding(data_uri)

    assert call.call_args.kwargs["input"] == [{"image": data_uri}]


def test_get_video_embedding_single_frame(mocker: Any, api_key: str) -> None:
    from src.embedding import mm_dashscope as mm

    expected = [0.9, 0.8, 0.7]
    call = mocker.patch(
        "src.embedding.mm_dashscope.dashscope.MultiModalEmbedding.call",
        return_value=_ok_resp([expected]),
    )
    url = "https://example.com/scenic.mp4"

    result = mm.get_video_embedding(url)

    assert result == expected
    assert call.call_args.kwargs["input"] == [{"video": url}]
    assert call.call_args.kwargs["model"] == MODEL


def test_get_video_embedding_multi_frame_mean(mocker: Any, api_key: str) -> None:
    from src.embedding import mm_dashscope as mm

    frames = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
    mocker.patch(
        "src.embedding.mm_dashscope.dashscope.MultiModalEmbedding.call",
        return_value=_ok_resp(frames),
    )

    result = mm.get_video_embedding("https://example.com/multi.mp4")

    expected = _mean_rows(frames)
    assert result == pytest.approx(expected)


def test_embedding_failure_message_readable_and_no_key_leak(
    mocker: Any, api_key: str
) -> None:
    from src.embedding import mm_dashscope as mm
    from src.embedding.mm_dashscope import EmbeddingError

    mocker.patch(
        "src.embedding.mm_dashscope.dashscope.MultiModalEmbedding.call",
        return_value=_fail_resp("InvalidParameter: bad input", status_code=400),
    )

    with pytest.raises(EmbeddingError) as exc_info:
        mm.get_text_embedding("x")

    message = str(exc_info.value)
    assert "text" in message.lower() or "embedding" in message.lower()
    assert "InvalidParameter" in message or "bad input" in message
    assert api_key not in message
    assert "sk-test" not in message


def test_missing_api_key_raises_without_leaking(
    monkeypatch: pytest.MonkeyPatch, mocker: Any
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    from src import config
    from src.embedding import mm_dashscope as mm

    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    call = mocker.patch(
        "src.embedding.mm_dashscope.dashscope.MultiModalEmbedding.call"
    )

    with pytest.raises(ValueError) as exc_info:
        mm.get_text_embedding("x")

    message = str(exc_info.value)
    assert "DASHSCOPE_API_KEY" in message
    assert "sk-" not in message.lower()
    call.assert_not_called()
