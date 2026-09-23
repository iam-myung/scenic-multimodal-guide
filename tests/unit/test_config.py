"""Unit tests for src.config — Key validation (SPEC §13)."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_get_dashscope_api_key_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    from src import config

    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env_file)

    with pytest.raises(ValueError) as exc_info:
        config.require_dashscope_api_key()

    message = str(exc_info.value)
    assert "DASHSCOPE_API_KEY" in message
    assert "sk-" not in message.lower()


def test_require_dashscope_api_key_returns_value_from_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    from src import config

    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=test-key-not-real\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env_file)

    assert config.require_dashscope_api_key() == "test-key-not-real"


def test_compat_url_defaults_and_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DASHSCOPE_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_HTTP_API_URL", raising=False)

    from src import config

    env_file = tmp_path / ".env"
    env_file.write_text("DASHSCOPE_API_KEY=test-key-not-real\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env_file)

    assert config.get_dashscope_compat_base_url() == config.DEFAULT_COMPAT_BASE_URL
    assert config.get_dashscope_http_api_url() == config.DEFAULT_HTTP_API_URL

    env_file.write_text(
        "DASHSCOPE_API_KEY=test-key-not-real\n"
        "DASHSCOPE_COMPAT_BASE_URL="
        "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "DASHSCOPE_COMPAT_BASE_URL",
        "https://ws-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    assert config.get_dashscope_compat_base_url().endswith("compatible-mode/v1")
    assert config.get_dashscope_http_api_url().endswith("/api/v1")


def test_llm_deepseek_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    from src import config

    env_file = tmp_path / ".env"
    env_file.write_text(
        "DASHSCOPE_API_KEY=\n"
        "LLM_API_KEY=deepseek-test-key\n"
        "LLM_BASE_URL=https://api.deepseek.com\n"
        "LLM_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setenv("LLM_API_KEY", "deepseek-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")

    assert config.require_llm_api_key() == "deepseek-test-key"
    assert config.get_llm_base_url() == "https://api.deepseek.com"
    assert config.get_llm_model() == "deepseek-chat"
    with pytest.raises(ValueError) as exc_info:
        config.require_dashscope_api_key()
    assert "DeepSeek" in str(exc_info.value) or "embedding" in str(exc_info.value).lower()


def test_load_env_raises_when_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from src import config

    missing = tmp_path / "does-not-exist.env"
    monkeypatch.setattr(config, "ENV_FILE", missing)

    with pytest.raises(ValueError) as exc_info:
        config.load_env()

    assert ".env" in str(exc_info.value) or "Env file missing" in str(exc_info.value)
