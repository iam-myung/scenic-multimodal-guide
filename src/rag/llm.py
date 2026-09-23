"""OpenAI-compatible LLM client (Qwen / DeepSeek / …)."""

from __future__ import annotations

from openai import OpenAI

from src.config import get_llm_base_url, get_llm_model, require_llm_api_key


class LLMError(RuntimeError):
    """Raised when the chat completion call fails."""


def call_chat(system: str, user: str) -> str:
    """Call configured chat model; never logs the API key."""
    api_key = require_llm_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url=get_llm_base_url(),
    )
    try:
        completion = client.chat.completions.create(
            model=get_llm_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as exc:  # noqa: BLE001 — wrap provider errors
        raise LLMError(f"LLM call failed: {type(exc).__name__}") from exc

    choice = completion.choices[0].message.content if completion.choices else None
    if not choice:
        raise LLMError("LLM call failed: empty response")
    return str(choice)


# Backward-compatible alias used by ask() / tests.
def call_qwen_flash(system: str, user: str) -> str:
    return call_chat(system, user)
