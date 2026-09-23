"""RAG package: ask() → AnswerDTO with retrieval-only sources."""

from src.rag.ask import ask
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from src.rag.sources import format_sources

__all__ = [
    "SYSTEM_PROMPT",
    "ask",
    "build_user_prompt",
    "format_sources",
]
