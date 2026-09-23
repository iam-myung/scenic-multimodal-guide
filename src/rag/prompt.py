"""Prompt templates for scenic RAG (SPEC §11.1)."""

from __future__ import annotations

from src.retrieve.search import SearchHit

SYSTEM_PROMPT = (
    "你是景区智能讲解助手。仅根据给定背景知识回答；不知则说明知识库未覆盖。"
    "演示知识不代表官方实时信息。"
)

# User-side guard: demo doc titles may not match body wording (e.g. 酒店会员制度 → VIP 权益).
_ANSWER_RULES = (
    "请仅依据上方背景知识回答。"
    "若某条背景的来源文件名或文首标题与用户问题主题相关，"
    "请将该条正文视为对该主题的说明并归纳要点作答，"
    "即使正文用词与问题不完全一致"
    "（例如来源为「酒店会员制度」而正文写 VIP/尊享卡/礼宾服务，仍应据此回答会员权益类问题）。"
    "不要在已提供相关背景时声称知识库未覆盖；"
    "仅当背景知识确实无法支撑该问题时，再说明知识库未覆盖。"
)

_MEDIA_ANSWER_RULES = (
    "若上方已列出「已检索媒体」，说明系统已匹配相关图片或视频；"
    "请据此告知用户已匹配到相关海报/图片或演示视频（可引用 path/文件名或 url/描述），"
    "不要声称知识库未覆盖；即使文本背景不足，仍应说明媒体已匹配。"
)


def _format_selected_media(
    image_hit: SearchHit | None,
    video_hit: SearchHit | None,
) -> str:
    lines: list[str] = []
    if image_hit is not None:
        meta = image_hit.metadata
        lines.append(
            f"- 图片: path={meta.get('path', '')}; "
            f"content={meta.get('content', '')}; "
            f"source={meta.get('source', '')}"
        )
    if video_hit is not None:
        meta = video_hit.metadata
        lines.append(
            f"- 视频: url={meta.get('url', '')}; "
            f"description={meta.get('description', '')}; "
            f"content={meta.get('content', '')}"
        )
    if not lines:
        return ""
    return "[已检索媒体]\n" + "\n".join(lines) + "\n\n"


def build_user_prompt(
    query: str,
    text_hits: list[SearchHit],
    *,
    image_hit: SearchHit | None = None,
    video_hit: SearchHit | None = None,
) -> str:
    """Assemble user message from text hits + optional selected media (retrieval only)."""
    blocks: list[str] = []
    for i, hit in enumerate(text_hits, start=1):
        meta = hit.metadata
        blocks.append(
            f"背景知识 {i} (来源: {meta.get('source', '')}, "
            f"chunk_id: {meta.get('chunk_id', 0)}, "
            f"相似度: {hit.similarity:.4f}):\n"
            f"{meta.get('content', '')}"
        )
    context = "\n\n".join(blocks) if blocks else "（本次检索未命中文本知识）"
    media_block = _format_selected_media(image_hit, video_hit)
    rules = _ANSWER_RULES
    if image_hit is not None or video_hit is not None:
        rules = f"{_ANSWER_RULES}\n{_MEDIA_ANSWER_RULES}"
    return (
        f"[背景知识]\n{context}\n\n"
        f"{media_block}"
        f"[用户问题]\n{query}\n\n"
        f"[作答要求]\n{rules}"
    )
