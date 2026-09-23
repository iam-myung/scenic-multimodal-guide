"""FastAPI thin wrapper around src.rag.ask.ask() (SPEC §21).

Do not change RAG / ingest / FAISS. Key stays server-side (.env).
Default bind: 127.0.0.1 (callers should use --host 127.0.0.1).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    AskRequest,
    AskResponse,
    BuildResponse,
    HealthResponse,
    SourceRefOut,
    StatusResponse,
    meta_to_image_out,
    meta_to_video_out,
)
from src.embedding.mm_dashscope import EmbeddingError
from src.ingest.pipeline import build_index, manifest_to_dict
from src.rag.ask import ask
from src.rag.llm import LLMError
from src.store.faiss_store import (
    DEFAULT_INDEX_DIR,
    MANIFEST_FILENAME,
    StoreError,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WEB_DIR = _PROJECT_ROOT / "web"
_KB_DIR = _PROJECT_ROOT / "data" / "knowledge_base"
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"


def _sanitize_message(exc: BaseException) -> str:
    """Human-readable error; never echo secret values."""
    text = str(exc)
    lower = text.lower()

    if "freetieronly" in lower or "free quota exhausted" in lower:
        return (
            "模型接口仍处于「仅免费额度」模式，建库所需的多模态 Embedding 被拒绝。"
            "请在百炼 / Model Studio 控制台关闭「仅使用免费额度」，"
            "并确认 tongyi-embedding-vision-plus-2026-03-06 已开通按量计费后，再点「一键建库」。"
        )

    if "dashscope_api_key is missing" in lower or (
        "dashscope_api_key" in lower and ("missing" in lower or "empty" in lower)
    ):
        return (
            "建库失败：.env 里还没有填写 DASHSCOPE_API_KEY（百炼 / DashScope）。"
            "DeepSeek 的 LLM_API_KEY 只能生成回答，不能做多模态 Embedding 建库。"
            "请填入百炼 Key 后重启服务，再点「一键建库」。"
        )

    if "llm_api_key" in lower and ("missing" in lower or "empty" in lower):
        return (
            "缺少 LLM_API_KEY（或可用的 DASHSCOPE_API_KEY）。"
            "请在 .env 中配置后重启服务。"
        )

    # Redact only if a key-looking token appears in the message body.
    if "sk-" in lower or "bearer " in lower:
        return "上游请求失败（响应中可能含密钥信息，已隐藏）。请检查 .env 与模型服务配置。"

    return text


def _index_ready() -> bool:
    return (_INDEX_DIR / MANIFEST_FILENAME).is_file()


def _map_ask_error(exc: BaseException) -> HTTPException:
    if isinstance(exc, ValueError) and "query must be" in str(exc):
        return HTTPException(status_code=400, detail="query must be a non-empty string")
    if isinstance(exc, StoreError):
        msg = _sanitize_message(exc)
        if "missing" in msg.lower() or "manifest" in msg.lower():
            return HTTPException(
                status_code=503,
                detail=(
                    "索引未就绪。请在本页点击「一键建库」，"
                    "或在项目根执行：python -m src.cli build"
                ),
            )
        return HTTPException(status_code=503, detail=msg)
    if isinstance(exc, EmbeddingError):
        return HTTPException(status_code=502, detail=_sanitize_message(exc))
    if isinstance(exc, LLMError):
        return HTTPException(status_code=502, detail=_sanitize_message(exc))
    if isinstance(exc, ValueError) and "DASHSCOPE_API_KEY" in str(exc):
        return HTTPException(
            status_code=503,
            detail="DASHSCOPE_API_KEY is missing or empty in .env "
            "(value is never logged).",
        )
    return HTTPException(status_code=500, detail=_sanitize_message(exc))


app = FastAPI(
    title="Scenic Multimodal Guide — local demo",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    ready = _index_ready()
    dashscope_set = False
    try:
        from src.config import ENV_FILE, load_env
        import os

        if ENV_FILE.is_file():
            load_env()
        dashscope_set = bool((os.getenv("DASHSCOPE_API_KEY") or "").strip())
    except Exception:  # noqa: BLE001
        dashscope_set = False

    if ready:
        hint = "索引已就绪，可以直接提问。"
    elif not dashscope_set:
        hint = (
            "索引未就绪，且 .env 未配置 DASHSCOPE_API_KEY。"
            "DeepSeek 只能回答，不能建库；请先填写百炼 Key 并重启服务。"
        )
    else:
        hint = "索引未就绪：请点击「一键建库」（需 Embedding 按量可用）。"
    return StatusResponse(
        index_ready=ready,
        index_dir=str(DEFAULT_INDEX_DIR).replace("\\", "/"),
        hint=hint,
    )


@app.post("/api/build", response_model=BuildResponse)
def api_build() -> BuildResponse:
    """Build FAISS index via the same pipeline as CLI (demo convenience)."""
    try:
        manifest = build_index(
            data_dir=_PROJECT_ROOT / "data",
            index_dir=_INDEX_DIR,
        )
    except Exception as exc:  # noqa: BLE001 — surface for demo UI
        raise HTTPException(status_code=502, detail=_sanitize_message(exc)) from exc

    data = manifest_to_dict(manifest)
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else None
    return BuildResponse(
        ok=True,
        message="建库成功，可以开始提问。",
        counts=counts,
    )


@app.post("/api/ask", response_model=AskResponse)
def api_ask(body: AskRequest) -> AskResponse:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be a non-empty string")

    try:
        dto = ask(query, k=body.k) if body.k is not None else ask(query)
    except Exception as exc:  # noqa: BLE001 — map to HTTP for demo shell
        raise _map_ask_error(exc) from exc

    # Hard rule: serialize ask() fields only — never parse answer text for sources/media.
    return AskResponse(
        answer=dto.answer,
        sources=[
            SourceRefOut(
                source=s.source,
                chunk_id=s.chunk_id,
                similarity=s.similarity,
                content_preview=s.content_preview,
            )
            for s in dto.sources
        ],
        image_ref=meta_to_image_out(dto.image_ref) if dto.image_ref is not None else None,
        video_ref=meta_to_video_out(dto.video_ref) if dto.video_ref is not None else None,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# Mount static last so /api/* wins. Only web/ and knowledge_base — never index/.env/models.
if _KB_DIR.is_dir():
    app.mount("/kb", StaticFiles(directory=str(_KB_DIR)), name="kb")

if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
