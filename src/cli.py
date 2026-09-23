"""CLI entry: python -m src.cli build|ask|experiment."""

from __future__ import annotations

import argparse
import json
import sys

from src.experiment.adapters_common import ExperimentDependencyError
from src.experiment.runner import run_experiment
from src.ingest.pipeline import build_index, manifest_to_dict
from src.rag.ask import ask
from src.types import AnswerDTO, ImageMeta, VideoMeta


def format_ask_output(dto: AnswerDTO) -> str:
    """Render AnswerDTO for CLI: answer + [检索证据] + optional media (SPEC §11.2)."""
    lines: list[str] = [dto.answer.strip(), "", "[检索证据]"]
    if not dto.sources:
        lines.append("(无文本检索证据)")
    else:
        for i, ref in enumerate(dto.sources, start=1):
            lines.append(
                f"{i}. source={ref.source} | chunk_id={ref.chunk_id} | "
                f"similarity={ref.similarity:.4f}"
            )
            lines.append(f"   preview={ref.content_preview}")

    if isinstance(dto.image_ref, ImageMeta):
        lines.extend(
            [
                "",
                "[相关图片]",
                f"path={dto.image_ref.path}",
                f"source={dto.image_ref.source}",
            ]
        )

    if isinstance(dto.video_ref, VideoMeta):
        lines.extend(
            [
                "",
                "[相关视频]",
                f"url={dto.video_ref.url}",
                f"description={dto.video_ref.description}",
            ]
        )

    return "\n".join(lines)


def _cmd_build(args: argparse.Namespace) -> int:
    manifest = build_index(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        mock_embedding=args.mock_embedding,
    )
    print(json.dumps(manifest_to_dict(manifest), ensure_ascii=False, indent=2))
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    query = (args.query or "").strip()
    if query:
        dto = ask(query, k=args.k, index_dir=args.index_dir)
        print(format_ask_output(dto))
        return 0

    print(
        "交互问答已启动。输入问题后回车；输入 quit / exit / 空行 退出。",
        flush=True,
    )
    while True:
        try:
            line = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not line or line.lower() in {"quit", "exit", "q"}:
            return 0
        try:
            dto = ask(line, k=args.k, index_dir=args.index_dir)
            print(format_ask_output(dto), flush=True)
            print(flush=True)
        except Exception as exc:  # noqa: BLE001 — keep REPL alive on one bad turn
            print(f"ERROR: {exc}", file=sys.stderr, flush=True)
            print(flush=True)


def _cmd_experiment(args: argparse.Namespace) -> int:
    try:
        report = run_experiment(
            model=args.model,
            testset_path=args.testset,
            cache_dir=args.cache_dir,
        )
    except ExperimentDependencyError as exc:
        print(
            f"ERROR: {exc}\n"
            "Hint: pip install -r requirements-experiment.txt "
            "(and a matching torch build). P0 H5 requires these deps.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.h5_passed else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    """Start local Web demo (uvicorn) and optionally open the browser."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Missing web deps. Install with: pip install -r requirements-web.txt",
            file=sys.stderr,
        )
        return 2

    host = args.host
    port = int(args.port)
    url = f"http://{host}:{port}/"
    print(f"Starting Web demo at {url}", flush=True)
    print("Index tip: open the page and use「一键建库」, or: python -m src.cli build", flush=True)

    if args.open:
        import threading
        import webbrowser

        def _open() -> None:
            import time

            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=(
            "Scenic multimodal RAG assistant CLI. "
            "With no subcommand, defaults to interactive ask."
        ),
    )
    # Bare `python -m src.cli` → interactive ask (same as `ask` with no query).
    parser.set_defaults(func=_cmd_ask, query=None, k=3, index_dir="data/index")
    sub = parser.add_subparsers(dest="command", required=False)

    build_p = sub.add_parser("build", help="Build FAISS index from whitelist corpus")
    build_p.add_argument(
        "--data-dir",
        default="data",
        help="Data root containing corpus_whitelist.json (default: data)",
    )
    build_p.add_argument(
        "--index-dir",
        default="data/index",
        help="Output directory for faiss/metadata/manifest (default: data/index)",
    )
    build_p.add_argument(
        "--mock-embedding",
        action="store_true",
        help="Use deterministic mock embeddings (no DashScope Key; smoke only)",
    )
    build_p.set_defaults(func=_cmd_build)

    ask_p = sub.add_parser(
        "ask",
        help="Ask a text question (omit query to enter interactive loop)",
    )
    ask_p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="User question (text only); omit for interactive 问题> loop",
    )
    ask_p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-k text contexts for RAG (default: 3)",
    )
    ask_p.add_argument(
        "--index-dir",
        default="data/index",
        help="Index directory with faiss/metadata/manifest (default: data/index)",
    )
    ask_p.set_defaults(func=_cmd_ask)

    exp_p = sub.add_parser(
        "experiment",
        help="Run contrast embedding H5 (BGE/GTE); does not write main FAISS index",
    )
    exp_p.add_argument(
        "--model",
        default="all",
        choices=("bge", "gte", "all"),
        help="Which contrast model(s) to run (default: all)",
    )
    exp_p.add_argument(
        "--testset",
        default="data/experiment/testset.json",
        help="Path to fixed testset JSON (default: data/experiment/testset.json)",
    )
    exp_p.add_argument(
        "--cache-dir",
        default=None,
        help="Override MODELSCOPE_CACHE_DIR for model downloads",
    )
    exp_p.set_defaults(func=_cmd_experiment)

    serve_p = sub.add_parser(
        "serve",
        help="Start local Web demo (uvicorn on 127.0.0.1:8000)",
    )
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument(
        "--open",
        action="store_true",
        help="Open the demo URL in the default browser after start",
    )
    serve_p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
