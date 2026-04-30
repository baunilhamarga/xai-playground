"""Backfill Trace Coverage Faithfulness (TCF) evaluation into existing experiment logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from .backends import (
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_GROQ_MODEL,
    build_backend,
)
from .evaluation import (
    EvaluationParams,
    append_evaluation_to_log,
    evaluate_experiment_log,
    evaluations_from_log,
    write_evaluation_to_log,
)
from .trace_loader import parse_trace_path


REPRESENTATIVE_CASES = {
    (0, 0),
    (5, 5),
    (10, 10),
    (2, 7),
    (0, 10),
    (10, 0),
    (2, 2),
    (8, 5),
    (5, 8),
    (9, 9),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_paths = resolve_log_paths(args)
    if not log_paths:
        print("No experiment logs found to evaluate.", file=sys.stderr)
        return 1

    backend = build_backend(
        backend_name=args.backend,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        allow_missing_api_key=args.allow_missing_api_key,
    )
    backend_metadata = {
        "name": backend.name,
        "base_url": backend.base_url,
        "chat_completions_url": backend.chat_completions_url,
        "api_key_env": backend.api_key_env,
        "api_key_logged": False,
    }
    params = EvaluationParams(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        question_max_tokens=args.question_max_tokens,
        answer_max_tokens=args.answer_max_tokens,
        timeout_seconds=args.timeout_seconds,
    )

    exit_code = 0
    for log_path in log_paths:
        try:
            updated = evaluate_log_path(
                log_path=log_path,
                backend=backend,
                backend_metadata=backend_metadata,
                params=params,
                force=args.force,
                dry_run=args.dry_run,
            )
            if updated:
                print(f"Evaluated: {log_path}")
        except Exception as exc:
            exit_code = 1
            print(f"Failed to evaluate {log_path}: {exc}", file=sys.stderr)
            traceback.print_exc()
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill LLM-as-judge Trace Coverage Faithfulness (TCF) evaluation into experiment logs.",
    )
    parser.add_argument(
        "--experiment-log",
        dest="experiment_logs",
        action="append",
        type=Path,
        default=[],
        help="Explicit experiment log path. Repeat to evaluate multiple logs.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Root logs directory used when --experiment-log is omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute TCF evaluation even if the log already contains one.",
    )
    parser.add_argument(
        "--representative-only",
        action="store_true",
        help=(
            "Restrict evaluation to the representative 10 food/service pairs "
            "used by run_representative_experiments.sh."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["groq", "openai-compatible"],
        default="groq",
        help="LLM backend adapter for evaluation.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI-compatible API base URL. Groq default: {DEFAULT_GROQ_BASE_URL}.",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the API key. Defaults by backend.",
    )
    parser.add_argument(
        "--allow-missing-api-key",
        action="store_true",
        help="Allow no Authorization header for local OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GROQ_MODEL,
        help="Judge model id.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--question-max-tokens", type=int, default=500)
    parser.add_argument("--answer-max-tokens", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which logs would be evaluated without modifying files.",
    )
    return parser.parse_args(argv)


def resolve_log_paths(args: argparse.Namespace) -> list[Path]:
    if args.experiment_logs:
        paths = [path for path in args.experiment_logs if path.exists()]
    else:
        paths = sorted(args.logs_dir.glob("*/*.json"))
    if args.representative_only:
        paths = [path for path in paths if is_representative_log(path)]
    return paths


def evaluate_log_path(
    log_path: Path,
    backend: Any,
    backend_metadata: dict[str, Any],
    params: EvaluationParams,
    force: bool,
    dry_run: bool,
) -> bool:
    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    if not isinstance(log_data, dict):
        raise ValueError("Log file does not contain a JSON object.")
    if not force and evaluations_from_log(log_data):
        return False
    if dry_run:
        return True

    evaluation = evaluate_experiment_log(
        log_data=log_data,
        backend=backend,
        backend_metadata=backend_metadata,
        params=params,
        root_dir=log_path.resolve().parents[2],
    )
    append_evaluation_to_log(log_data, evaluation)
    write_evaluation_to_log(log_path, log_data)
    return True


def is_representative_log(log_path: Path) -> bool:
    try:
        log_data = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(log_data, dict):
        return False

    trace = log_data.get("trace") or {}
    path_metadata = trace.get("path_metadata") or {}
    food = path_metadata.get("food_score")
    service = path_metadata.get("service_score")

    if food is None or service is None:
        trace_path = trace.get("path")
        if trace_path:
            parsed = parse_trace_path(Path(str(trace_path)))
            food = parsed.get("food_score")
            service = parsed.get("service_score")

    if food is None or service is None:
        return False

    try:
        pair = (int(food), int(service))
    except (TypeError, ValueError):
        return False
    return pair in REPRESENTATIVE_CASES


if __name__ == "__main__":
    raise SystemExit(main())
