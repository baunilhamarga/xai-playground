"""CLI for fuzzy trace explanation experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from textwrap import dedent
import traceback
from typing import Any

from . import __version__
from .backends import (
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GROQ_MODEL_LABEL,
    BackendError,
    GenerationParams,
    build_backend,
)
from .experiment_logging import JSONExperimentLogger, output_metrics
from .prompts import (
    DEFAULT_3SHOT_EXAMPLES,
    EXPERIMENT_MODES,
    PromptExample,
    build_messages,
    default_instructions_for_mode,
    examples_metadata,
    mode_descriptions_text,
)
from .runtime import runtime_metadata
from .trace_loader import (
    DEFAULT_FOOD_SCORE,
    DEFAULT_SERVICE_SCORE,
    DEFAULT_TRACE_MODEL,
    DEFAULT_TRACES_DIR,
    Rulebase,
    build_trace_path,
    format_rulebase_section,
    format_trace,
    infer_rulebase,
    load_rulebase_file,
    load_trace,
    parse_trace_path,
    sha256_file,
    sha256_text,
    summarize_trace,
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = (
        list(EXPERIMENT_MODES.values())
        if args.mode == "all"
        else [EXPERIMENT_MODES[args.mode]]
    )

    exit_code = 0
    log_paths: list[Path] = []
    for mode in modes:
        try:
            result = run_single_experiment(args, mode.name)
        except Exception as exc:
            print(f"Experiment failed before logging: {exc}", file=sys.stderr)
            traceback.print_exc()
            return 1

        if result.get("log_path"):
            log_paths.append(Path(str(result["log_path"])))
        if result.get("status") != "success":
            exit_code = 1

    if args.dry_run:
        return 0

    for path in log_paths:
        print(f"Logged experiment: {path}")
    return exit_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate natural-language explanations from fuzzy-logic tip traces.",
        epilog=dedent(
            f"""
            {mode_descriptions_text()}
            """
        ).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=[*EXPERIMENT_MODES.keys(), "all"],
        default="zero-shot+trace",
        help=(
            "Experiment mode to run. Use 'all' to run every defined mode. "
            "See the mode descriptions below."
        ),
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help=(
            "Explicit trace JSON path. If omitted, the path is built from "
            "--traces-dir, --trace-model, --food, and --service."
        ),
    )
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--trace-model", default=DEFAULT_TRACE_MODEL)
    parser.add_argument("--food", type=int, default=DEFAULT_FOOD_SCORE)
    parser.add_argument("--service", type=int, default=DEFAULT_SERVICE_SCORE)
    parser.add_argument(
        "--rulebase-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit rulebase text file for the rulebase mode. "
            "If omitted, the rulebase is inferred from MamdaniRule nodes in "
            "the selected trace model directory."
        ),
    )
    parser.add_argument(
        "--instructions-file",
        type=Path,
        default=None,
        help="Optional file replacing the default explanation instructions.",
    )
    parser.add_argument(
        "--examples-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON examples file replacing the built-in 3-shot examples. "
            "Use a list or {'examples': [...]} with fields id, trace or trace_path, "
            "and explanation."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["groq", "openai-compatible"],
        default="groq",
        help="LLM backend adapter. Default uses Groq's OpenAI-compatible API.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI-compatible API base URL. Groq default: "
            f"{DEFAULT_GROQ_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the API key. Default: GROQ_API_KEY.",
    )
    parser.add_argument(
        "--allow-missing-api-key",
        action="store_true",
        help="Allow no Authorization header for local OpenAI-compatible servers.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GROQ_MODEL,
        help=(
            f"Model id. Default maps the requested {DEFAULT_GROQ_MODEL_LABEL} "
            f"model to Groq id {DEFAULT_GROQ_MODEL}."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assembled messages as JSON and skip backend calls and logging.",
    )
    return parser.parse_args(argv)


def run_single_experiment(args: argparse.Namespace, mode_name: str) -> dict[str, Any]:
    cwd = Path.cwd()
    mode = EXPERIMENT_MODES[mode_name]
    trace_path = (
        args.trace_path
        if args.trace_path
        else build_trace_path(args.traces_dir, args.trace_model, args.food, args.service)
    )

    trace = load_trace(trace_path)
    trace_text = format_trace(trace)
    trace_summary = summarize_trace(trace)
    trace_path_metadata = parse_trace_path(trace_path)

    rulebase: Rulebase | None = None
    prompt_trace_text = trace_text
    if mode.include_rulebase:
        rulebase = (
            load_rulebase_file(args.rulebase_file)
            if args.rulebase_file
            else infer_rulebase(trace_path.parent, seed_trace_path=trace_path)
        )
        prompt_trace_text = f"{format_rulebase_section(rulebase)}\n\n{trace_text}"

    instructions = (
        args.instructions_file.read_text(encoding="utf-8")
        if args.instructions_file
        else default_instructions_for_mode(mode)
    )
    examples = load_examples(args.examples_file) if mode.include_examples else []
    messages = build_messages(instructions, prompt_trace_text, examples)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": mode.name,
                    "trace_path": str(trace_path),
                    "messages": messages,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"status": "dry_run", "log_path": None}

    params = GenerationParams(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    backend = build_backend(
        backend_name=args.backend,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        allow_missing_api_key=args.allow_missing_api_key,
    )

    prompt_char_count = sum(len(message["content"]) for message in messages)
    record: dict[str, Any] = {
        "experiment_id": None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "framework_version": __version__,
        "mode": mode.name,
        "mode_description": mode.description,
        "trace_variant": mode.trace_variant,
        "status": "started",
        "cli": {
            "argv": sys.argv,
            "parameters": _serializable_args(args),
        },
        "runtime": runtime_metadata(cwd),
        "backend": {
            "name": backend.name,
            "base_url": backend.base_url,
            "chat_completions_url": backend.chat_completions_url,
            "api_key_env": backend.api_key_env,
            "api_key_logged": False,
        },
        "generation_parameters": {
            "model": params.model,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "timeout_seconds": params.timeout_seconds,
        },
        "prompt": {
            "instructions_source": (
                str(args.instructions_file) if args.instructions_file else "builtin"
            ),
            "instructions_sha256": sha256_text(instructions),
            "messages": messages,
            "message_count": len(messages),
            "char_count": prompt_char_count,
        },
        "examples": (
            examples_metadata(
                examples,
                source=str(args.examples_file) if args.examples_file else "builtin",
            )
            if examples
            else {"source": None, "count": 0}
        ),
        "trace": {
            "path": str(trace_path),
            "path_convention": "traces/<model_name>/tip_trace_food<foodScore>_service<serviceScore>.json",
            "format": mode.trace_variant,
            "path_metadata": trace_path_metadata,
            "sha256": sha256_file(trace_path),
            "formatted_sha256": sha256_text(trace_text),
            "formatted_char_count": len(trace_text),
            "summary": {
                "inputs": trace_summary.inputs,
                "outputs": trace_summary.outputs,
                "proposition_count": trace_summary.proposition_count,
                "active_proposition_count": trace_summary.active_proposition_count,
                "rule_count": trace_summary.rule_count,
                "active_rule_count": trace_summary.active_rule_count,
            },
        },
        "rulebase": (
            {
                "source": rulebase.source,
                "method": rulebase.method,
                "rule_count": len(rulebase.rules),
                "sha256": sha256_text(rulebase.text),
                "text": rulebase.text,
            }
            if rulebase
            else {"source": None, "method": None, "rule_count": 0}
        ),
        "metrics": {
            "prompt_char_count": prompt_char_count,
            "prompt_message_count": len(messages),
            "trace_char_count": len(trace_text),
            "example_count": len(examples),
            "active_proposition_count": trace_summary.active_proposition_count,
            "rulebase_rule_count": len(rulebase.rules) if rulebase else 0,
        },
    }

    try:
        result = backend.generate(messages, params)
        record["status"] = "success"
        record["output"] = {
            "text": result.text,
            "metrics": output_metrics(result.text),
            "response_metadata": result.response_metadata,
        }
        record["metrics"]["generation_time_seconds"] = result.elapsed_seconds
        record["metrics"]["output_char_count"] = len(result.text)
        record["metrics"]["output_word_count"] = len(result.text.split())
        usage = result.response_metadata.get("usage") or {}
        if usage:
            record["metrics"]["usage"] = usage
    except BackendError as exc:
        record["status"] = "error"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    log_path = JSONExperimentLogger(args.logs_dir).log(record)
    return {"status": record["status"], "log_path": log_path}


def load_examples(path: Path | None) -> list[PromptExample]:
    if path is None:
        return list(DEFAULT_3SHOT_EXAMPLES)

    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("examples") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError("Examples JSON must be a list or an object with an 'examples' list.")

    examples: list[PromptExample] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Example {index} is not an object.")
        trace_text = entry.get("trace")
        if trace_text is None and entry.get("trace_path"):
            trace_text = format_trace(load_trace(Path(str(entry["trace_path"]))))
        explanation = entry.get("explanation")
        if not trace_text or not explanation:
            raise ValueError(
                f"Example {index} must include 'trace' or 'trace_path', and 'explanation'."
            )
        examples.append(
            PromptExample(
                example_id=str(entry.get("id") or f"example_{index}"),
                trace=str(trace_text),
                explanation=str(explanation),
            )
        )
    return examples


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
