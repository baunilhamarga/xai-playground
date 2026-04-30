from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT_DIR / "logs"
RULEBASES_DIR = ROOT_DIR / "rulebases"

PLOT_ORDER = [
    "food_memberships",
    "service_memberships",
    "tip_memberships",
]


@dataclass(frozen=True)
class ExperimentRecord:
    path: Path
    data: dict[str, Any]
    timestamp: str
    mode: str
    status: str
    model: str
    backend: str
    trace_model: str | None
    food_score: float | None
    service_score: float | None
    tip_value: float | None
    label: str


@dataclass(frozen=True)
class MetricSummary:
    key: str
    label: str
    status: str
    score: float | None
    timestamp: str | None
    evaluation_count: int
    evaluation: dict[str, Any]


DEFAULT_QUALITY_METRIC = "TCF"
BUILTIN_METRIC_ALIASES = {
    "TCF": {"TCF", "trace_coverage_faithfulness", "faithfulness_score"},
}


def load_experiments(logs_dir: Path = LOGS_DIR) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    for path in sorted(logs_dir.glob("*/*.json"), reverse=True):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        timestamp = str(data.get("timestamp_utc") or "")
        mode = str(data.get("mode") or "unknown")
        status = str(data.get("status") or "unknown")
        generation_parameters = data.get("generation_parameters") or {}
        trace = data.get("trace") or {}
        path_metadata = trace.get("path_metadata") or {}
        trace_summary = trace.get("summary") or {}
        inputs = trace_summary.get("inputs") or {}
        outputs = trace_summary.get("outputs") or {}
        tip_output = outputs.get("tip") or {}

        model = str(generation_parameters.get("model") or "unknown")
        backend = str((data.get("backend") or {}).get("name") or "unknown")
        trace_model = _first_non_empty(
            path_metadata.get("model_name"),
            _trace_model_from_path(str(trace.get("path") or "")),
        )
        food_score = _to_float(_first_non_empty(path_metadata.get("food_score"), inputs.get("food")))
        service_score = _to_float(_first_non_empty(path_metadata.get("service_score"), inputs.get("service")))
        tip_value = _to_float(tip_output.get("value"))

        label = build_experiment_label(
            timestamp=timestamp,
            mode=mode,
            status=status,
            trace_model=trace_model,
            food_score=food_score,
            service_score=service_score,
            model=model,
        )

        records.append(
            ExperimentRecord(
                path=path,
                data=data,
                timestamp=timestamp,
                mode=mode,
                status=status,
                model=model,
                backend=backend,
                trace_model=trace_model,
                food_score=food_score,
                service_score=service_score,
                tip_value=tip_value,
                label=label,
            )
        )

    records.sort(key=lambda record: record.timestamp, reverse=True)
    return records


def build_experiment_label(
    timestamp: str,
    mode: str,
    status: str,
    trace_model: str | None,
    food_score: float | None,
    service_score: float | None,
    model: str,
) -> str:
    stamp = format_timestamp(timestamp)
    scores = f"food={format_score(food_score)} service={format_score(service_score)}"
    trace_name = trace_model or "unknown-trace"
    return f"{stamp} | {mode} | {status} | {trace_name} | {scores} | {model}"


def format_timestamp(timestamp: str) -> str:
    if not timestamp:
        return "unknown-time"
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return timestamp
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def format_score(value: float | None) -> str:
    if value is None:
        return "?"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.2f} %"


def format_seconds(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "Unavailable"
    return f"{numeric:.3f} s"


def format_number(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "Unavailable"
    if float(numeric).is_integer():
        return str(int(numeric))
    return f"{numeric:.2f}"


def prompt_messages(record: ExperimentRecord) -> list[dict[str, str]]:
    prompt = record.data.get("prompt") or {}
    messages = prompt.get("messages") or []
    return [message for message in messages if isinstance(message, dict)]


def evaluation_history(record: ExperimentRecord) -> list[dict[str, Any]]:
    evaluations = record.data.get("evaluations")
    if isinstance(evaluations, list):
        result = [item for item in evaluations if isinstance(item, dict)]
        if result:
            return _sort_evaluations(result)

    evaluation = record.data.get("evaluation") or {}
    if isinstance(evaluation, dict) and evaluation:
        return [evaluation]
    return []


def available_quality_metrics(records: list[ExperimentRecord]) -> list[str]:
    metrics = {DEFAULT_QUALITY_METRIC}
    for record in records:
        for evaluation in evaluation_history(record):
            metric_key = metric_key_from_evaluation(evaluation)
            if metric_key:
                metrics.add(metric_key)
    return sorted(metrics, key=lambda item: (item != DEFAULT_QUALITY_METRIC, item))


def metric_evaluations(record: ExperimentRecord, metric_key: str) -> list[dict[str, Any]]:
    aliases = metric_aliases(metric_key)
    return [
        evaluation
        for evaluation in evaluation_history(record)
        if metric_key_from_evaluation(evaluation) in aliases
        or str(evaluation.get("metric_name") or "") in aliases
        or str(evaluation.get("metric_short_name") or "") in aliases
    ]


def metric_summary(record: ExperimentRecord, metric_key: str) -> MetricSummary:
    evaluations = metric_evaluations(record, metric_key)
    if not evaluations:
        return MetricSummary(
            key=metric_key,
            label=metric_label(metric_key),
            status="missing",
            score=None,
            timestamp=None,
            evaluation_count=0,
            evaluation={},
        )

    latest = evaluations[0]
    raw_status = str(latest.get("status") or "unknown")
    score = _to_float(latest.get("score")) if raw_status == "success" else None
    if raw_status == "success" and score is not None:
        status = "success"
    elif raw_status == "error":
        status = "error"
    elif raw_status == "skipped":
        status = "skipped"
    else:
        status = "missing"

    return MetricSummary(
        key=metric_key,
        label=metric_label(metric_key),
        status=status,
        score=score,
        timestamp=str(latest.get("timestamp_utc") or "") or None,
        evaluation_count=len(evaluations),
        evaluation=latest,
    )


def tcf_evaluations(record: ExperimentRecord) -> list[dict[str, Any]]:
    return metric_evaluations(record, DEFAULT_QUALITY_METRIC)


def tcf_evaluation(record: ExperimentRecord) -> dict[str, Any]:
    evaluations = tcf_evaluations(record)
    return evaluations[0] if evaluations else {}


def tcf_score(record: ExperimentRecord) -> float | None:
    evaluation = tcf_evaluation(record)
    return _to_float(evaluation.get("score"))


def faithfulness_evaluation(record: ExperimentRecord) -> dict[str, Any]:
    return tcf_evaluation(record)


def faithfulness_score(record: ExperimentRecord) -> float | None:
    return tcf_score(record)


def tcf_evaluation_label(evaluation: dict[str, Any], index: int) -> str:
    timestamp = format_timestamp(str(evaluation.get("timestamp_utc") or ""))
    status = str(evaluation.get("status") or "unknown")
    model = str(((evaluation.get("generation_parameters") or {}).get("model")) or "unknown-model")
    score = _to_float(evaluation.get("score"))
    metric_key = metric_key_from_evaluation(evaluation) or DEFAULT_QUALITY_METRIC
    score_text = "no-score" if score is None else f"{metric_key} {score:.2f}"
    newest_suffix = " | newest" if index == 0 else ""
    return f"{timestamp} | {status} | {model} | {score_text}{newest_suffix}"


def llm_trace_input(record: ExperimentRecord) -> str | None:
    for message in reversed(prompt_messages(record)):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        marker = "Trace to explain:\n\n"
        if marker in content:
            return content.split(marker, 1)[1].strip()
    return None


def load_full_trace(record: ExperimentRecord) -> list[dict[str, Any]] | None:
    trace = record.data.get("trace") or {}
    raw_path = trace.get("path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return None
    data = _read_json(path)
    return data if isinstance(data, list) else None


def parse_trace_nodes(trace_nodes: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    if not trace_nodes:
        return {
            "inputs": [],
            "outputs": [],
            "propositions": [],
            "active_propositions": [],
            "expressions": [],
            "active_expressions": [],
            "rules": [],
            "active_rules": [],
        }

    inputs = []
    outputs = []
    propositions = []
    expressions = []
    rules = []

    for node in trace_nodes:
        node_type = str(node.get("nodeType") or "")
        label = str(node.get("string") or "")
        value = _to_float(node.get("value"))
        item = {
            "label": label,
            "value": value,
            "value_text": _string_value(node.get("value")),
            "node_type": node_type,
            "confidence_text": _string_value(node.get("confidence")),
        }
        if "LogicalInput" in node_type:
            inputs.append(item)
        elif "LogicalOutput" in node_type:
            outputs.append(item)
        elif "FuzzyProposition" in node_type:
            propositions.append(item)
        elif "AndExpression" in node_type or "OrExpression" in node_type:
            expressions.append(item)
        elif "MamdaniRule" in node_type:
            rules.append(item)

    return {
        "inputs": inputs,
        "outputs": outputs,
        "propositions": propositions,
        "active_propositions": [item for item in propositions if (item["value"] or 0.0) > 0.0],
        "expressions": expressions,
        "active_expressions": [item for item in expressions if (item["value"] or 0.0) > 0.0],
        "rules": rules,
        "active_rules": [item for item in rules if (item["value"] or 0.0) > 0.0],
    }


def extract_rulebase_name(record: ExperimentRecord) -> str | None:
    trace = record.data.get("trace") or {}
    path_metadata = trace.get("path_metadata") or {}
    return _first_non_empty(
        path_metadata.get("model_name"),
        _trace_model_from_path(str(trace.get("path") or "")),
        _basename((record.data.get("rulebase") or {}).get("source")),
    )


def load_rulebase_resources(
    rulebase_name: str | None,
    trace_details: dict[str, list[dict[str, Any]]] | None = None,
    rulebase_text: str | None = None,
) -> dict[str, Any]:
    description_path = None
    description_text = None
    plots: list[Path] = []
    if rulebase_name:
        description_path = RULEBASES_DIR / rulebase_name / "DESCRIPTION.md"
        if description_path.exists():
            description_text = description_path.read_text(encoding="utf-8")
        plots_dir = RULEBASES_DIR / rulebase_name / "plots"
        if plots_dir.exists():
            plots = sort_plot_paths(list(plots_dir.glob("*.jpg")))

    rules_text = rulebase_text
    if not rules_text and trace_details:
        rules = trace_details.get("rules") or []
        if rules:
            unique_rules: list[str] = []
            seen = set()
            for item in rules:
                label = item["label"]
                if label and label not in seen:
                    seen.add(label)
                    unique_rules.append(label)
            if unique_rules:
                rules_text = "\n".join(
                    f"{index}. {rule}" for index, rule in enumerate(unique_rules, start=1)
                )

    return {
        "name": rulebase_name,
        "description_path": description_path,
        "description_text": description_text,
        "plots": plots,
        "rules_text": rules_text,
    }


def sort_plot_paths(paths: list[Path]) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        if stem in PLOT_ORDER:
            return (PLOT_ORDER.index(stem), stem)
        return (len(PLOT_ORDER), stem)

    return sorted(paths, key=sort_key)


def select_main_parameters(record: ExperimentRecord) -> dict[str, Any]:
    data = record.data
    metrics = data.get("metrics") or {}
    generation_parameters = data.get("generation_parameters") or {}
    trace = data.get("trace") or {}
    path_metadata = trace.get("path_metadata") or {}
    return {
        "mode": record.mode,
        "model": record.model,
        "status": record.status,
        "backend": record.backend,
        "trace_model": _first_non_empty(record.trace_model, path_metadata.get("model_name")),
        "generation_time_seconds": metrics.get("generation_time_seconds"),
        "output_word_count": metrics.get("output_word_count"),
        "output_char_count": metrics.get("output_char_count"),
        "total_tokens": ((metrics.get("usage") or {}).get("total_tokens")),
        "prompt_tokens": ((metrics.get("usage") or {}).get("prompt_tokens")),
        "max_tokens": generation_parameters.get("max_tokens"),
        "temperature": generation_parameters.get("temperature"),
        "top_p": generation_parameters.get("top_p"),
    }


def metric_key_from_evaluation(evaluation: dict[str, Any]) -> str | None:
    metric_short_name = str(evaluation.get("metric_short_name") or "").strip()
    metric_name = str(evaluation.get("metric_name") or "").strip()

    if metric_short_name:
        return metric_short_name
    if metric_name:
        for canonical_key, aliases in BUILTIN_METRIC_ALIASES.items():
            if metric_name in aliases:
                return canonical_key
        return metric_name
    return None


def metric_aliases(metric_key: str) -> set[str]:
    aliases = set(BUILTIN_METRIC_ALIASES.get(metric_key, set()))
    aliases.add(metric_key)
    return aliases


def metric_label(metric_key: str) -> str:
    if metric_key == DEFAULT_QUALITY_METRIC:
        return "Trace Coverage Faithfulness (TCF)"
    return metric_key


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def _trace_model_from_path(trace_path: str) -> str | None:
    if not trace_path:
        return None
    parts = Path(trace_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return None


def _basename(value: Any) -> str | None:
    if not value:
        return None
    return Path(str(value)).name


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


def _sort_evaluations(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        evaluations,
        key=lambda item: str(item.get("timestamp_utc") or ""),
        reverse=True,
    )
