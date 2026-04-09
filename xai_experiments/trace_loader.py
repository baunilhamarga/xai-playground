"""Trace loading, formatting, and rulebase extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_TRACE_PATH = Path("traces/heitor2/tip_trace_food2_service7.json")
DEFAULT_TRACES_DIR = Path("traces")
DEFAULT_TRACE_MODEL = "heitor2"
DEFAULT_FOOD_SCORE = 2
DEFAULT_SERVICE_SCORE = 7

INPUT_DOMAINS = {
    "food": "[0 ; 10]",
    "service": "[0 ; 10]",
}
OUTPUT_DOMAINS = {
    "tip": "[0 % ; 30 %]",
}
ACTIVATION_DOMAIN = "[0 ; 1]"

TRACE_FILENAME_RE = re.compile(
    r"tip_trace_food(?P<food>\d+)_service(?P<service>\d+)\.json$"
)


@dataclass(frozen=True)
class TraceSummary:
    inputs: dict[str, str]
    outputs: dict[str, dict[str, str | None]]
    proposition_count: int
    active_proposition_count: int
    rule_count: int
    active_rule_count: int


@dataclass(frozen=True)
class Rulebase:
    text: str
    rules: list[str]
    source: str
    method: str


def build_trace_path(
    traces_dir: Path,
    model_name: str,
    food_score: int,
    service_score: int,
) -> Path:
    return traces_dir / model_name / f"tip_trace_food{food_score}_service{service_score}.json"


def parse_trace_path(path: Path) -> dict[str, str | int | None]:
    match = TRACE_FILENAME_RE.search(path.name)
    return {
        "model_name": path.parent.name if path.parent.name else None,
        "food_score": int(match.group("food")) if match else None,
        "service_score": int(match.group("service")) if match else None,
    }


def load_trace(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Trace must be a JSON array: {path}")
    for index, node in enumerate(data):
        if not isinstance(node, dict):
            raise ValueError(f"Trace node {index} is not an object: {path}")
    return data


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def node_type_contains(node: dict[str, Any], marker: str) -> bool:
    return marker in str(node.get("nodeType", ""))


def node_value(node: dict[str, Any]) -> str:
    value = node.get("value")
    return "" if value is None else str(value)


def node_label(node: dict[str, Any]) -> str:
    label = node.get("string")
    return "" if label is None else str(label)


def format_trace(trace: list[dict[str, Any]]) -> str:
    inputs = [node for node in trace if node_type_contains(node, "LogicalInput")]
    outputs = [node for node in trace if node_type_contains(node, "LogicalOutput")]
    propositions = active_fuzzy_propositions(trace)

    if not inputs:
        raise ValueError("Trace does not contain any LogicalInput nodes.")
    if not outputs:
        raise ValueError("Trace does not contain any LogicalOutput nodes.")
    if not propositions:
        raise ValueError("Trace does not contain any active FuzzyProposition nodes.")

    lines: list[str] = ["<=== INPUT ===>"]
    for node in inputs:
        name = node_label(node)
        domain = INPUT_DOMAINS.get(name, "[unknown]")
        lines.append(f"[{name}, {node_value(node)}] ; Definition Domain : {domain}")

    lines.extend(["", "<=== OUTPUT ===>"])
    for node in outputs:
        name = node_label(node)
        domain = OUTPUT_DOMAINS.get(name, "[unknown]")
        lines.append(f"({name}, {node_value(node)}) ; Definition Domain : {domain}")

    lines.extend(["", "<=== EXPLANATION ===>"])
    for node in propositions:
        lines.append(
            f"({node_label(node)}, Activation Value: {node_value(node)}) "
            f"; Definition Domain : {ACTIVATION_DOMAIN}"
        )

    return "\n".join(lines)


def summarize_trace(trace: list[dict[str, Any]]) -> TraceSummary:
    inputs = {
        node_label(node): node_value(node)
        for node in trace
        if node_type_contains(node, "LogicalInput")
    }
    outputs = {
        node_label(node): {
            "value": node_value(node),
            "confidence": (
                None
                if node.get("confidence") is None
                else str(node.get("confidence"))
            ),
        }
        for node in trace
        if node_type_contains(node, "LogicalOutput")
    }
    propositions = [
        node for node in trace if node_type_contains(node, "FuzzyProposition")
    ]
    active_propositions = active_fuzzy_propositions(trace)
    rules = [node for node in trace if node_type_contains(node, "MamdaniRule")]
    active_rules = [node for node in rules if _to_float_or_none(node_value(node)) not in (0.0, None)]
    return TraceSummary(
        inputs=inputs,
        outputs=outputs,
        proposition_count=len(propositions),
        active_proposition_count=len(active_propositions),
        rule_count=len(rules),
        active_rule_count=len(active_rules),
    )


def load_rulebase_file(path: Path) -> Rulebase:
    text = path.read_text(encoding="utf-8").strip()
    rules = [line.strip() for line in text.splitlines() if line.strip()]
    return Rulebase(text=text, rules=rules, source=str(path), method="file")


def infer_rulebase(
    model_trace_dir: Path,
    seed_trace_path: Path | None = None,
) -> Rulebase:
    paths = sorted(model_trace_dir.glob("tip_trace_food*_service*.json"))
    if seed_trace_path and seed_trace_path.exists():
        seed_trace_path = seed_trace_path.resolve()
        paths = [seed_trace_path] + [
            path for path in paths if path.resolve() != seed_trace_path
        ]

    seen: set[str] = set()
    rules: list[str] = []
    for path in paths:
        for node in load_trace(path):
            if not node_type_contains(node, "MamdaniRule"):
                continue
            rule = node_label(node)
            if rule and rule not in seen:
                seen.add(rule)
                rules.append(rule)

    if not rules:
        raise ValueError(f"No MamdaniRule nodes found under {model_trace_dir}")

    text = "\n".join(f"{index}. {rule}" for index, rule in enumerate(rules, start=1))
    return Rulebase(
        text=text,
        rules=rules,
        source=str(model_trace_dir),
        method="inferred_from_trace_directory",
    )


def format_rulebase_section(rulebase: Rulebase) -> str:
    return f"<=== MODEL RULEBASE ===>\n{rulebase.text}"


def active_fuzzy_propositions(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    propositions = [
        node for node in trace if node_type_contains(node, "FuzzyProposition")
    ]
    active = [
        node
        for node in propositions
        if (_to_float_or_none(node_value(node)) or 0.0) > 0.0
    ]
    return active if active else propositions


def _to_float_or_none(value: str) -> float | None:
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None
