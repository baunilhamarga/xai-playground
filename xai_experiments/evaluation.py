"""LLM-as-judge Trace Coverage Faithfulness (TCF) evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from . import __version__
from .backends import ChatBackend, GenerationParams
from .trace_loader import format_trace, load_trace, sha256_text


SECTION_RE = re.compile(r"^<===\s*(?P<section>.+?)\s*===>$")


@dataclass(frozen=True)
class TraceFact:
    line_id: str
    section: str
    text: str


@dataclass(frozen=True)
class EvaluationParams:
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    question_max_tokens: int = 500
    answer_max_tokens: int = 200
    timeout_seconds: int = 120

    def question_generation_params(self) -> GenerationParams:
        return GenerationParams(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.question_max_tokens,
            timeout_seconds=self.timeout_seconds,
        )

    def answer_generation_params(self) -> GenerationParams:
        return GenerationParams(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.answer_max_tokens,
            timeout_seconds=self.timeout_seconds,
        )


def extract_trace_facts(trace_text: str) -> list[TraceFact]:
    facts: list[TraceFact] = []
    section = ""
    counts: dict[str, int] = {}
    for raw_line in trace_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            section = section_match.group("section").strip()
            continue
        if not section:
            continue
        counts[section] = counts.get(section, 0) + 1
        facts.append(
            TraceFact(
                line_id=f"{section.lower()}_{counts[section]}",
                section=section,
                text=line,
            )
        )
    return facts


def run_trace_coverage_faithfulness_evaluation(
    trace_text: str,
    explanation_text: str,
    backend: ChatBackend,
    backend_metadata: dict[str, Any],
    params: EvaluationParams,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    facts = extract_trace_facts(trace_text)

    evaluation: dict[str, Any] = {
        "status": "started",
        "timestamp_utc": timestamp,
        "framework_version": __version__,
        "metric_name": "trace_coverage_faithfulness",
        "metric_short_name": "TCF",
        "trace_sha256": sha256_text(trace_text),
        "trace_fact_count": len(facts),
        "trace_facts": [
            {"line_id": fact.line_id, "section": fact.section, "text": fact.text}
            for fact in facts
        ],
        "backend": backend_metadata,
        "generation_parameters": {
            "model": params.model,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "question_max_tokens": params.question_max_tokens,
            "answer_max_tokens": params.answer_max_tokens,
            "timeout_seconds": params.timeout_seconds,
        },
    }

    if not explanation_text.strip():
        evaluation["status"] = "skipped"
        evaluation["skip_reason"] = "empty_explanation"
        evaluation["score"] = None
        evaluation["question_generation"] = {"status": "skipped"}
        evaluation["judge"] = {"status": "skipped", "results": []}
        return evaluation

    fixed_questions = build_fixed_questions(facts)
    generated_facts = [fact for fact in facts if fact.section == "EXPLANATION"]

    question_result = None
    raw_generated_questions: list[dict[str, str]] = []
    generated_questions: list[dict[str, str]] = []
    question_normalization: list[dict[str, Any]] = []
    normalized_generated_question_text = "[]"

    if generated_facts:
        question_messages = build_question_generation_messages(trace_text, generated_facts)
        question_result = backend.generate(
            question_messages,
            params.question_generation_params(),
        )
        (
            raw_generated_questions,
            normalized_generated_question_text,
        ) = parse_question_generation_response(
            question_result.text,
            generated_facts,
        )
        generated_questions = normalize_explanation_questions(
            generated_facts,
            raw_generated_questions,
        )
        question_normalization = describe_question_normalization(
            generated_facts,
            raw_generated_questions,
            generated_questions,
        )
        normalized_generated_question_text = json.dumps(
            generated_questions,
            ensure_ascii=False,
            sort_keys=True,
        )
        question_generation_elapsed = question_result.elapsed_seconds
        question_generation_response_text = question_result.text
        question_generation_response_metadata = question_result.response_metadata
    else:
        question_messages = []
        question_generation_elapsed = 0.0
        question_generation_response_text = ""
        question_generation_response_metadata = {}

    questions = merge_questions(facts, fixed_questions, generated_questions)
    evaluation["question_generation"] = {
        "status": "success",
        "messages": question_messages,
        "response_text": question_generation_response_text,
        "response_sha256": (
            sha256_text(question_generation_response_text)
            if question_generation_response_text
            else None
        ),
        "response_metadata": question_generation_response_metadata,
        "elapsed_seconds": question_generation_elapsed,
        "fixed_questions": fixed_questions,
        "fixed_question_count": len(fixed_questions),
        "raw_generated_questions": raw_generated_questions,
        "generated_questions": generated_questions,
        "generated_question_count": len(generated_questions),
        "question_normalization": question_normalization,
        "questions": questions,
        "question_count": len(questions),
        "normalized_generated_questions_sha256": sha256_text(
            normalized_generated_question_text
        ),
    }

    yes_count = 0
    judge_results: list[dict[str, Any]] = []
    total_elapsed = question_generation_elapsed

    for question in questions:
        fact = next(fact for fact in facts if fact.line_id == question["line_id"])
        judge_messages = build_judge_messages(
            trace_text=trace_text,
            explanation_text=explanation_text,
            fact=fact,
            question=question["question"],
        )
        judge_result = backend.generate(
            judge_messages,
            params.answer_generation_params(),
        )
        total_elapsed += judge_result.elapsed_seconds
        answer = parse_judge_response(judge_result.text)
        is_yes = answer["answer"] == "yes"
        if is_yes:
            yes_count += 1
        judge_results.append(
            {
                "line_id": fact.line_id,
                "section": fact.section,
                "trace_line": fact.text,
                "question": question["question"],
                "messages": judge_messages,
                "raw_response_text": judge_result.text,
                "response_metadata": judge_result.response_metadata,
                "elapsed_seconds": judge_result.elapsed_seconds,
                "answer": answer["answer"],
                "reason": answer["reason"],
                "is_yes": is_yes,
            }
        )

    total_questions = len(judge_results)
    score = yes_count / total_questions if total_questions else 0.0
    evaluation["status"] = "success"
    evaluation["score"] = score
    evaluation["yes_count"] = yes_count
    evaluation["total_questions"] = total_questions
    evaluation["judge"] = {
        "status": "success",
        "results": judge_results,
        "yes_count": yes_count,
        "no_count": total_questions - yes_count,
        "total_elapsed_seconds": total_elapsed,
    }
    return evaluation


def evaluate_experiment_log(
    log_data: dict[str, Any],
    backend: ChatBackend,
    backend_metadata: dict[str, Any],
    params: EvaluationParams,
    root_dir: Path,
) -> dict[str, Any]:
    output = log_data.get("output") or {}
    explanation_text = str(output.get("text") or "")
    trace_data = log_data.get("trace") or {}
    trace_path = Path(str(trace_data.get("path") or ""))
    if not trace_path:
        raise ValueError("Experiment log does not contain a trace path.")
    if not trace_path.is_absolute():
        trace_path = root_dir / trace_path
    trace_text = format_trace(load_trace(trace_path))
    return run_trace_coverage_faithfulness_evaluation(
        trace_text=trace_text,
        explanation_text=explanation_text,
        backend=backend,
        backend_metadata=backend_metadata,
        params=params,
    )


def write_evaluation_to_log(log_path: Path, log_data: dict[str, Any]) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(log_data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def evaluations_from_log(log_data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_evaluations = log_data.get("evaluations")
    if isinstance(raw_evaluations, list):
        evaluations = [item for item in raw_evaluations if isinstance(item, dict)]
        if evaluations:
            return sort_evaluations(evaluations)

    legacy_evaluation = log_data.get("evaluation")
    if isinstance(legacy_evaluation, dict):
        return [legacy_evaluation]
    return []


def latest_evaluation(log_data: dict[str, Any]) -> dict[str, Any] | None:
    evaluations = evaluations_from_log(log_data)
    return evaluations[0] if evaluations else None


def append_evaluation_to_log(
    log_data: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    evaluations = evaluations_from_log(log_data)
    evaluations.append(evaluation)
    evaluations = sort_evaluations(evaluations)
    log_data["evaluations"] = evaluations
    log_data["evaluation"] = evaluations[0]
    update_tcf_metrics(log_data)
    return log_data


def update_tcf_metrics(log_data: dict[str, Any]) -> None:
    metrics = log_data.setdefault("metrics", {})
    for key in (
        "tcf_score",
        "tcf_yes_count",
        "tcf_total_questions",
        "tcf_evaluation_count",
    ):
        metrics.pop(key, None)

    evaluations = evaluations_from_log(log_data)
    metrics["tcf_evaluation_count"] = len(evaluations)
    latest = evaluations[0] if evaluations else None
    if latest and latest.get("status") == "success":
        metrics["tcf_score"] = latest.get("score")
        metrics["tcf_yes_count"] = latest.get("yes_count")
        metrics["tcf_total_questions"] = latest.get("total_questions")


def sort_evaluations(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        evaluations,
        key=lambda item: str(item.get("timestamp_utc") or ""),
        reverse=True,
    )


def build_question_generation_messages(
    trace_text: str,
    facts: list[TraceFact],
) -> list[dict[str, str]]:
    fact_lines = "\n".join(
        f'- {fact.line_id} | {fact.section} | {fact.text}' for fact in facts
    )
    return [
        {
            "role": "system",
            "content": (
                "You generate yes/no Trace Coverage Faithfulness (TCF) questions from reduced fuzzy "
                "execution traces. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Reduced trace:\n"
                f"{trace_text}\n\n"
                "Atomic trace facts:\n"
                f"{fact_lines}\n\n"
                "Task:\n"
                "- Create exactly one yes/no question for each listed trace fact.\n"
                "- Keep each question focused on the core information in that fact.\n"
                "- These remaining facts are all from the EXPLANATION section.\n"
                "- Every valid question must be a positive coverage check: if a faithful explanation "
                "communicates the fact, the correct answer should be YES.\n"
                "- Ask whether the explanation communicates the trace fact. Do not ask whether the "
                "trace fact itself is true in the world.\n"
                "- Use the exact linguistic label from the fact as the basis of the question.\n"
                "- Preserve the direction and polarity of the fact. Do not invert it, negate it, "
                "or replace it with an alternative label.\n"
                "- Do not ask about an opposite or contradictory label.\n"
                "- Do not replace one label with a different label from the rulebase.\n"
                "- A question may allow equivalent wording, but it must preserve the original trace meaning.\n"
                "- Do not require the explanation to repeat activation values, domains, or formatting.\n"
                "- Start the question with 'Does the explanation ...'.\n"
                "- The question must be answerable by checking whether the explanation communicates that fact.\n\n"
                "Return JSON with this shape:\n"
                '{\n'
                '  "questions": [\n'
                '    {"line_id": "input_1", "question": "Does the explanation say ...?"}\n'
                "  ]\n"
                "}\n"
                "Use every provided line_id exactly once and do not add extra fields."
            ),
        },
    ]


def build_judge_messages(
    trace_text: str,
    explanation_text: str,
    fact: TraceFact,
    question: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict TCF judge. Decide whether a generated "
                "explanation includes the core information asked by a yes/no question. "
                "Each question is intended to be a positive coverage check, so answer yes "
                "only when the explanation clearly communicates the same trace fact. "
                "Paraphrases count as present. Exact activation values, domains, and "
                "formatting are not required unless the explanation contradicts the "
                "core fact. For INPUT score questions, count the answer as yes if the "
                "explanation clearly uses the underlying input information even when "
                "it describes a qualitative interpretation instead of repeating the raw "
                "number. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Reduced trace:\n"
                f"{trace_text}\n\n"
                f"Atomic fact ({fact.line_id} | {fact.section}):\n"
                f"{fact.text}\n\n"
                "Generated explanation:\n"
                f"{explanation_text}\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Return JSON with this shape:\n"
                '{\n'
                '  "answer": "yes" or "no",\n'
                '  "reason": "short justification"\n'
                "}\n"
                "Answer yes only if the explanation clearly communicates the core "
                "information needed by the question."
            ),
        },
    ]


def parse_question_generation_response(
    text: str,
    facts: list[TraceFact],
) -> tuple[list[dict[str, str]], str]:
    data = _parse_json_payload(text)
    questions = data.get("questions") if isinstance(data, dict) else data
    if not isinstance(questions, list):
        raise ValueError("Question generation response did not contain a questions list.")

    by_id: dict[str, str] = {}
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question item {index} is not an object.")
        line_id = str(item.get("line_id") or "").strip()
        question = str(item.get("question") or "").strip()
        if not line_id or not question:
            raise ValueError(f"Question item {index} is missing line_id or question.")
        by_id[line_id] = question

    ordered_questions: list[dict[str, str]] = []
    for fact in facts:
        question = by_id.get(fact.line_id)
        if not question:
            raise ValueError(f"Missing generated question for {fact.line_id}.")
        ordered_questions.append({"line_id": fact.line_id, "question": question})

    normalized_text = json.dumps(ordered_questions, ensure_ascii=False, sort_keys=True)
    return ordered_questions, normalized_text


def build_fixed_questions(facts: list[TraceFact]) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for fact in facts:
        if fact.section == "INPUT":
            variable, value = parse_input_fact(fact.text)
            questions.append(
                {
                    "line_id": fact.line_id,
                    "question": (
                        f"Does the explanation use the {variable} input information "
                        f"from the trace ({variable} = {value} out of 10), either by "
                        "stating the score or by clearly describing the corresponding "
                        f"{variable} quality?"
                    ),
                }
            )
        elif fact.section == "OUTPUT":
            variable, value = parse_output_fact(fact.text)
            questions.append(
                {
                    "line_id": fact.line_id,
                    "question": (
                        f"Does the explanation state or clearly communicate the "
                        f"recommended {variable} value from the trace ({variable} = {value})?"
                    ),
                }
            )
    return questions


def merge_questions(
    facts: list[TraceFact],
    fixed_questions: list[dict[str, str]],
    generated_questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_line_id = {
        question["line_id"]: question
        for question in [*fixed_questions, *generated_questions]
    }
    merged: list[dict[str, str]] = []
    for fact in facts:
        question = by_line_id.get(fact.line_id)
        if not question:
            raise ValueError(f"Missing question for {fact.line_id}.")
        merged.append(question)
    return merged


def parse_input_fact(text: str) -> tuple[str, str]:
    match = re.match(r"^\[(?P<name>[^,\]]+),\s*(?P<value>[^\]]+)\]", text)
    if not match:
        return ("input", text)
    return (match.group("name").strip(), match.group("value").strip())


def parse_output_fact(text: str) -> tuple[str, str]:
    if not (text.startswith("(") and ")" in text):
        return ("output", text)
    inner = text[1 : text.index(")")]
    if "," not in inner:
        return ("output", inner.strip())
    name, value = inner.split(",", 1)
    return (name.strip(), value.strip())


def normalize_explanation_questions(
    facts: list[TraceFact],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    by_line_id = {question["line_id"]: question for question in questions}
    for fact in facts:
        question = by_line_id[fact.line_id]
        if explanation_question_is_valid(fact, question["question"]):
            normalized.append(question)
        else:
            normalized.append(build_explanation_fallback_question(fact))
    return normalized


def describe_question_normalization(
    facts: list[TraceFact],
    raw_questions: list[dict[str, str]],
    normalized_questions: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw_by_line_id = {question["line_id"]: question for question in raw_questions}
    normalized_by_line_id = {
        question["line_id"]: question for question in normalized_questions
    }
    entries: list[dict[str, Any]] = []
    for fact in facts:
        raw_question = str(raw_by_line_id[fact.line_id]["question"])
        normalized_question = str(normalized_by_line_id[fact.line_id]["question"])
        entries.append(
            {
                "line_id": fact.line_id,
                "section": fact.section,
                "trace_line": fact.text,
                "raw_question": raw_question,
                "normalized_question": normalized_question,
                "replaced": raw_question != normalized_question,
                "replacement_reason": (
                    None
                    if raw_question == normalized_question
                    else invalid_explanation_question_reason(fact, raw_question)
                ),
            }
        )
    return entries


def explanation_question_is_valid(fact: TraceFact, question: str) -> bool:
    return (
        explanation_question_preserves_label(fact, question)
        and explanation_question_is_coverage_oriented(question)
        and not explanation_question_negates_fact(fact, question)
    )


def explanation_question_preserves_label(fact: TraceFact, question: str) -> bool:
    label = extract_explanation_label(fact.text).lower()
    question_lower = question.lower()
    if label and label in question_lower:
        return True
    subject, descriptor = split_label_subject_descriptor(label)
    return bool(subject and descriptor and subject in question_lower and descriptor in question_lower)


def explanation_question_is_coverage_oriented(question: str) -> bool:
    question_lower = question.lower()
    if "explanation" not in question_lower:
        return False
    coverage_verbs = (
        "mention",
        "communicate",
        "describe",
        "state",
        "say",
        "indicate",
        "reflect",
    )
    return any(verb in question_lower for verb in coverage_verbs)


def explanation_question_negates_fact(fact: TraceFact, question: str) -> bool:
    label = extract_explanation_label(fact.text).lower()
    subject, descriptor = split_label_subject_descriptor(label)
    question_lower = question.lower()

    negation_patterns = [
        rf"{re.escape(subject)}\s+is\s+not\s+{re.escape(descriptor)}",
        rf"not\s+{re.escape(label)}",
        rf"isn['’]?t\s+{re.escape(descriptor)}",
        rf"doesn['’]?t\s+(?:mention|say|state|indicate|describe|communicate).+{re.escape(descriptor)}",
    ]
    return any(re.search(pattern, question_lower) for pattern in negation_patterns if subject or descriptor)


def invalid_explanation_question_reason(fact: TraceFact, question: str) -> str:
    reasons: list[str] = []
    if not explanation_question_preserves_label(fact, question):
        reasons.append("does_not_preserve_trace_label")
    if not explanation_question_is_coverage_oriented(question):
        reasons.append("not_a_positive_coverage_check")
    if explanation_question_negates_fact(fact, question):
        reasons.append("negates_or_inverts_trace_fact")
    return ", ".join(reasons) or "failed_validation"


def build_explanation_fallback_question(fact: TraceFact) -> dict[str, str]:
    label = extract_explanation_label(fact.text)
    return {
        "line_id": fact.line_id,
        "question": (
            f"Does the explanation mention that {label}, or clearly communicate "
            "an equivalent meaning?"
        ),
    }


def extract_explanation_label(text: str) -> str:
    match = re.match(r"^\((?P<label>.+?),\s*Activation Value:", text)
    if match:
        return match.group("label").strip()
    stripped = text.strip()
    if stripped.startswith("(") and ")" in stripped:
        return stripped[1 : stripped.index(")")].strip()
    return stripped


def split_label_subject_descriptor(label: str) -> tuple[str, str]:
    if " is " not in label:
        return (label, "")
    subject, descriptor = label.split(" is ", 1)
    return (subject.strip(), descriptor.strip())


def parse_judge_response(text: str) -> dict[str, str]:
    try:
        data = _parse_json_payload(text)
    except ValueError:
        return _parse_freeform_judge_response(text)

    if not isinstance(data, dict):
        return _parse_freeform_judge_response(text)

    answer = str(data.get("answer") or "").strip().lower()
    if answer not in {"yes", "no"}:
        return _parse_freeform_judge_response(text)

    reason = str(data.get("reason") or "").strip()
    if not reason:
        reason = _fallback_reason_text(text, default=f"Parsed {answer} from JSON response.")
    return {"answer": answer, "reason": reason}


def _parse_json_payload(text: str) -> Any:
    stripped = _strip_markdown_fences(text.strip())
    candidates = [stripped]
    first_object = stripped.find("{")
    last_object = stripped.rfind("}")
    if first_object != -1 and last_object != -1 and last_object > first_object:
        candidates.append(stripped[first_object : last_object + 1])
    first_array = stripped.find("[")
    last_array = stripped.rfind("]")
    if first_array != -1 and last_array != -1 and last_array > first_array:
        candidates.append(stripped[first_array : last_array + 1])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON payload from LLM response.")


def _parse_freeform_judge_response(text: str) -> dict[str, str]:
    stripped = _strip_markdown_fences(text.strip())
    answer = _extract_yes_no_answer(stripped)
    if answer is None:
        raise ValueError("Judge response did not contain a valid yes/no answer.")
    return {
        "answer": answer,
        "reason": _fallback_reason_text(
            stripped,
            default=f"Parsed {answer} from non-JSON judge response.",
        ),
    }


def _extract_yes_no_answer(text: str) -> str | None:
    patterns = [
        r"answer\s*[:=-]\s*(yes|no)\b",
        r"verdict\s*[:=-]\s*(yes|no)\b",
        r"^\s*[*_`#>\-\d\.\)\s]*(yes|no)\b",
        r"\b(yes|no)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).lower()
    return None


def _fallback_reason_text(text: str, default: str) -> str:
    match = re.search(r"reason\s*[:=-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        reason = match.group(1).strip()
        if reason:
            return reason
    compact = " ".join(text.split())
    if compact:
        return compact[:300]
    return default


def _strip_markdown_fences(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return text
