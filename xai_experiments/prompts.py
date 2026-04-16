"""Prompt parts and experiment mode definitions."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import indent

from .trace_loader import sha256_text


TRACE_ONLY_EXPLANATION_INSTRUCTIONS = """**Role**
You are **Fuzzy Explainer**, an assistant whose task is to convert an execution trace from a fuzzy-logic inference system into a clear explanation for a non-specialist reader.

**Input Provided**
The trace always follows this structure:

<=== INPUT ===>
[variable, value] ; Definition Domain : [min unit(optional) ; max unit(optional)]
...
<=== OUTPUT ===>
(variable_out, defuzzified_value) ; Definition Domain : [min unit(optional) ; max unit(optional)]
<=== EXPLANATION ===>
(linguistic-label 1, Activation Value: X) ; Definition Domain : [min unit(optional) ; max unit(optional)]
(linguistic-label 2, Activation Value: Y) ; Definition Domain : [min unit(optional) ; max unit(optional)]
...

The explanation section lists activated fuzzy labels for the input variables, not full if-then rules.

All necessary data are contained in the trace; do **NOT** invent information.

**Goal - Produce an English explanation containing:**
1. **Summary** (1 sentence) - the main conclusion, e.g. "The system recommends a 10 % tip."
2. **Plain-language reasoning** (<= 150 words) - why the algorithm reached this value without using technical fuzzy system jargon and metrics, like activation values.

**Style Constraints**
- Clear, educational English for a non-expert user; avoid unnecessary mathematical jargon.
- Round all numeric values to two decimal places.
- Use bullet points where they improve readability.
- Total length <= 300 words.
"""


RULEBASE_EXPLANATION_INSTRUCTIONS = """**Role**
You are **Fuzzy Explainer**, an assistant whose task is to convert an execution trace from a fuzzy-logic inference system into a clear explanation for a non-specialist reader.

**Input Provided**
The trace always follows this structure:

<=== MODEL RULEBASE ===>
1. rule-description
2. rule-description
...

<=== INPUT ===>
[variable, value] ; Definition Domain : [min unit(optional) ; max unit(optional)]
...
<=== OUTPUT ===>
(variable_out, defuzzified_value) ; Definition Domain : [min unit(optional) ; max unit(optional)]
<=== EXPLANATION ===>
(linguistic-label 1, Activation Value: X) ; Definition Domain : [min unit(optional) ; max unit(optional)]
(linguistic-label 2, Activation Value: Y) ; Definition Domain : [min unit(optional) ; max unit(optional)]
...

The explanation section lists activated fuzzy labels for the input variables. Use the model rulebase section to connect those labels to the system's rule logic.

All necessary data are contained in the trace and model rulebase context; do **NOT** invent information.

**Goal - Produce an English explanation containing:**
1. **Summary** (1 sentence) - the main conclusion, e.g. "The system recommends a 10 % tip."
2. **Plain-language reasoning** (<= 150 words) - why the algorithm reached this value without using technical fuzzy system jargon and metrics, like activation values.

**Style Constraints**
- Clear, educational English for a non-expert user; avoid unnecessary mathematical jargon.
- Round all numeric values to two decimal places.
- Use bullet points where they improve readability.
- Total length <= 300 words.
"""


@dataclass(frozen=True)
class PromptExample:
    example_id: str
    trace: str
    explanation: str


@dataclass(frozen=True)
class ExperimentMode:
    name: str
    include_examples: bool
    include_rulebase: bool
    instruction_variant: str
    trace_variant: str
    description: str


EXPERIMENT_MODES = {
    "zero-shot+trace": ExperimentMode(
        name="zero-shot+trace",
        include_examples=False,
        include_rulebase=False,
        instruction_variant="trace_only",
        trace_variant="activated_labels",
        description=(
            "Only a system prompt is used, including the simplified trace "
            "produced by the reduction method."
        ),
    ),
    "3-shot+trace": ExperimentMode(
        name="3-shot+trace",
        include_examples=True,
        include_rulebase=False,
        instruction_variant="trace_only",
        trace_variant="activated_labels",
        description=(
            "Given the generated explanations, we add 3-shot prompting; that "
            "is, we provide examples to the LLM to constrain its responses. "
            "This consists of playing the role of a user and the role of the "
            "LLM to indicate which interactions should take place."
        ),
    ),
    "3-shot+(trace with model rulebase)": ExperimentMode(
        name="3-shot+(trace with model rulebase)",
        include_examples=True,
        include_rulebase=True,
        instruction_variant="trace_with_rulebase",
        trace_variant="activated_labels",
        description=(
            "This time, we give the rules and their activations directly to the "
            "LLM, together with the activations of each fuzzy proposition. "
            "Therefore, the reduction function is no longer used. The presence "
            "of the rules provides the LLM with a logical structure, namely the "
            "conjunctions and disjunctions present in the premises of the rules."
        ),
    ),
}


DEFAULT_3SHOT_EXAMPLES = [
    PromptExample(
        example_id="heitor2_food0_service0",
        trace="""<=== INPUT ===>
[food, 0] ; Definition Domain : [0 ; 10]
[service, 0] ; Definition Domain : [0 ; 10]

<=== OUTPUT ===>
(tip, 5) ; Definition Domain : [0 % ; 30 %]

<=== EXPLANATION ===>
(food is rancid, Activation Value: 1) ; Definition Domain : [0 ; 1]
(service is poor, Activation Value: 1) ; Definition Domain : [0 ; 1]""",
        explanation="""**Summary** The system recommends a 5.00 % tip.

**Plain-language reasoning**
- The trace activates "food is rancid" and "service is poor" at 1.00.
- These labels describe a clearly negative dining experience.
- That leads the system to a low tip recommendation of 5.00 %.""",
    ),
    PromptExample(
        example_id="heitor2_food5_service5",
        trace="""<=== INPUT ===>
[food, 5] ; Definition Domain : [0 ; 10]
[service, 5] ; Definition Domain : [0 ; 10]

<=== OUTPUT ===>
(tip, 15) ; Definition Domain : [0 % ; 30 %]

<=== EXPLANATION ===>
(food is okay, Activation Value: 1) ; Definition Domain : [0 ; 1]
(service is good, Activation Value: 1) ; Definition Domain : [0 ; 1]""",
        explanation="""**Summary** The system recommends a 15.00 % tip.

**Plain-language reasoning**
- The trace activates "food is okay" at 1.00 and "service is good" at 1.00.
- These labels describe a solid but not exceptional experience.
- That leads the system to an average tip recommendation of 15.00 %.""",
    ),
    PromptExample(
        example_id="heitor2_food10_service10",
        trace="""<=== INPUT ===>
[food, 10] ; Definition Domain : [0 ; 10]
[service, 10] ; Definition Domain : [0 ; 10]

<=== OUTPUT ===>
(tip, 25) ; Definition Domain : [0 % ; 30 %]

<=== EXPLANATION ===>
(food is delicious, Activation Value: 1) ; Definition Domain : [0 ; 1]
(service is excellent, Activation Value: 1) ; Definition Domain : [0 ; 1]""",
        explanation="""**Summary** The system recommends a 25.00 % tip.

**Plain-language reasoning**
- The trace activates "food is delicious" at 1.00 and "service is excellent" at 1.00.
- These labels describe the strongest positive case in the system.
- That leads the system to a generous tip recommendation of 25.00 %.""",
    ),
]


def build_messages(
    instructions: str,
    trace_input: str,
    examples: list[PromptExample] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": instructions.strip()}]

    for example in examples or []:
        messages.append(
            {
                "role": "user",
                "content": f"Example trace:\n\n{example.trace.strip()}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": example.explanation.strip(),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": f"Trace to explain:\n\n{trace_input.strip()}",
        }
    )
    return messages


def default_instructions_for_mode(mode: ExperimentMode) -> str:
    if mode.instruction_variant == "trace_only":
        return TRACE_ONLY_EXPLANATION_INSTRUCTIONS
    if mode.instruction_variant == "trace_with_rulebase":
        return RULEBASE_EXPLANATION_INSTRUCTIONS
    raise ValueError(f"Unknown instruction variant: {mode.instruction_variant}")


def examples_metadata(
    examples: list[PromptExample],
    source: str = "builtin",
) -> dict[str, object]:
    combined = "\n\n".join(
        f"{example.example_id}\n{example.trace}\n{example.explanation}"
        for example in examples
    )
    return {
        "source": source,
        "count": len(examples),
        "ids": [example.example_id for example in examples],
        "sha256": sha256_text(combined),
    }


def mode_descriptions_text() -> str:
    lines = ["Modes:"]
    for mode in EXPERIMENT_MODES.values():
        lines.append(f"- {mode.name}:")
        lines.append(indent(mode.description, "  "))
    return "\n".join(lines)
