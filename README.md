# xai-playground

Explainable AI Playground.

## Fuzzy Trace Explanation Experiments

This repository includes a small, extensible experiment runner for generating natural-language explanations from ExpressIF fuzzy-logic execution traces for the classic tip problem.

Default behavior:

- Backend: Groq through its OpenAI-compatible chat-completions API.
- API key: `GROQ_API_KEY` in the environment.
- Model: requested `Llama3.1-8b`, mapped to the Groq model id `llama-3.1-8b-instant`.
- Trace: `./traces/heitor2/tip_trace_food2_service7.json`.
- Logs: local JSON files under `./logs`.

References:

- Groq OpenAI-compatible API: https://console.groq.com/docs/overview
- Groq supported models: https://console.groq.com/docs/models
- Groq model deprecations: https://console.groq.com/docs/deprecations

## Run

Set the Groq key, then run one mode:

```bash
export GROQ_API_KEY="..."
python3 -m xai_experiments --mode zero-shot+trace
```

Run all initial modes:

```bash
python3 -m xai_experiments --mode all
```

Run one specific case with an explicit model, trace-model rulebase, and food /
service scores:

```bash
python3 -m xai_experiments \
  --backend openai-compatible \
  --model gpt-4o \
  --mode "3-shot+(trace with model rulebase)" \
  --trace-model heitor2 \
  --food 2 \
  --service 7
```

In the common case, `--trace-model heitor2` selects both:

- the trace file under `./traces/heitor2/`
- the inferred rulebase for mode 3 from that same trace-model directory

If you want to override the inferred rulebase text for mode 3, add
`--rulebase-file`:

```bash
python3 -m xai_experiments \
  --backend openai-compatible \
  --model gpt-4o \
  --mode "3-shot+(trace with model rulebase)" \
  --trace-model heitor2 \
  --food 2 \
  --service 7 \
  --rulebase-file path/to/custom_rulebase.txt
```

Run every food/service combination from `0..10` for all three modes on the
default model:

```bash
./run_all_experiments.sh
```

The script keeps the runner defaults for the LLM model and trace model unless
you override them through forwarded CLI arguments, for example:

```bash
./run_all_experiments.sh --trace-model heitor3 --logs-dir logs/heitor3
```

It forwards extra arguments to each run, while always sweeping all three modes
and all `food`/`service` pairs. To run only part of the grid, use environment
variables:

```bash
FOOD_MIN=0 FOOD_MAX=2 SERVICE_MIN=0 SERVICE_MAX=2 ./run_all_experiments.sh --dry-run
```

If a run fails because a required API key is missing, the batch stops
immediately instead of continuing through the rest of the grid.

Run a fixed 10-case representative subset for each mode:

```bash
./run_representative_experiments.sh
```

This script is intended for quicker comparison runs when the full grid is too
large. It executes the same 10 preset food/service pairs for all three modes,
for a total of 30 runs:

- `(food=0, service=0)`
- `(food=5, service=5)`
- `(food=10, service=10)`
- `(food=2, service=7)`
- `(food=0, service=10)`
- `(food=10, service=0)`
- `(food=2, service=2)`
- `(food=8, service=5)`
- `(food=5, service=8)`
- `(food=9, service=9)`

Like the full-grid runner, it keeps the experiment defaults unless you forward
extra CLI flags:

```bash
./run_representative_experiments.sh --backend openai-compatible --model gpt-4o
```

This representative runner also stops immediately if a required API key is
missing.

The same CLI is available through the top-level wrapper:

```bash
python3 run_experiment.py --mode "3-shot+(trace with model rulebase)"
```

Use `--dry-run` to print the assembled messages without calling an LLM or writing a log:

```bash
python3 -m xai_experiments --mode all --dry-run
```

Use `--smoke-run` to call the backend and run the pipeline in memory without
writing any experiment log:

```bash
python3 -m xai_experiments --mode zero-shot+trace --smoke-run
```

This is the simplest way to verify that API keys, backend connectivity, trace
loading, prompt assembly, and optional evaluation are working, without creating
JSON log files under `./logs`. To test only the main generation call and skip
the evaluation stage, add:

```bash
python3 -m xai_experiments --mode zero-shot+trace --smoke-run --skip-faithfulness-eval
```

## Trace Coverage Faithfulness (TCF)

Each successful experiment now runs an LLM-as-judge Trace Coverage Faithfulness
(TCF) evaluation by default and writes the result into the experiment log.
Evaluations are append-only:

- `evaluations`: full history, newest first
- `evaluation`: compatibility alias pointing to the newest evaluation

The evaluation uses the reduced trace as ground truth:

1. one LLM call generates one yes/no question for each non-empty reduced-trace line
2. another set of LLM calls judges each question against the reduced trace and the generated explanation
3. the TCF score is computed as `yes_count / total_questions`

For the reduced trace below, the evaluation would produce 5 questions: 2 input
facts, 1 output fact, and 2 explanation facts.

```text
<=== INPUT ===>
[food, 5] ; Definition Domain : [0 ; 10]
[service, 8] ; Definition Domain : [0 ; 10]

<=== OUTPUT ===>
(tip, 25) ; Definition Domain : [0 % ; 30 %]

<=== EXPLANATION ===>
(service is excellent, Activation Value: 1) ; Definition Domain : [0 ; 1]
(food is okay, Activation Value: 1) ; Definition Domain : [0 ; 1]
```

Skip this stage when needed:

```bash
python3 -m xai_experiments --mode zero-shot+trace --skip-faithfulness-eval
```

Override the judge backend or model independently from the explanation model:

```bash
python3 -m xai_experiments \
  --mode "3-shot+(trace with model rulebase)" \
  --backend groq \
  --model llama-3.1-8b-instant \
  --eval-backend openai-compatible \
  --eval-model gpt-4o
```

Relevant evaluation flags:

- `--skip-faithfulness-eval`
- `--eval-backend`
- `--eval-base-url`
- `--eval-api-key-env`
- `--eval-model`
- `--eval-temperature`
- `--eval-top-p`
- `--eval-question-max-tokens`
- `--eval-answer-max-tokens`
- `--eval-timeout-seconds`

To backfill TCF evaluation into older logs without rerunning the
original experiments:

```bash
python3 -m xai_experiments.evaluate \
  --experiment-log logs/2026-04-10/2026-04-10T071346348381Z_3-shot-trace-with-model-rulebase_0fb1a87f.json \
  --backend openai-compatible \
  --model gpt-4o
```

The top-level wrapper is also available:

```bash
python3 evaluate_faithfulness.py --backend openai-compatible --model gpt-4o
```

Backfill behavior:

- without `--force`: evaluate only logs that do not already contain any TCF evaluation
- with `--force`: append a new TCF evaluation to the log history

To backfill or re-evaluate only the representative subset used by
`./run_representative_experiments.sh`, add:

```bash
python3 -m xai_experiments.evaluate --representative-only
```

For a fresh representative-only re-evaluation pass:

```bash
python3 -m xai_experiments.evaluate --representative-only --force
```

## Experiment Modes

The initial modes are defined in `xai_experiments/prompts.py`:

- `zero-shot+trace`: Only a system prompt is used, including the simplified trace produced by the reduction method.
- `3-shot+trace`: Given the generated explanations, we add 3-shot prompting; that is, we provide examples to the LLM to constrain its responses. This consists of playing the role of a user and the role of the LLM to indicate which interactions should take place.
- `3-shot+(trace with model rulebase)`: This time, we give the rules and their activations directly to the LLM, together with the activations of each fuzzy proposition. Therefore, the reduction function is no longer used. The presence of the rules provides the LLM with a logical structure, namely the conjunctions and disjunctions present in the premises of the rules.

For the rulebase mode, the runner uses `--rulebase-file` if provided. Otherwise it infers the rulebase from unique `MamdaniRule` nodes under the selected trace model directory, for example `./traces/heitor2/`.

In modes 1 and 2, the trace itself exposes activated fuzzy labels under `<=== EXPLANATION ===>`, not `if ... then` rules. In mode 3, the trace still exposes activated fuzzy labels, and the model rulebase is added separately through `<=== MODEL RULEBASE ===>`.

## Trace Loading

Traces are loaded from the nested path convention:

```text
traces/<model_name>/tip_trace_food<foodScore>_service<serviceScore>.json
```

The default expands to:

```text
traces/heitor2/tip_trace_food2_service7.json
```

Select another trace by score:

```bash
python3 -m xai_experiments --trace-model heitor1 --food 0 --service 0
```

Or pass an explicit path:

```bash
python3 -m xai_experiments --trace-path traces/heitor3/tip_trace_food10_service10.json
```

The formatter converts raw JSON nodes into the prompt trace structure:

```text
<=== INPUT ===>
[food, 2] ; Definition Domain : [0 ; 10]
[service, 7] ; Definition Domain : [0 ; 10]

<=== OUTPUT ===>
(tip, 2.5) ; Definition Domain : [0 % ; 30 %]

<=== EXPLANATION ===>
(food is rancid, Activation Value: 0.5) ; Definition Domain : [0 ; 1]
(service is excellent, Activation Value: 0.5) ; Definition Domain : [0 ; 1]
(service is good, Activation Value: 0.33333333333333337) ; Definition Domain : [0 ; 1]
```

## Logging

Every non-dry-run experiment writes one JSON record under:

```text
logs/YYYY-MM-DD/<timestamp>_<mode>_<experiment-id>.json
```

Each record includes:

- CLI parameters and runtime metadata, including git commit and dirty status.
- Backend name, base URL, API-key environment variable name, and generation parameters.
- Full prompt messages, instruction hash, example metadata, trace hash, and formatted trace hash.
- Rulebase source, extraction method, hash, and rules for rulebase mode.
- Output text, response metadata, generation time, output size, prompt size, and token usage when returned by the backend.
- TCF evaluation prompts, per-question yes/no judgments, score, and summary counts when evaluation is enabled.
- Evaluation history under `evaluations`, with `evaluation` kept as the newest-evaluation alias.
- Error details if the backend call fails.

API key values are not written to logs.

## Streamlit Visualization

The repository includes a Streamlit app under `./streamlit_app` for browsing
experiment logs, prompt inputs, explanation outputs, TCF scores,
traces, rulebases, and available rulebase plots.

Install the app dependency:

```bash
pip install -r streamlit_app/requirements.txt
```

Run the app from the repository root:

```bash
streamlit run streamlit_app/app.py
```

Alternative launch from inside the app directory:

```bash
cd streamlit_app
streamlit run app.py
```

The app reads experiment data from `./logs` and, when available, loads
rulebase descriptions and images from `./rulebases/<name>/plots`. The
newest TCF score appears directly under the explanation, with an expander for
the full question-by-question evaluation and a selector for older evaluation
runs when available. The sidebar also includes quality filters so you can
filter and sort experiments by metric score, status, and quick good/mixed/bad
buckets.

## Backend Swapping

The default backend is `groq`, implemented as an OpenAI-compatible chat-completions adapter in `xai_experiments/backends.py`. Use another OpenAI-compatible endpoint with:

```bash
python3 -m xai_experiments \
  --backend openai-compatible \
  --base-url http://localhost:11434/v1 \
  --allow-missing-api-key \
  --model llama3.1:8b
```

For hosted OpenAI-compatible APIs:

```bash
python3 -m xai_experiments \
  --backend openai-compatible \
  --base-url https://api.example.com/v1 \
  --api-key-env EXAMPLE_API_KEY \
  --model some-model-id
```

To add a non-OpenAI-compatible local backend, implement the `ChatBackend` protocol in `xai_experiments/backends.py` and extend `build_backend`.

## Custom Prompts And Examples

Override the explanation instructions:

```bash
python3 -m xai_experiments --instructions-file prompts/tip_explainer.md
```

Note: `--instructions-file` replaces the mode-specific default instructions completely. If you want zero-shot or 3-shot trace-only behavior, do not mention `<=== MODEL RULEBASE ===>` or `if ... then` rule descriptions in that custom file.

Override the built-in 3-shot examples:

```bash
python3 -m xai_experiments --mode "3-shot+trace" --examples-file examples/tip_examples.json
```

Example JSON shape:

```json
{
  "examples": [
    {
      "id": "case_1",
      "trace_path": "traces/heitor2/tip_trace_food0_service0.json",
      "explanation": "**Summary** The system recommends a 5.00 % tip.\n\n**Plain-language reasoning**\n- ..."
    }
  ]
}
```

## Extending Experiment Modes

Add a mode by registering another `ExperimentMode` in `xai_experiments/prompts.py`. The current mode flags cover the initial prompt parts:

- `include_examples`: adds the example chat turns.
- `include_rulebase`: prefixes the trace input with `<=== MODEL RULEBASE ===>`.

If a new mode needs another prompt component or logging field, add the component in `xai_experiments/cli.py` near prompt assembly and include its metadata in the JSON record.
