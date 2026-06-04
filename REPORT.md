# Research Implementation Report

Generated on `2026-06-05T00:55:00` from local experiment logs under `./logs`.

This document is intentionally verbose. It is meant as a source text for a later research report, not as a polished final paper. The numbers in the tables are computed from the logs available in this workspace at generation time. The companion notebook [report.ipynb](report.ipynb) contains reproducible code for rebuilding the data tables and PDF plots.


## Executive Summary

This project implements an experiment framework for generating natural-language explanations from fuzzy-logic execution traces for the classic restaurant tipping problem. The central object of study is whether a large language model can convert a reduced fuzzy execution trace into a concise, faithful explanation for a non-specialist reader.

The framework supports multiple prompt modes, multiple trace-model/rulebase variants, multiple LLM backends, reproducible JSON logging, LLM-as-judge evaluation, and a Streamlit visualization interface. The current local corpus contains `785` experiment log files. Of these, `644` are successful experiment runs and `141` are non-success logs. Successful runs with non-empty explanation text total `643`. The logs span `2026-04-08 17:26:49.199215+00:00` to `2026-04-30 13:00:25.489201+00:00`.

The primary automatic evaluation metric is Trace Coverage Faithfulness (TCF). TCF measures whether each atomic line of the reduced trace is represented in the generated explanation. For each trace fact, the evaluator asks a yes/no question and scores the explanation as `yes_count / total_questions`. In the current logs, `643` logs have a latest successful TCF score. The mean latest TCF is `0.971` and the median latest TCF is `1.000`. `552` scored logs have perfect TCF (`1.0`), and `91` scored logs have TCF below `1.0`.


## Research Goal and Problem Framing

The research goal is to study natural-language explanation generation for fuzzy-logic inference traces. The fuzzy system is the classic tip problem, where food quality and service quality are inputs, and the model outputs a recommended tip percentage. The fuzzy inference process produces intermediate linguistic activations such as `food is rancid`, `food is delicious`, `service is poor`, or `service is excellent`. The explanation task is to transform this execution trace into readable English that preserves the relevant input, output, and activated fuzzy-label information.

The project focuses on a deliberately constrained explanation setting. The LLM is not supposed to infer hidden context, introduce real-world assumptions, or rationalize beyond the trace. It should explain what the fuzzy model did, not what a person might independently believe about tipping. This makes the task useful for studying faithfulness: the trace provides an explicit ground truth, and the explanation can be evaluated for coverage of that trace.

A second goal is methodological reproducibility. Each experiment records the prompt, trace metadata, model parameters, backend information, output text, runtime metadata, and evaluation results. This makes it possible to compare prompt modes, models, rulebases, and trace cases after the fact.


## Implemented Experiment Framework

The framework is implemented under `xai_experiments/` and can be run through `python3 -m xai_experiments` or the top-level wrapper `run_experiment.py`. It treats each explanation generation as an experiment with a structured configuration and a structured JSON log.

Important implementation components:

- `xai_experiments/cli.py`: command-line orchestration for loading traces, building prompts, calling the backend, running TCF evaluation, and writing logs.
- `xai_experiments/backends.py`: backend abstraction for Groq and OpenAI-compatible chat-completions APIs. The default backend is Groq, but the same request path supports OpenAI-compatible APIs with different base URLs and API-key variables.
- `xai_experiments/trace_loader.py`: trace loading, reduction, formatting, metadata parsing, and rulebase extraction from raw fuzzy trace JSON.
- `xai_experiments/prompts.py`: prompt instructions, examples, and experiment-mode definitions.
- `xai_experiments/evaluation.py`: TCF implementation and evaluation-history helpers.
- `xai_experiments/evaluate.py`: backfill/re-evaluation CLI for older logs.
- `streamlit_app/`: Streamlit interface for browsing experiments, explanations, prompts, traces, rulebases, plots, and metric history.

The framework also includes batch scripts:

- `run_all_experiments.sh`: runs all food/service combinations from `0..10` for all three modes.
- `run_representative_experiments.sh`: runs a fixed representative subset of ten food/service pairs for all three modes.
- `evaluate_faithfulness.py`: top-level wrapper for TCF backfill.
- `test_groq_api.py`: direct Groq connectivity and payload replay test.

A `--smoke-run` mode was added for checking API keys and pipeline connectivity without writing logs. A `--dry-run` mode prints assembled prompts without calling any backend.


## Prompt Modes

Three prompt modes were implemented. They differ in how much context is given to the LLM and whether examples or rulebase information are included.

1. `zero-shot+trace`: only a system prompt is used, plus the reduced trace. The prompt does not include examples and does not include the full model rulebase. The trace includes input values, output value, and activated fuzzy propositions.

2. `3-shot+trace`: the same reduced trace representation is used, but the prompt also includes three example user/assistant interactions. The examples constrain the expected output format and writing style.

3. `3-shot+(trace with model rulebase)`: the prompt includes three examples and additionally includes the model rulebase. This mode gives the LLM a direct logical structure, including conjunctions and disjunctions in the premises of fuzzy rules. It is the only mode where rulebase information is included in the LLM input.

A key design correction was made during development: modes 1 and 2 must not include rulebase information, either in the instructions or in the trace. For these modes, the trace only exposes active fuzzy proposition labels. Mode 3 remains the rulebase-aware condition.


## Trace Representation

Raw traces are stored under the convention:

```text
traces/<model_name>/tip_trace_food<foodScore>_service<serviceScore>.json
```

The default trace is `traces/heitor2/tip_trace_food2_service7.json`. The trace formatter converts raw trace nodes into a compact text structure:

```text
<=== INPUT ===>
[food, value] ; Definition Domain : [0 ; 10]
[service, value] ; Definition Domain : [0 ; 10]

<=== OUTPUT ===>
(tip, value) ; Definition Domain : [0 % ; 30 %]

<=== EXPLANATION ===>
(label, Activation Value: value) ; Definition Domain : [0 ; 1]
```

For modes 1 and 2, the `EXPLANATION` section lists activated fuzzy propositions such as `food is rancid` or `service is excellent`. It does not list full `if ... then ...` rules. This is the reduced representation used as the main factual substrate for explanation.

For mode 3, the prompt includes a separate `<=== MODEL RULEBASE ===>` section. The trace still uses the reduced activated-label representation, but the model rulebase is available as additional context.


## Rulebases

The project currently contains rulebase descriptions and plots for `heitor1`, `heitor2`, and `heitor3` under `rulebases/`. These rulebases correspond to different fuzzy model variants. Their plots include food, service, and tip membership functions, surfaces, heatmaps, and selected slices.

Rulebase resources are used in two ways:

- Experiment generation: mode 3 can infer the rulebase from unique `MamdaniRule` trace nodes under the selected trace-model directory, or it can use an explicit `--rulebase-file`.
- Visualization: the Streamlit app loads `rulebases/<name>/DESCRIPTION.md` and plots from `rulebases/<name>/plots` when available.

The logs use `trace_model` as the main operational rulebase identifier because the trace directory determines which fuzzy model produced the trace.


## Backend and Model Handling

The backend layer is intentionally OpenAI-compatible. The default backend is Groq through `https://api.groq.com/openai/v1`, using `GROQ_API_KEY`. OpenAI-compatible calls can also target OpenAI by selecting `--backend openai-compatible`, setting `OPENAI_API_KEY`, and using model IDs such as `gpt-4o`.

Several backend-specific issues were handled during development:

- Groq requests include `Accept` and `User-Agent` headers because earlier missing headers contributed to `403` failures in some contexts.
- OpenAI reasoning models such as `o3` require `max_completion_tokens` rather than `max_tokens` and do not accept non-default `temperature`/`top_p` values in the same way as standard chat models.
- `o3` can spend the entire output budget on hidden reasoning tokens and return no visible text. Logs made this visible through token usage metadata, especially `completion_tokens_details.reasoning_tokens`.
- Batch runners now stop immediately when the CLI detects a missing required API key, rather than continuing through a whole sweep.

The API key value itself is never written to experiment logs. Logs store only the API-key environment variable name.


## Logging and Reproducibility

Every non-dry-run experiment writes a JSON file under `logs/YYYY-MM-DD/`. Logs include:

- CLI parameters and runtime metadata.
- Backend name, base URL, chat-completions URL, API-key environment variable name, and generation parameters.
- Prompt messages, prompt hashes, example metadata, trace hashes, and formatted trace hashes.
- Trace path metadata, input/output summaries, active proposition counts, and rulebase metadata.
- Output text, output size, response metadata, generation time, and token usage when returned by the backend.
- TCF evaluation history, including prompts, raw judge responses, yes/no decisions, and score summaries.
- Error details when generation or evaluation fails.

The logs are designed to support reproducibility and post-hoc analysis. They also support evaluation backfill: older logs can be re-evaluated without rerunning the explanation model.


## Trace Coverage Faithfulness (TCF)

TCF is the main automatic evaluation metric implemented in this project. It treats the reduced trace as ground truth and asks whether each atomic trace fact appears in the generated explanation.

The metric pipeline is:

1. Extract non-empty lines from the reduced trace under `INPUT`, `OUTPUT`, and `EXPLANATION` sections.
2. Build fixed questions for `INPUT` and `OUTPUT` facts.
3. Use an LLM to generate questions for `EXPLANATION` facts.
4. Normalize generated questions so that every counted question is a positive coverage check.
5. Ask an LLM judge each question using the reduced trace and explanation side by side.
6. Compute `TCF = yes_count / total_questions`.

Important refinements were made to reduce metric artifacts:

- Input and output questions are fixed instead of generated.
- Explanation questions must preserve the exact trace label and polarity.
- Questions must be positive coverage checks, meaning a faithful explanation should answer `YES`.
- Questions such as `Is the food quality good?` for a trace fact `food is rancid` are rejected and replaced with deterministic fallback questions.
- Multiple evaluations are append-only. Logs store `evaluations` as history and `evaluation` as a compatibility alias for the newest evaluation.

The current Streamlit app displays the newest TCF score by default and allows selection of older evaluations.


## Streamlit Visualization

The Streamlit app provides a research browser over the log corpus. It includes:

- experiment selection from the log tree;
- filters for mode, status, model, rulebase, date, food score, and service score;
- metric filters for finding good, mixed, bad, missing, skipped, or errored evaluations;
- sorting by newest, highest score, or lowest score;
- top-level food/service/tip summaries;
- readable fuzzy model output;
- collapsible views for the actual LLM input trace, raw trace JSON, prompt messages, response metadata, full rulebase text, plots, and raw log JSON;
- TCF score and full per-question judge details directly below the generated explanation.

The app is designed for inspecting many experiments without losing access to the full reproducibility payload.


## Current Log Corpus Overview

| Item | Value |
| --- | --- |
| Total JSON logs parsed | 785 |
| Successful experiment logs | 644 |
| Non-success logs | 141 |
| Successful logs with non-empty output text | 643 |
| Logs with latest successful TCF score | 643 |
| Mean latest TCF | 0.971 |
| Median latest TCF | 1.000 |
| Perfect latest TCF logs | 552 |
| Non-perfect latest TCF logs | 91 |
| Earliest log timestamp | 2026-04-08 17:26:49.199215+00:00 |
| Latest log timestamp | 2026-04-30 13:00:25.489201+00:00 |
| Unique generation models | 4 |
| Unique modes | 3 |
| Unique trace models / rulebases | 3 |
| Unique logged food/service cases | 121 |


## Status Summary

| status | logs |
| --- | --- |
| success | 644 |
| error | 141 |


## TCF Status Summary

| tcf_status | logs |
| --- | --- |
| success | 643 |
| skipped | 118 |
| missing | 24 |


## TCF Quality Buckets

| quality_bucket | logs |
| --- | --- |
| good >=0.80 | 634 |
| unscored | 142 |
| mixed 0.40-0.79 | 9 |


## Model Summary

| model | logs | success | success_rate | scored | mean_tcf | median_tcf | mean_time_s | mean_words | mean_total_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama-3.1-8b-instant | 603 | 488 | 0.809 | 488 | 0.963 | 1.000 | 0.415 | 94.436 | 1014.191 |
| o3 | 88 | 62 | 0.705 | 61 | 1.000 | 1.000 | 5.502 | 134.984 | 1259.339 |
| gpt-4o | 64 | 64 | 1.000 | 64 | 0.991 | 1.000 | 2.432 | 108.359 | 1007.797 |
| gpt-4.1-nano | 30 | 30 | 1.000 | 30 | 1.000 | 1.000 | 1.605 | 104.400 | 989.300 |


## Mode Summary

| mode | logs | success | success_rate | scored | mean_tcf | median_tcf | mean_time_s | mean_words | mean_total_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero-shot+trace | 265 | 217 | 0.819 | 217 | 0.976 | 1.000 | 1.303 | 136.180 | 653.083 |
| 3-shot+(trace with model rulebase) | 263 | 216 | 0.821 | 215 | 0.964 | 1.000 | 1.226 | 95.255 | 1311.208 |
| 3-shot+trace | 257 | 211 | 0.821 | 211 | 0.972 | 1.000 | 0.948 | 68.223 | 1148.066 |


## Rulebase / Trace-Model Summary

| trace_model | logs | success | success_rate | scored | mean_tcf | median_tcf | mean_time_s | mean_words | mean_total_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heitor2 | 699 | 582 | 0.833 | 581 | 0.969 | 1.000 | 1.242 | 101.612 | 1040.048 |
| heitor1 | 54 | 30 | 0.556 | 30 | 1.000 | 1.000 | 0.388 | 81.200 | 995.067 |
| heitor3 | 32 | 32 | 1.000 | 32 | 0.966 | 1.000 | 0.398 | 92.094 | 1000.688 |


## Backend Summary

| backend | logs | success | success_rate | scored | mean_tcf | median_tcf | mean_time_s | mean_words | mean_total_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groq | 603 | 488 | 0.809 | 488 | 0.963 | 1.000 | 0.415 | 94.436 | 1014.191 |
| openai-compatible | 182 | 156 | 0.857 | 155 | 0.996 | 1.000 | 3.493 | 118.179 | 1104.212 |


## Mode by Model Summary

| model | mode | logs | success | scored | mean_tcf |
| --- | --- | --- | --- | --- | --- |
| gpt-4.1-nano | 3-shot+(trace with model rulebase) | 10 | 10 | 10 | 1.000 |
| gpt-4.1-nano | 3-shot+trace | 10 | 10 | 10 | 1.000 |
| gpt-4.1-nano | zero-shot+trace | 10 | 10 | 10 | 1.000 |
| gpt-4o | 3-shot+(trace with model rulebase) | 23 | 23 | 23 | 0.974 |
| gpt-4o | 3-shot+trace | 20 | 20 | 20 | 1.000 |
| gpt-4o | zero-shot+trace | 21 | 21 | 21 | 1.000 |
| llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | 199 | 161 | 161 | 0.956 |
| llama-3.1-8b-instant | 3-shot+trace | 199 | 161 | 161 | 0.964 |
| llama-3.1-8b-instant | zero-shot+trace | 205 | 166 | 166 | 0.968 |
| o3 | 3-shot+(trace with model rulebase) | 31 | 22 | 21 | 1.000 |
| o3 | 3-shot+trace | 28 | 20 | 20 | 1.000 |
| o3 | zero-shot+trace | 29 | 20 | 20 | 1.000 |


## Error Summary

| error_type | error_message | logs |
| --- | --- | --- |
| BackendError | groq requires an API key in GROQ_API_KEY. | 114 |
| BackendError | groq returned HTTP 403: error code: 1010 | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdebf6fa94aff-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdec269729ef0-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdec52ac79ea5-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdec8da411310-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdecbbf286981-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdecfaed3d128-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fded308a8f0ec-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fded67c997e80-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fded9b8a1709b-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdedcdf9d6ffa-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdedf6ad90288-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdee928ecd377-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |
| BackendError | openai-compatible returned HTTP 400 headers={"cf-ray": "9e9fdeec5a9dbb33-CDG", "server": "cloudflare", "content-type": "application/json", "content-length": "245"}: {   "error": {     "message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",     "type": "invalid_request_error",     "param": "max_tokens",     "code": "unsupported_parameter"   } } | 1 |


## TCF NO Judgments by Section, Model, and Mode

| section | model | mode | no_judgments |
| --- | --- | --- | --- |
| INPUT | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | 29 |
| INPUT | llama-3.1-8b-instant | 3-shot+trace | 27 |
| INPUT | llama-3.1-8b-instant | zero-shot+trace | 24 |
| EXPLANATION | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | 9 |
| EXPLANATION | llama-3.1-8b-instant | 3-shot+trace | 4 |
| EXPLANATION | llama-3.1-8b-instant | zero-shot+trace | 4 |
| INPUT | gpt-4o | 3-shot+(trace with model rulebase) | 3 |


## Representative Case Coverage

| representative_case | status | logs |
| --- | --- | --- |
| False | success | 333 |
| True | error | 141 |
| True | success | 311 |


## Example Non-Perfect Latest TCF Logs

| path | model | mode | trace_model | food | service | tcf_score | tcf_yes_count | tcf_total_questions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123849677482Z_zero-shot-trace_e6633154.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 1 | 0 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123850275720Z_3-shot-trace_b90f1dc9.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 1 | 0 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123850726372Z_3-shot-trace-with-model-rulebase_880d4206.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 1 | 0 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123852518702Z_zero-shot-trace_f4e84e31.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 1 | 2 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123853037907Z_3-shot-trace_830130ce.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 1 | 2 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123853463894Z_3-shot-trace-with-model-rulebase_2d37bb98.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 1 | 2 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123910558781Z_3-shot-trace_8a40ad65.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 2 | 2 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123910972647Z_3-shot-trace-with-model-rulebase_b844a194.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 2 | 2 | 0.600 | 3.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123929541713Z_3-shot-trace_1eebb3e6.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 3 | 3 | 0.667 | 4.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123936180012Z_zero-shot-trace_bd817ced.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 3 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123936677516Z_3-shot-trace_777404f1.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 3 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123937086036Z_3-shot-trace-with-model-rulebase_2ca2358b.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 3 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123937591499Z_zero-shot-trace_18b59a11.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 3 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123938102786Z_3-shot-trace_b0ef756d.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 3 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123939738314Z_3-shot-trace-with-model-rulebase_72d180eb.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 3 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123940260156Z_zero-shot-trace_ca510004.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 3 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123940717734Z_3-shot-trace_104a4425.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 3 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123941075242Z_3-shot-trace-with-model-rulebase_77d667bf.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 3 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123953568062Z_zero-shot-trace_731b7b0d.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 4 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123954179501Z_3-shot-trace_d6ec8ecf.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 4 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123954599106Z_3-shot-trace-with-model-rulebase_f26ea52b.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 4 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123954998154Z_zero-shot-trace_e5ac95fe.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 4 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123955514831Z_3-shot-trace_0aed939c.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 4 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123955929491Z_3-shot-trace-with-model-rulebase_77cf9321.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 4 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123956329215Z_zero-shot-trace_5ed9062c.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 4 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123956843584Z_3-shot-trace_ae32b67b.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 4 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123957235087Z_3-shot-trace-with-model-rulebase_414bd1cc.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 4 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124010769491Z_zero-shot-trace_2c97edc6.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 5 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124011390440Z_3-shot-trace_e623a619.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 5 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124011813287Z_3-shot-trace-with-model-rulebase_a94414c4.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 5 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124012210129Z_zero-shot-trace_bc4cd250.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 5 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124012721063Z_3-shot-trace_0163e8e9.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 5 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124013051663Z_3-shot-trace-with-model-rulebase_64148fba.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 5 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124026223306Z_zero-shot-trace_e22acd53.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 6 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124026687341Z_3-shot-trace_97949989.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 6 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124027162496Z_3-shot-trace-with-model-rulebase_36ff0abb.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 6 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124027565260Z_zero-shot-trace_bb574dba.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 6 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124028006038Z_3-shot-trace_90334071.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 6 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124028391629Z_3-shot-trace-with-model-rulebase_a4e5b07e.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 6 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124028795475Z_zero-shot-trace_1d80f8d3.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 6 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124029304641Z_3-shot-trace_9424406b.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 6 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124029691479Z_3-shot-trace-with-model-rulebase_80deb5b6.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 6 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124043026722Z_zero-shot-trace_e65fb5de.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 7 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124043494919Z_3-shot-trace_f1cc48ee.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 7 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124043955219Z_3-shot-trace-with-model-rulebase_df5f25dc.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 7 | 8 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124044359746Z_zero-shot-trace_f1385e7c.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 7 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124044873011Z_3-shot-trace_5bd1796f.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 7 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124045288688Z_3-shot-trace-with-model-rulebase_a4b2b4de.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 7 | 9 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124045695905Z_zero-shot-trace_ba1fed4d.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 7 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124046199898Z_3-shot-trace_3bf6b05e.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 7 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124046616736Z_3-shot-trace-with-model-rulebase_e0127263.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 7 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-10/2026-04-10T065814487148Z_3-shot-trace-with-model-rulebase_4288747e.json | gpt-4o | 3-shot+(trace with model rulebase) | heitor2 | 2 | 2 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-10/2026-04-10T071346348381Z_3-shot-trace-with-model-rulebase_0fb1a87f.json | gpt-4o | 3-shot+(trace with model rulebase) | heitor2 | 2 | 2 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-10/2026-04-10T072602746623Z_3-shot-trace-with-model-rulebase_3ddc0d17.json | gpt-4o | 3-shot+(trace with model rulebase) | heitor2 | 2 | 2 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T125837775606Z_zero-shot-trace_17e73cc7.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 2 | 2 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T125841214486Z_3-shot-trace-with-model-rulebase_1bf863b3.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 2 | 2 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T125956954489Z_3-shot-trace-with-model-rulebase_7efcdaf9.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor3 | 0 | 10 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T130007057803Z_3-shot-trace_ee60416f.json | llama-3.1-8b-instant | 3-shot+trace | heitor3 | 2 | 2 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T130008706931Z_3-shot-trace-with-model-rulebase_9664bb62.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor3 | 2 | 2 | 0.800 | 4.000 | 5.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123837792278Z_3-shot-trace_f5e403bf.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 0 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123838297751Z_3-shot-trace-with-model-rulebase_54d967ad.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 0 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123844661669Z_3-shot-trace-with-model-rulebase_b01aab7e.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 0 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123854898132Z_3-shot-trace-with-model-rulebase_4900924d.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 1 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123900837239Z_3-shot-trace-with-model-rulebase_f635a2c2.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 1 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123912534315Z_3-shot-trace-with-model-rulebase_610d952b.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 2 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123930023186Z_3-shot-trace-with-model-rulebase_9cf326bb.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 3 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123934617711Z_zero-shot-trace_e1555d41.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 3 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123935256877Z_3-shot-trace_7e4d918a.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 3 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123935651698Z_3-shot-trace-with-model-rulebase_acde30ba.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 3 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123946031907Z_zero-shot-trace_b758d6a8.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 4 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123947330351Z_3-shot-trace-with-model-rulebase_15838bf4.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 4 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123952033233Z_zero-shot-trace_baa04081.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 4 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123952649478Z_3-shot-trace_b1b8a5c0.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 4 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T123953063223Z_3-shot-trace-with-model-rulebase_f3a57406.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 4 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124001964863Z_zero-shot-trace_25de2492.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 5 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124002467035Z_3-shot-trace_da6546d6.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 5 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124002986743Z_3-shot-trace-with-model-rulebase_6970353e.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 5 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124007687249Z_zero-shot-trace_d72bf0e0.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 5 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124008768974Z_3-shot-trace-with-model-rulebase_5f67285c.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 5 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124019185253Z_3-shot-trace-with-model-rulebase_2aff6433.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 6 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124024485304Z_zero-shot-trace_80ff2d4e.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 6 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124025124063Z_3-shot-trace_324b163c.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 6 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124035036257Z_zero-shot-trace_fcc30fc6.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 7 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124035858251Z_3-shot-trace_81e8e5cd.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 7 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124036375989Z_3-shot-trace-with-model-rulebase_1520a8ae.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 7 | 3 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124041390894Z_zero-shot-trace_d54aae11.json | llama-3.1-8b-instant | zero-shot+trace | heitor2 | 7 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124042003419Z_3-shot-trace_e6ff2bb0.json | llama-3.1-8b-instant | 3-shot+trace | heitor2 | 7 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-09/2026-04-09T124042509531Z_3-shot-trace-with-model-rulebase_9628e6ce.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor2 | 7 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T125949521986Z_3-shot-trace_6974daca.json | llama-3.1-8b-instant | 3-shot+trace | heitor3 | 2 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T125951276325Z_3-shot-trace-with-model-rulebase_f44b28f8.json | llama-3.1-8b-instant | 3-shot+(trace with model rulebase) | heitor3 | 2 | 7 | 0.833 | 5.000 | 6.000 |
| /home/baunilha/Repositories/xai-playground/logs/2026-04-30/2026-04-30T130016113210Z_zero-shot-trace_e2fe118c.json | llama-3.1-8b-instant | zero-shot+trace | heitor3 | 5 | 8 | 0.833 | 5.000 | 6.000 |


## Generated Plots

The following PDF plots were generated under `./plots`:

- [experiment_status_by_date.pdf](plots/experiment_status_by_date.pdf)
- [experiments_by_model.pdf](plots/experiments_by_model.pdf)
- [success_rate_by_model.pdf](plots/success_rate_by_model.pdf)
- [successful_explanations_by_model_mode.pdf](plots/successful_explanations_by_model_mode.pdf)
- [successful_explanations_by_rulebase_mode.pdf](plots/successful_explanations_by_rulebase_mode.pdf)
- [tcf_score_distribution.pdf](plots/tcf_score_distribution.pdf)
- [tcf_by_model_boxplot.pdf](plots/tcf_by_model_boxplot.pdf)
- [tcf_by_mode_boxplot.pdf](plots/tcf_by_mode_boxplot.pdf)
- [tcf_by_rulebase_boxplot.pdf](plots/tcf_by_rulebase_boxplot.pdf)
- [mean_tcf_heatmap_model_mode.pdf](plots/mean_tcf_heatmap_model_mode.pdf)
- [mean_tcf_heatmap_food_service.pdf](plots/mean_tcf_heatmap_food_service.pdf)
- [mean_tip_heatmap_food_service.pdf](plots/mean_tip_heatmap_food_service.pdf)
- [mean_generation_time_by_model.pdf](plots/mean_generation_time_by_model.pdf)
- [mean_output_words_by_model.pdf](plots/mean_output_words_by_model.pdf)
- [mean_token_usage_by_model.pdf](plots/mean_token_usage_by_model.pdf)
- [tcf_vs_output_words.pdf](plots/tcf_vs_output_words.pdf)
- [error_types.pdf](plots/error_types.pdf)
- [tcf_no_judgments_by_section.pdf](plots/tcf_no_judgments_by_section.pdf)
- [mean_tcf_evaluation_count_by_model.pdf](plots/mean_tcf_evaluation_count_by_model.pdf)
- [case_coverage_by_rulebase.pdf](plots/case_coverage_by_rulebase.pdf)
- [tcf_question_normalization.pdf](plots/tcf_question_normalization.pdf)
- [mean_tcf_by_judge_model.pdf](plots/mean_tcf_by_judge_model.pdf)


## Suggested Interpretation Angles for the Final Report

The final research report can use these artifacts to discuss several distinct questions:

1. Prompt design: compare zero-shot reduced-trace prompting against 3-shot prompting and rulebase-aware prompting.
2. Model behavior: compare smaller, faster models against stronger models on TCF, output length, token usage, and error rates.
3. Rulebase sensitivity: compare whether `heitor1`, `heitor2`, and `heitor3` traces are equally easy to explain.
4. Input-space sensitivity: inspect whether boundary cases such as `(0,0)`, `(10,10)`, `(0,10)`, and `(10,0)` are explained more reliably than mixed cases.
5. Metric reliability: discuss TCF refinements, especially the need for positive coverage questions and the distinction between missing trace coverage and stylistic differences.
6. Reproducibility: emphasize that each experiment is logged with prompts, parameters, trace hashes, model IDs, outputs, evaluation prompts, judge responses, and runtime metadata.

The most important caveat is that TCF is an LLM-as-judge metric, so it measures judged trace coverage, not human explanation quality in full. It is useful for detecting missing trace facts, but it does not directly measure fluency, helpfulness, concision, causal correctness beyond the trace, or user preference. It should be reported as a faithfulness/coverage metric, not as a complete explanation-quality metric.


## Threats to Validity and Limitations

Several limitations should be stated clearly in the final report:

- The local log corpus may include experiments generated at different stages of prompt and evaluator development. Comparisons should use the newest TCF evaluations when possible, and older evaluations should be interpreted carefully.
- TCF depends on an LLM judge. Even with stricter prompts and deterministic question fallbacks, judge errors remain possible.
- TCF rewards coverage of reduced trace facts. It does not penalize all forms of verbosity, awkward wording, or unsupported elaboration unless they contradict or obscure the trace facts.
- Mode 3 gives the LLM additional rulebase structure, so it is not directly comparable to modes 1 and 2 as a pure prompt-format change. It changes the available information.
- Models may differ in token accounting and response metadata, especially across Groq and OpenAI-compatible backends.
- Some failed logs and smoke/error logs are useful for engineering diagnosis but should be excluded from performance claims about explanation quality.
- The rulebase variants are not necessarily equally complex; rule count, rule operators, and label definitions can affect explanation difficulty.


## Reproducibility Commands

Useful commands for reproducing or extending the study:

```bash
python3 -m xai_experiments --mode zero-shot+trace
python3 -m xai_experiments --mode all
./run_representative_experiments.sh --trace-model heitor2
./run_all_experiments.sh --trace-model heitor3 --logs-dir logs/heitor3
python3 -m xai_experiments.evaluate --representative-only --force
python3 -m xai_experiments --mode zero-shot+trace --smoke-run --skip-faithfulness-eval
./xai/bin/streamlit run streamlit_app/app.py
```

The notebook `report.ipynb` can be rerun after new logs are generated to refresh the tables and plots.
