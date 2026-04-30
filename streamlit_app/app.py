from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import streamlit as st

try:
    import streamlit_app.data as data_module
except ModuleNotFoundError:
    import data as data_module


LOGS_DIR = data_module.LOGS_DIR
ROOT_DIR = data_module.ROOT_DIR
ExperimentRecord = data_module.ExperimentRecord
MetricSummary = data_module.MetricSummary
available_quality_metrics = data_module.available_quality_metrics
extract_rulebase_name = data_module.extract_rulebase_name
metric_label = data_module.metric_label
metric_summary = data_module.metric_summary
tcf_evaluation_label = data_module.tcf_evaluation_label
tcf_evaluations = data_module.tcf_evaluations
tcf_evaluation = getattr(
    data_module,
    "tcf_evaluation",
    getattr(data_module, "faithfulness_evaluation"),
)
tcf_score = getattr(
    data_module,
    "tcf_score",
    getattr(data_module, "faithfulness_score"),
)
format_number = data_module.format_number
format_percent = data_module.format_percent
format_score = data_module.format_score
format_seconds = data_module.format_seconds
format_timestamp = data_module.format_timestamp
llm_trace_input = data_module.llm_trace_input
load_experiments = data_module.load_experiments
load_full_trace = data_module.load_full_trace
load_rulebase_resources = data_module.load_rulebase_resources
parse_trace_nodes = data_module.parse_trace_nodes
prompt_messages = data_module.prompt_messages
select_main_parameters = data_module.select_main_parameters


st.set_page_config(
    page_title="Experiment Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("Experiment Viewer")
    st.caption(f"Repository root: `{ROOT_DIR}`")

    experiments = load_experiments(LOGS_DIR)
    if not experiments:
        st.warning("No experiment logs were found under `./logs`.")
        return

    selected = render_sidebar(experiments)
    if selected is None:
        st.info("No experiments match the current filters.")
        return

    full_trace = load_full_trace(selected)
    trace_details = parse_trace_nodes(full_trace)
    rulebase_resources = load_rulebase_resources(
        rulebase_name=extract_rulebase_name(selected),
        trace_details=trace_details,
        rulebase_text=((selected.data.get("rulebase") or {}).get("text")),
    )

    render_main(selected, full_trace, trace_details, rulebase_resources)


def render_sidebar(experiments: list[ExperimentRecord]) -> ExperimentRecord | None:
    with st.sidebar:
        header_cols = st.columns([0.72, 0.28])
        header_cols[0].header("Experiment Selection")
        if header_cols[1].button("Refresh", key="refresh_sidebar_filters"):
            st.rerun()

        rulebase_names = {
            record.path: (extract_rulebase_name(record) or "Unavailable")
            for record in experiments
        }
        record_dates = {
            record.path: parse_record_date(record)
            for record in experiments
        }
        mode_options = ["All"] + sorted({record.mode for record in experiments})
        status_options = ["All"] + sorted({record.status for record in experiments})
        model_options = ["All"] + sorted({record.model for record in experiments})
        rulebase_options = ["All"] + sorted(set(rulebase_names.values()))
        experiment_dates = [
            parsed_date
            for parsed_date in record_dates.values()
            if parsed_date is not None
        ]
        food_options = ["All"] + sorted(
            {format_score(record.food_score) for record in experiments if record.food_score is not None},
            key=lambda value: float(value),
        )
        service_options = ["All"] + sorted(
            {format_score(record.service_score) for record in experiments if record.service_score is not None},
            key=lambda value: float(value),
        )

        mode_filter = st.selectbox("Mode", options=mode_options, index=0)
        status_filter = st.selectbox("Status", options=status_options, index=0)
        model_filter = st.selectbox("Model", options=model_options, index=0)
        rulebase_filter = st.selectbox("Rulebase", options=rulebase_options, index=0)
        from_date_filter = None
        include_all_history = True
        if experiment_dates:
            min_date = min(experiment_dates)
            max_date = max(experiment_dates)
            date_cols = st.columns([0.72, 0.28])
            include_all_history = date_cols[1].checkbox(
                "All",
                value=True,
            )
            from_date_filter = date_cols[0].date_input(
                "From date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                disabled=include_all_history,
            )
            if include_all_history:
                from_date_filter = None
        score_cols = st.columns(2)
        food_filter = score_cols[0].selectbox("Food", options=food_options, index=0)
        service_filter = score_cols[1].selectbox("Service", options=service_options, index=0)

        st.divider()
        st.subheader("Quality Filters")
        quality_metric_options = available_quality_metrics(experiments)
        selected_metric = st.selectbox(
            "Metric",
            options=quality_metric_options,
            index=0,
            format_func=metric_label,
        )
        metric_summaries = {
            record.path: metric_summary(record, selected_metric)
            for record in experiments
        }
        metric_status_cols = st.columns(2)
        metric_status_filter = metric_status_cols[0].selectbox(
            "Metric status",
            options=["All", "Has score", "Missing", "Skipped", "Error"],
            index=0,
        )

        slider_min, slider_max = metric_score_bounds(metric_summaries.values())
        quality_preset_options = ["All"]
        if slider_min >= 0.0 and slider_max <= 1.0:
            quality_preset_options.extend(["Good", "Mixed", "Bad"])
        quality_preset = metric_status_cols[1].selectbox(
            "Quality",
            options=quality_preset_options,
            index=0,
        )
        score_range = st.slider(
            "Score range",
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=(float(slider_min), float(slider_max)),
            step=0.01,
        )
        sort_by = st.selectbox(
            "Sort by",
            options=["Newest", "Highest score first", "Lowest score first"],
            index=0,
        )

        filtered = [
            record
            for record in experiments
            if (mode_filter == "All" or record.mode == mode_filter)
            and (status_filter == "All" or record.status == status_filter)
            and (model_filter == "All" or record.model == model_filter)
            and (rulebase_filter == "All" or rulebase_names[record.path] == rulebase_filter)
            and (
                from_date_filter is None
                or (
                    record_dates[record.path] is not None
                    and record_dates[record.path] >= from_date_filter
                )
            )
            and (food_filter == "All" or format_score(record.food_score) == food_filter)
            and (service_filter == "All" or format_score(record.service_score) == service_filter)
            and matches_metric_filters(
                metric_summaries[record.path],
                metric_status_filter=metric_status_filter,
                quality_preset=quality_preset,
                score_range=score_range,
            )
        ]
        filtered = sort_records(
            filtered,
            metric_summaries=metric_summaries,
            sort_by=sort_by,
        )

        if not filtered:
            return None

        st.caption(f"{len(filtered)} experiments match the current filters.")
        experiment_labels = {
            str(record.path): build_experiment_option_label(
                record,
                metric_summaries[record.path],
            )
            for record in filtered
        }
        selected_path = st.selectbox(
            "Experiment",
            options=[str(record.path) for record in filtered],
            format_func=lambda value: experiment_labels[value],
        )
        selected = next(record for record in filtered if str(record.path) == selected_path)

        st.divider()
        st.subheader("Main Parameters")
        params = select_main_parameters(selected)

        st.write(f"**Mode:** `{params['mode']}`")
        st.write(f"**Model:** `{params['model']}`")
        st.write(f"**Status:** `{params['status']}`")
        st.write(f"**Backend:** `{params['backend']}`")
        st.write(f"**Rulebase:** `{params['trace_model'] or 'Unavailable'}`")

        cols = st.columns(2)
        cols[0].metric("Time", format_seconds(params["generation_time_seconds"]))
        cols[1].metric("Output size", f"{format_number(params['output_word_count'])} words")

        cols = st.columns(2)
        cols[0].metric("Total tokens", format_number(params["total_tokens"]))
        cols[1].metric("Prompt tokens", format_number(params["prompt_tokens"]))

        sidebar_advanced = st.toggle("Advanced", value=False, key="sidebar_advanced")
        if sidebar_advanced:
            st.markdown("**Generation parameters**")
            st.json(selected.data.get("generation_parameters") or {})
            st.markdown("**CLI parameters**")
            st.json(((selected.data.get("cli") or {}).get("parameters")) or {})
            st.markdown("**Backend details**")
            st.json(selected.data.get("backend") or {})
            st.markdown("**Runtime details**")
            st.json(selected.data.get("runtime") or {})
            st.markdown("**Metrics**")
            st.json(selected.data.get("metrics") or {})

        return selected


def render_main(
    record: ExperimentRecord,
    full_trace: list[dict] | None,
    trace_details: dict[str, list[dict]],
    rulebase_resources: dict[str, object],
) -> None:
    st.caption(
        f"{format_timestamp(record.timestamp)} | {record.mode} | {record.backend} | "
        f"log: `{record.path.relative_to(ROOT_DIR)}`"
    )
    if record.data.get("mode_description"):
        st.write(record.data["mode_description"])

    render_top_metrics(record, trace_details)
    st.divider()
    render_llm_output(record)
    st.divider()
    render_fuzzy_model_output(record, full_trace, trace_details)
    st.divider()
    render_prompt_views(record)
    st.divider()
    render_rulebase_resources(record, rulebase_resources)
    st.divider()
    render_other_resources(record)


def render_top_metrics(record: ExperimentRecord, trace_details: dict[str, list[dict]]) -> None:
    outputs = trace_details.get("outputs") or []
    confidence = outputs[0]["confidence_text"] if outputs else None

    cols = st.columns(4)
    cols[0].metric("Food", f"{format_score(record.food_score)}/10")
    cols[1].metric("Service", f"{format_score(record.service_score)}/10")
    cols[2].metric("Tip", format_percent(record.tip_value))
    #cols[3].metric("Confidence", format_number(confidence))


def render_fuzzy_model_output(
    record: ExperimentRecord,
    full_trace: list[dict] | None,
    trace_details: dict[str, list[dict]],
) -> None:
    st.subheader("Fuzzy Model Output")

    left, right = st.columns(2)
    with left:
        st.markdown("**Inputs**")
        render_value_list(trace_details.get("inputs") or [], default_domain="/10")
        st.markdown("**Activated fuzzy propositions**")
        render_value_list(trace_details.get("active_propositions") or [])
    with right:
        st.markdown("**Outputs**")
        render_output_list(trace_details.get("outputs") or [])
        active_expressions = trace_details.get("active_expressions") or []
        if active_expressions:
            st.markdown("**Active logical expressions**")
            render_value_list(active_expressions)
        active_rules = trace_details.get("active_rules") or []
        if active_rules:
            st.markdown("**Active rules**")
            render_value_list(active_rules)

    toggle_cols = st.columns(2)
    show_llm_trace = toggle_cols[0].toggle(
        "Show actual input trace given to the LLM",
        value=False,
        key=f"show_llm_trace_{record.path.name}",
    )
    show_full_trace = toggle_cols[1].toggle(
        "Show full trace",
        value=False,
        key=f"show_full_trace_{record.path.name}",
    )

    if show_llm_trace:
        trace_text = llm_trace_input(record)
        if trace_text:
            st.code(trace_text, language="text")
        else:
            st.info("The simplified trace input was not found in the prompt payload.")

    if show_full_trace:
        if full_trace:
            st.json(full_trace)
        else:
            st.info("The raw trace file could not be loaded from the trace path in the log.")


def render_prompt_views(record: ExperimentRecord) -> None:
    show_prompts = st.toggle(
        "Show all LLM input prompts",
        value=False,
        key=f"show_all_prompts_{record.path.name}",
    )
    if not show_prompts:
        return

    messages = prompt_messages(record)
    if not messages:
        st.info("No prompt messages are available in this log.")
        return

    st.subheader("Prompt Messages")
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown").upper()
        st.markdown(f"**{index}. {role}**")
        st.code(str(message.get("content") or ""), language="text")


def render_llm_output(record: ExperimentRecord) -> None:
    st.subheader("LLM Explanation")
    if record.status == "success":
        output = record.data.get("output") or {}
        text = str(output.get("text") or "").strip()
        if text:
            st.markdown(text)
        else:
            st.info("This successful experiment log does not contain output text.")
        render_tcf_evaluation(record)
        response_metadata = output.get("response_metadata") or {}
        if response_metadata:
            with st.expander("Response metadata", expanded=False):
                st.json(response_metadata)
    else:
        error = record.data.get("error") or {}
        st.error(str(error.get("message") or "This experiment did not complete successfully."))
        if error:
            with st.expander("Error details", expanded=False):
                st.json(error)


def render_tcf_evaluation(record: ExperimentRecord) -> None:
    evaluations = tcf_evaluations(record)
    if not evaluations:
        st.caption("TCF score: unavailable")
        return

    latest = evaluations[0]
    status = str(latest.get("status") or "unknown")
    score = tcf_score(record)
    yes_count = latest.get("yes_count")
    total_questions = latest.get("total_questions")

    if status == "success" and score is not None:
        st.caption(
            f"TCF score: {score:.2f} ({format_number(yes_count)}/{format_number(total_questions)})"
        )
    elif status == "skipped":
        st.caption(
            f"TCF score: unavailable ({latest.get('skip_reason') or 'skipped'})"
        )
    elif status == "error":
        st.caption("TCF score: evaluation error")
    else:
        st.caption(f"TCF score: {status}")

    with st.expander("TCF evaluation details", expanded=False):
        if len(evaluations) > 1:
            selected_index = st.selectbox(
                "Evaluation run",
                options=list(range(len(evaluations))),
                index=0,
                format_func=lambda idx: tcf_evaluation_label(evaluations[idx], idx),
                key=f"tcf_evaluation_select_{record.path.name}",
            )
            evaluation = evaluations[selected_index]
        else:
            evaluation = latest

        status = str(evaluation.get("status") or "unknown")
        if status == "success":
            st.markdown("**Summary**")
            st.json(
                {
                    "timestamp_utc": evaluation.get("timestamp_utc"),
                    "score": evaluation.get("score"),
                    "yes_count": evaluation.get("yes_count"),
                    "total_questions": evaluation.get("total_questions"),
                    "metric_name": evaluation.get("metric_name"),
                    "metric_short_name": evaluation.get("metric_short_name"),
                }
            )
            for index, result in enumerate(
                ((evaluation.get("judge") or {}).get("results")) or [],
                start=1,
            ):
                st.markdown(f"**{index}. {result.get('section')} | {result.get('line_id')}**")
                st.write(f"Trace line: `{result.get('trace_line')}`")
                st.write(f"Question: {result.get('question')}")
                st.write(f"Answer: `{str(result.get('answer') or '').upper()}`")
                reason = str(result.get("reason") or "").strip()
                if reason:
                    st.write(f"Reason: {reason}")
        elif status == "error":
            st.json(evaluation.get("error") or evaluation)
        else:
            st.json(evaluation)


def render_rulebase_resources(
    record: ExperimentRecord,
    rulebase_resources: dict[str, object],
) -> None:
    show_advanced = st.toggle(
        "Advanced: show full rulebase and plots",
        value=False,
        key=f"show_rulebase_resources_{record.path.name}",
    )
    if not show_advanced:
        return

    rulebase_name = rulebase_resources.get("name") or "Unavailable"
    st.subheader("Rulebase Resources")
    st.write(f"**Rulebase:** `{rulebase_name}`")

    rules_text = rulebase_resources.get("rules_text")
    if rules_text:
        st.markdown("**Full rulebase**")
        st.code(str(rules_text), language="text")
    else:
        st.info("No rulebase text is available for this experiment.")

    description_text = rulebase_resources.get("description_text")
    if description_text:
        with st.expander("Rulebase description", expanded=False):
            st.markdown(str(description_text))

    plots = rulebase_resources.get("plots") or []
    if plots:
        st.markdown("**Plots**")
        render_plot_grid([Path(str(plot_path)) for plot_path in plots])
    else:
        st.info("No plots were found for this rulebase.")


def render_other_resources(record: ExperimentRecord) -> None:
    show_log_json = st.toggle(
        "Show experiment log JSON",
        value=False,
        key=f"show_log_json_{record.path.name}",
    )
    if show_log_json:
        st.subheader("Experiment Log JSON")
        st.json(record.data)


def render_plot_grid(plot_paths: list[Path]) -> None:
    if not plot_paths:
        return

    render_plot_row(plot_paths[:3], column_count=3)

    remaining = plot_paths[3:]
    for start in range(0, len(remaining), 2):
        render_plot_row(remaining[start : start + 2], column_count=2)


def render_plot_row(plot_paths: list[Path], column_count: int) -> None:
    columns = st.columns(column_count)
    for index, plot_path in enumerate(plot_paths):
        caption = plot_path.stem.replace("_", " ")
        with columns[index]:
            st.image(str(plot_path), caption=caption, width="stretch")


def render_value_list(items: list[dict], default_domain: str | None = None) -> None:
    if not items:
        st.caption("Unavailable")
        return
    lines = []
    for item in items:
        label = item.get("label") or "unknown"
        value_text = item.get("value_text") or format_number(item.get("value"))
        suffix = default_domain or ""
        if suffix and value_text:
            lines.append(f"- `{label}`: `{value_text}{suffix}`")
        else:
            lines.append(f"- `{label}`: `{value_text}`")
    st.markdown("\n".join(lines))


def render_output_list(items: list[dict]) -> None:
    if not items:
        st.caption("Unavailable")
        return
    lines = []
    for item in items:
        label = item.get("label") or "unknown"
        value_text = item.get("value_text") or format_number(item.get("value"))
        confidence_text = item.get("confidence_text") or ""
        line = f"- `{label}`: `{value_text} %`"
        if confidence_text:
            line += f" | confidence `{confidence_text}`"
        lines.append(line)
    st.markdown("\n".join(lines))


def build_experiment_option_label(record: ExperimentRecord, summary: MetricSummary) -> str:
    metric_text = metric_badge_text(summary)
    return f"{record.label} | {metric_text}"


def matches_metric_filters(
    summary: MetricSummary,
    metric_status_filter: str,
    quality_preset: str,
    score_range: tuple[float, float],
) -> bool:
    if metric_status_filter == "Has score" and summary.status != "success":
        return False
    if metric_status_filter == "Missing" and summary.status != "missing":
        return False
    if metric_status_filter == "Skipped" and summary.status != "skipped":
        return False
    if metric_status_filter == "Error" and summary.status != "error":
        return False

    if summary.status != "success" or summary.score is None:
        return quality_preset == "All"

    low, high = score_range
    if summary.score < low or summary.score > high:
        return False

    if quality_preset == "Good":
        return summary.score >= 0.80
    if quality_preset == "Mixed":
        return 0.40 <= summary.score < 0.80
    if quality_preset == "Bad":
        return summary.score < 0.40
    return True


def sort_records(
    records: list[ExperimentRecord],
    metric_summaries: dict[Path, MetricSummary],
    sort_by: str,
) -> list[ExperimentRecord]:
    if sort_by == "Highest score first":
        scored = [
            record
            for record in records
            if metric_summaries[record.path].status == "success"
            and metric_summaries[record.path].score is not None
        ]
        scored_paths = {record.path for record in scored}
        unscored = [record for record in records if record.path not in scored_paths]
        scored = sorted(scored, key=lambda record: record.timestamp, reverse=True)
        scored = sorted(
            scored,
            key=lambda record: metric_summaries[record.path].score or 0.0,
            reverse=True,
        )
        unscored = sorted(unscored, key=lambda record: record.timestamp, reverse=True)
        return scored + unscored
    if sort_by == "Lowest score first":
        scored = [
            record
            for record in records
            if metric_summaries[record.path].status == "success"
            and metric_summaries[record.path].score is not None
        ]
        scored_paths = {record.path for record in scored}
        unscored = [record for record in records if record.path not in scored_paths]
        scored = sorted(scored, key=lambda record: record.timestamp, reverse=True)
        scored = sorted(
            scored,
            key=lambda record: metric_summaries[record.path].score or 0.0,
        )
        unscored = sorted(unscored, key=lambda record: record.timestamp, reverse=True)
        return scored + unscored
    return sorted(records, key=lambda record: record.timestamp, reverse=True)


def metric_score_bounds(summaries: list[MetricSummary]) -> tuple[float, float]:
    scores = [summary.score for summary in summaries if summary.score is not None]
    if not scores:
        return (0.0, 1.0)

    minimum = min(scores)
    maximum = max(scores)
    if minimum >= 0.0 and maximum <= 1.0:
        return (0.0, 1.0)
    if minimum == maximum:
        return (minimum, maximum + 1.0)
    return (minimum, maximum)


def metric_badge_text(summary: MetricSummary) -> str:
    if summary.status == "success" and summary.score is not None:
        return f"{summary.key}={summary.score:.2f}"
    if summary.status == "error":
        return f"{summary.key}=error"
    if summary.status == "skipped":
        return f"{summary.key}=skipped"
    return f"{summary.key}=missing"


def parse_record_date(record: ExperimentRecord) -> date | None:
    timestamp = str(record.timestamp or "").strip()
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()
    except ValueError:
        return None


if __name__ == "__main__":
    main()
