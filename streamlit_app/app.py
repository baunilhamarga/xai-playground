from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from streamlit_app.data import (
        LOGS_DIR,
        ROOT_DIR,
        ExperimentRecord,
        extract_rulebase_name,
        format_number,
        format_percent,
        format_score,
        format_seconds,
        format_timestamp,
        llm_trace_input,
        load_experiments,
        load_full_trace,
        load_rulebase_resources,
        parse_trace_nodes,
        prompt_messages,
        select_main_parameters,
    )
except ModuleNotFoundError:
    from data import (
        LOGS_DIR,
        ROOT_DIR,
        ExperimentRecord,
        extract_rulebase_name,
        format_number,
        format_percent,
        format_score,
        format_seconds,
        format_timestamp,
        llm_trace_input,
        load_experiments,
        load_full_trace,
        load_rulebase_resources,
        parse_trace_nodes,
        prompt_messages,
        select_main_parameters,
    )


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
        st.header("Experiment Selection")

        mode_options = ["All"] + sorted({record.mode for record in experiments})
        status_options = ["All"] + sorted({record.status for record in experiments})
        model_options = ["All"] + sorted({record.model for record in experiments})
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
        score_cols = st.columns(2)
        food_filter = score_cols[0].selectbox("Food", options=food_options, index=0)
        service_filter = score_cols[1].selectbox("Service", options=service_options, index=0)

        filtered = [
            record
            for record in experiments
            if (mode_filter == "All" or record.mode == mode_filter)
            and (status_filter == "All" or record.status == status_filter)
            and (model_filter == "All" or record.model == model_filter)
            and (food_filter == "All" or format_score(record.food_score) == food_filter)
            and (service_filter == "All" or format_score(record.service_score) == service_filter)
        ]

        if not filtered:
            return None

        selected_path = st.selectbox(
            "Experiment",
            options=[str(record.path) for record in filtered],
            format_func=lambda value: next(
                record.label for record in filtered if str(record.path) == value
            ),
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


if __name__ == "__main__":
    main()
