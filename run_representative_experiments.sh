#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/xai/bin/python" ]]; then
    python_bin="$ROOT_DIR/xai/bin/python"
else
    python_bin="python3"
fi

modes=(
    "zero-shot+trace"
    "3-shot+trace"
    "3-shot+(trace with model rulebase)"
)

# Stable 10-case subset covering low, medium, high, and asymmetric inputs.
# Each case is run once per mode so comparisons stay aligned.
cases=(
    "0|0|worst_case"
    "5|5|balanced_midpoint"
    "10|10|best_case"
    "2|7|default_trace_case"
    "0|10|service_dominant_positive"
    "10|0|food_dominant_negative_service"
    "2|2|low_inputs"
    "8|5|strong_food_mid_service"
    "5|8|mid_food_strong_service"
    "9|9|near_best_case"
)

total_runs=$((${#cases[@]} * ${#modes[@]}))
run_index=0
failure_count=0

for case_entry in "${cases[@]}"; do
    IFS='|' read -r food service label <<< "$case_entry"
    for mode in "${modes[@]}"; do
        run_index=$((run_index + 1))

        printf '[%d/%d] case=%s mode=%s food=%s service=%s\n' \
            "$run_index" "$total_runs" "$label" "$mode" "$food" "$service"

        if ! "$python_bin" -m xai_experiments "$@" \
            --mode "$mode" \
            --food "$food" \
            --service "$service"; then
            failure_count=$((failure_count + 1))
            printf '  failed: case=%s mode=%s food=%s service=%s\n' \
                "$label" "$mode" "$food" "$service" >&2
        fi
    done
done

if (( failure_count > 0 )); then
    printf 'Completed with %d failed runs.\n' "$failure_count" >&2
    exit 1
fi

printf 'Completed %d representative runs successfully.\n' "$total_runs"
