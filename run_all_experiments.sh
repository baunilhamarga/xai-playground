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

FOOD_MIN="${FOOD_MIN:-0}"
FOOD_MAX="${FOOD_MAX:-10}"
SERVICE_MIN="${SERVICE_MIN:-0}"
SERVICE_MAX="${SERVICE_MAX:-10}"

modes=(
    "zero-shot+trace"
    "3-shot+trace"
    "3-shot+(trace with model rulebase)"
)

food_count=$((FOOD_MAX - FOOD_MIN + 1))
service_count=$((SERVICE_MAX - SERVICE_MIN + 1))
total_runs=$((food_count * service_count * ${#modes[@]}))

run_index=0
failure_count=0

for food in $(seq "$FOOD_MIN" "$FOOD_MAX"); do
    for service in $(seq "$SERVICE_MIN" "$SERVICE_MAX"); do
        for mode in "${modes[@]}"; do
            run_index=$((run_index + 1))
            printf '[%d/%d] mode=%s food=%s service=%s\n' \
                "$run_index" "$total_runs" "$mode" "$food" "$service"

            if ! "$python_bin" -m xai_experiments "$@" \
                --mode "$mode" \
                --food "$food" \
                --service "$service"; then
                failure_count=$((failure_count + 1))
                printf '  failed: mode=%s food=%s service=%s\n' \
                    "$mode" "$food" "$service" >&2
            fi
        done
    done
done

if (( failure_count > 0 )); then
    printf 'Completed with %d failed runs.\n' "$failure_count" >&2
    exit 1
fi

printf 'Completed %d runs successfully.\n' "$total_runs"
