#!/bin/bash

#MSUB -r theta_lambda_batches
#MSUB -o logs/theta_lambda_batches_out
#MSUB -e logs/theta_lambda_batches_err
#MSUB -q milan
#MSUB -Q normal
#MSUB -N 1
#MSUB -n 1
#MSUB -c 1
#MSUB -T 86400
#MSUB -m scratch,work

# Submit from scripts/AMS_butane/theta_lambda_mp0a with:
#   ccc_msub submit_lambda_batches_topaze.sh
# The 70 A100 AMS jobs of one (sample, lambda) point finish before the next
# point starts, so the initial-condition pool can safely be reused.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd -P)}"
CLEAN_SCRIPT="${CLEAN_SCRIPT:-$ROOT_DIR/../clean_conds.py}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
START_SLEEP_SECONDS="${START_SLEEP_SECONDS:-10}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-1800}"

job_count() {
    ccc_mstat -u "$USER" 2>/dev/null | awk '
        NR > 2 && $0 !~ /^[-[:space:]]*$/ { n++ }
        END { print n + 0 }
    '
}

wait_for_jobs_to_appear() {
    local baseline="$1" waited=0 n_jobs
    while true; do
        n_jobs="$(job_count)"
        if [ "$n_jobs" -gt "$baseline" ]; then
            return
        fi
        if [ "$waited" -ge "$START_TIMEOUT_SECONDS" ]; then
            echo "Aucun job AMS visible apres ${START_TIMEOUT_SECONDS}s." >&2
            exit 1
        fi
        sleep "$START_SLEEP_SECONDS"
        waited=$((waited + START_SLEEP_SECONDS))
    done
}

wait_for_queue_to_return() {
    local baseline="$1" n_jobs
    while true; do
        n_jobs="$(job_count)"
        if [ "$n_jobs" -le "$baseline" ]; then
            return
        fi
        echo "Queue: $n_jobs jobs; attente avant le point lambda suivant..."
        sleep "$SLEEP_SECONDS"
    done
}

if [ ! -f "$CLEAN_SCRIPT" ]; then
    echo "Script de nettoyage introuvable: $CLEAN_SCRIPT" >&2
    exit 1
fi

mkdir -p "$ROOT_DIR/logs"
baseline_jobs="$(job_count)"
echo "ROOT_DIR: $ROOT_DIR"
echo "Baseline queue: $baseline_jobs jobs"

for sample_dir in "$ROOT_DIR"/sample_[0-9]*; do
    [ -d "$sample_dir" ] || continue
    for lambda_name in \
        lambda_m05 lambda_m04 lambda_m03 lambda_m02 lambda_m01 \
        lambda_p00 lambda_p01 lambda_p02 lambda_p03 lambda_p04 lambda_p05; do
        lambda_dir="$sample_dir/$lambda_name"
        [ -d "$lambda_dir" ] || continue
        echo "=== ${lambda_dir#$ROOT_DIR/} ==="
        python "$CLEAN_SCRIPT"
        (cd "$lambda_dir" && python make_input_files.py)
        (cd "$lambda_dir" && ccc_msub run_all_topaze)
        wait_for_jobs_to_appear "$baseline_jobs"
        wait_for_queue_to_return "$baseline_jobs"
        (cd "$lambda_dir" && python compute_probs.py)
    done
done

echo "Termine. Agreger ensuite avec: python get_results.py"
