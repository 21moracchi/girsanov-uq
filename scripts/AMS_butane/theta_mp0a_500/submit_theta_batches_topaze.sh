#!/bin/bash

#MSUB -r theta_batches
#MSUB -o logs/theta_batches_out
#MSUB -e logs/theta_batches_err
#MSUB -q milan
#MSUB -Q normal
#MSUB -N 1
#MSUB -n 1
#MSUB -c 1
#MSUB -T 86400
#MSUB -m scratch,work

set -euo pipefail

# Submit from scripts/AMS_butane_3/theta_mp0a_500 on Topaze.
# Usage:
#   ccc_msub submit_theta_batches_topaze.sh
#   THETA_RANGE=5-10 ccc_msub submit_theta_batches_topaze.sh
# Optional:
#   THETA_LIST="theta_0 theta_3" ccc_msub submit_theta_batches_topaze.sh

# With ccc_msub, $0 can point to a scheduler copy of this script.
# The job is expected to start from the directory where ccc_msub was called.
ROOT_DIR="${ROOT_DIR:-$(pwd -P)}"
CLEAN_DIR="$ROOT_DIR/../ams_runs_mp0a_500/sample_ini_conds"
CLEAN_SCRIPT="${CLEAN_SCRIPT:-clean_conds.py}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
START_SLEEP_SECONDS="${START_SLEEP_SECONDS:-10}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-1800}"

job_count() {
    ccc_mstat -u "$USER" 2>/dev/null | awk '
        NR > 2 && $0 !~ /^[-[:space:]]*$/ { n++ }
        END { print n + 0 }
    '
}

wait_for_baseline_queue() {
    local baseline_jobs="$1"
    local n_jobs

    while true; do
        n_jobs="$(job_count)"
        if [ "$n_jobs" -le "$baseline_jobs" ]; then
            break
        fi
        echo "Queue: $n_jobs jobs, attente retour a $baseline_jobs avant le theta suivant..."
        sleep "$SLEEP_SECONDS"
    done
}

wait_for_theta_jobs_to_appear() {
    local baseline_jobs="$1"
    local waited=0
    local n_jobs

    while true; do
        n_jobs="$(job_count)"
        if [ "$n_jobs" -gt "$baseline_jobs" ]; then
            echo "Queue: $n_jobs jobs, les jobs du theta sont visibles."
            break
        fi

        if [ "$waited" -ge "$START_TIMEOUT_SECONDS" ]; then
            echo "Erreur: aucun job supplementaire visible apres ${START_TIMEOUT_SECONDS}s." >&2
            echo "Queue actuelle: $n_jobs jobs, baseline: $baseline_jobs." >&2
            exit 1
        fi

        echo "Queue: $n_jobs jobs, attente de l'apparition des jobs du theta..."
        sleep "$START_SLEEP_SECONDS"
        waited=$((waited + START_SLEEP_SECONDS))
    done
}

make_theta_range() {
    local range="$1"
    local start="${range%-*}"
    local end="${range#*-}"
    local i

    if ! [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$start" -le "$end" ]]; then
        echo "Erreur: intervalle invalide '$range' (exemple attendu: 5-10)." >&2
        exit 1
    fi

    theta_dirs=()
    for ((i = start; i <= end; i++)); do
        theta_dirs+=("theta_$i")
    done
}

if [ -n "${THETA_RANGE:-}" ]; then
    make_theta_range "$THETA_RANGE"
elif [ "$#" -eq 1 ] && [[ "$1" =~ ^[0-9]+-[0-9]+$ ]]; then
    make_theta_range "$1"
elif [ "$#" -gt 0 ]; then
    theta_dirs=("$@")
elif [ -n "${THETA_LIST:-}" ]; then
    read -r -a theta_dirs <<< "$THETA_LIST"
else
    theta_dirs=(theta_*)
fi

mkdir -p "$ROOT_DIR/logs"
echo "ROOT_DIR: $ROOT_DIR"
echo "Theta dirs: ${theta_dirs[*]}"
baseline_jobs="$(job_count)"
echo "Baseline queue: $baseline_jobs jobs."

for theta_dir in "${theta_dirs[@]}"; do
    if [ ! -d "$ROOT_DIR/$theta_dir" ]; then
        echo "Skip: $theta_dir n'existe pas."
        continue
    fi

    if [ ! -f "$CLEAN_DIR/$CLEAN_SCRIPT" ]; then
        echo "Erreur: $CLEAN_DIR/$CLEAN_SCRIPT introuvable."
        exit 1
    fi

    echo "=== $theta_dir ==="
    echo "Nettoyage des conditions initiales..."
    (cd "$CLEAN_DIR" && python "$CLEAN_SCRIPT")

    echo "Preparation des 70 dossiers AMS..."
    (cd "$ROOT_DIR/$theta_dir" && python make_input_files.py)

    echo "Soumission du job de lancement AMS de $theta_dir..."
    (cd "$ROOT_DIR/$theta_dir" && ccc_msub run_all_topaze)
  
    echo "Attente de la fin de tous les jobs avant le theta suivant..."
    wait_for_theta_jobs_to_appear "$baseline_jobs"
    wait_for_baseline_queue "$baseline_jobs"
    echo "Calcul des probs..."
    (cd "$ROOT_DIR/$theta_dir" && python compute_probs.py)
done

echo "Termine."
