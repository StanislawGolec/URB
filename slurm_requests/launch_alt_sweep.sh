#!/bin/bash
# Master launcher for Altruistic Sweep experiments (Issue #40)
# Usage:
#   bash launch_alt_sweep.sh array       -> Launches all 10 experiments on bonk
#   bash launch_alt_sweep.sh 10          -> Launches single experiment for alt 10
#   bash launch_alt_sweep.sh all         -> Submits individual jobs for 10 to 100

set -euo pipefail

MODE="${1:-array}"

cd "$(dirname "$0")"

mkdir -p container_printouts

if [ "$MODE" = "array" ]; then
    bash launch_sweep.sh altruistic
elif [ "$MODE" = "all" ]; then
    echo "Submitting 10 individual sbatch jobs for ratios 10% to 100%..."
    for RATIO in 10 20 30 40 50 60 70 80 90 100; do
        JOB_NAME="qmix_ing2_alt${RATIO}_s42"
        EXP_ID="qmix_clusters_alt${RATIO}_ing2_s42${EXP_SUFFIX:-_bonk}"
        TASK_CONF="altruistic_${RATIO}"
        echo "Submitting $EXP_ID..."
        sbatch --job-name="$JOB_NAME" \
               --output="container_printouts/slurm-${EXP_ID}-%j.out" \
               exp1.sbatch \
               scripts/qmix_torchrl_clusters.py \
               --id "$EXP_ID" \
               --alg-conf config1 \
               --env-conf clusters_eta \
               --task-conf "$TASK_CONF" \
               --net ingolstadt_custom2 \
               --env-seed 42 \
               --torch-seed 42
    done
else
    # Single ratio passed as argument (10, 20, ..., 100)
    RATIO="$MODE"
    [[ "$RATIO" =~ ^([1-9]0|100)$ ]] || { echo "Expected array, all, or 10..100 in steps of 10" >&2; exit 2; }
    JOB_NAME="qmix_ing2_alt${RATIO}_s42"
    EXP_ID="qmix_clusters_alt${RATIO}_ing2_s42${EXP_SUFFIX:-_bonk}"
    TASK_CONF="altruistic_${RATIO}"
    echo "Submitting single experiment: $EXP_ID..."
    sbatch --job-name="$JOB_NAME" \
           --output="container_printouts/slurm-${EXP_ID}-%j.out" \
           exp1.sbatch \
           scripts/qmix_torchrl_clusters.py \
           --id "$EXP_ID" \
           --alg-conf config1 \
           --env-conf clusters_eta \
           --task-conf "$TASK_CONF" \
           --net ingolstadt_custom2 \
           --env-seed 42 \
           --torch-seed 42
fi
