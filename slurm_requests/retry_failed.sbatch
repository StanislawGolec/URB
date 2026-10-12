#!/bin/bash
# Fifteen runs that failed with CUDA OOM. Submit through launch_retry_failed.sh.
#SBATCH --job-name=urb_qmix_retry
#SBATCH --partition=rknodes
#SBATCH --qos=big
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-15
#SBATCH --output=container_printouts/retry-%A_%a.out

set -euo pipefail
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([1-9]|1[0-5])$ ]] || { echo "Expected task 1-15" >&2; exit 2; }

RATIOS=(40 50 60 70 80 90 100)
MIXTURES=(
    mixed_reward_ind100_all0
    mixed_reward_ind85_all15
    mixed_reward_ind70_all30
    mixed_reward_ind50_all50
    mixed_reward_ind30_all70
    mixed_reward_ind15_all85
    mixed_reward_ind05_all95
    mixed_reward_ind0_all100
)
if (( SLURM_ARRAY_TASK_ID <= 7 )); then
    RATIO="${RATIOS[$((SLURM_ARRAY_TASK_ID - 1))]}"
    TASK_CONF="altruistic_${RATIO}"
    EXP_ID="qmix_clusters_alt${RATIO}_ing2_s42"
else
    TASK_CONF="${MIXTURES[$((SLURM_ARRAY_TASK_ID - 8))]}"
    EXP_ID="qmix_clusters_${TASK_CONF}_ing2_s42"
fi

PATH_PROGRAM="${PATH_PROGRAM:-/home/$USER/URB}"
PATH_SUMO_CONTAINER="${PATH_SUMO_CONTAINER:-/shared/sets/singularity/sumo1_26_0patched.sif}"
CMD_PATH="$PATH_PROGRAM/slurm_requests/cmd_container.sh"
PRINTS_SAVE_PATH="$PATH_PROGRAM/slurm_requests/container_printouts/retry_output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

if [[ -e "$PATH_PROGRAM/results/$EXP_ID" ]]; then
    echo "Refusing to overwrite existing result directory: $PATH_PROGRAM/results/$EXP_ID" >&2
    exit 2
fi
[[ -d "$PATH_PROGRAM/results/${EXP_ID}_failed_oom" ]] || {
    echo "Missing archived failed result: $PATH_PROGRAM/results/${EXP_ID}_failed_oom" >&2
    exit 2
}
mkdir -p "$PATH_PROGRAM/slurm_requests/container_printouts"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Retrying $EXP_ID with the original algorithm config1"

singularity exec --nv \
    --bind "$PATH_PROGRAM":/app \
    "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" \
    scripts/qmix_torchrl_clusters.py \
    --id "$EXP_ID" \
    --alg-conf config1 \
    --env-conf clusters_eta \
    --task-conf "$TASK_CONF" \
    --net ingolstadt_custom2 \
    --env-seed 42 \
    --torch-seed 42 \
    2>&1 | tee "$PRINTS_SAVE_PATH"
