#!/bin/bash
#SBATCH --job-name=urb_mixed_sweep
#SBATCH --partition=student
#SBATCH --qos=big
#SBATCH --time=7-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-16
#SBATCH --output=container_printouts/slurm-%A_%a.out

nvidia-smi -L

# 8 mixed reward configurations (Issue #44)
CONFIGS=(
    "mixed_reward_ind100_all0"
    "mixed_reward_ind85_all15"
    "mixed_reward_ind70_all30"
    "mixed_reward_ind50_all50"
    "mixed_reward_ind30_all70"
    "mixed_reward_ind15_all85"
    "mixed_reward_ind05_all95"
    "mixed_reward_ind0_all100"
)

# Tasks 1-8: QMIX Clusters
# Tasks 9-16: IQL Clusters
if [ "$SLURM_ARRAY_TASK_ID" -le 8 ]; then
    ALGO="qmix_clusters"
    SCRIPT="scripts/qmix_torchrl_clusters.py"
    ALG_CONF="config1"
    ENV_CONF="clusters_eta"
    RATIO_IDX=$((SLURM_ARRAY_TASK_ID - 1))
    TASK_CONF="${CONFIGS[$RATIO_IDX]}"
    EXP_ID="qmix_clusters_${TASK_CONF}_ing2_s42"
else
    ALGO="iql_clusters"
    SCRIPT="scripts/iql_clusters.py"
    ALG_CONF="config1"
    ENV_CONF="clusters_eta"
    RATIO_IDX=$((SLURM_ARRAY_TASK_ID - 9))
    TASK_CONF="${CONFIGS[$RATIO_IDX]}"
    EXP_ID="iql_clusters_${TASK_CONF}_ing2_s42"
fi

PATH_PROGRAM="${PATH_PROGRAM:-/home/$USER/URB}"
PUT_PROGRAM_TO="/app"
PATH_SUMO_CONTAINER="/shared/sets/singularity/sumo1_26_0patched.sif"
CMD_PATH="$PATH_PROGRAM/slurm_requests/cmd_container.sh"
PRINTS_SAVE_PATH="$PATH_PROGRAM/slurm_requests/container_printouts/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

mkdir -p "$PATH_PROGRAM/slurm_requests/container_printouts"

echo "=========================================================="
echo "Starting Job: $EXP_ID"
echo "Algorithm:    $ALGO ($SCRIPT)"
echo "Task Config:  $TASK_CONF"
echo "Array Task:   $SLURM_ARRAY_TASK_ID / 16"
echo "=========================================================="

singularity exec --nv \
    --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" \
    "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" \
    "$SCRIPT" \
    --id "$EXP_ID" \
    --alg-conf "$ALG_CONF" \
    --env-conf "$ENV_CONF" \
    --task-conf "$TASK_CONF" \
    --net ingolstadt_custom2 \
    --env-seed 42 \
    --torch-seed 42 \
    2>&1 | tee "$PRINTS_SAVE_PATH"
