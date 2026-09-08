#!/bin/bash
#SBATCH --job-name=qmix_alt_sweep
#SBATCH --partition=student
#SBATCH --qos=big
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --output=container_printouts/slurm-%A_%a.out

nvidia-smi -L

# Array of altruistic percentages: 10% to 100%
RATIOS=(10 20 30 40 50 60 70 80 90 100)
INDEX=$((SLURM_ARRAY_TASK_ID - 1))
ALT_RATIO="${RATIOS[$INDEX]}"

TASK_CONF="altruistic_${ALT_RATIO}"
EXP_ID="qmix_clusters_alt${ALT_RATIO}_ing2_s42"

PATH_PROGRAM="${PATH_PROGRAM:-/home/$USER/URB}"
PUT_PROGRAM_TO="/app"
PATH_SUMO_CONTAINER="/shared/sets/singularity/sumo1_26_0patched.sif"
CMD_PATH="$PATH_PROGRAM/slurm_requests/cmd_container.sh"
PRINTS_SAVE_PATH="$PATH_PROGRAM/slurm_requests/container_printouts/output_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"

mkdir -p "$PATH_PROGRAM/slurm_requests/container_printouts"

echo "=========================================================="
echo "Starting Job: $EXP_ID"
echo "Task Config:  $TASK_CONF"
echo "Array Task:   $SLURM_ARRAY_TASK_ID / 10 (Ratio: ${ALT_RATIO}%)"
echo "=========================================================="

singularity exec --nv \
    --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" \
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
