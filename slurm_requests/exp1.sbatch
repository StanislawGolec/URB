#!/bin/bash
#SBATCH --job-name=qmix_ing2_alt10_s42
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=container_printouts/slurm-%j.out

nvidia-smi -L

PATH_PROGRAM="${PATH_PROGRAM:-/home/$USER/URB}"
PUT_PROGRAM_TO="/app"
PATH_SUMO_CONTAINER="/shared/sets/singularity/sumo1_26_0patched.sif"
CMD_PATH="$PATH_PROGRAM/slurm_requests/cmd_container.sh"
PRINTS_SAVE_PATH="$PATH_PROGRAM/slurm_requests/container_printouts/output_${SLURM_JOB_ID}.txt"

mkdir -p "$PATH_PROGRAM/slurm_requests/container_printouts"

if [ "$#" -gt 0 ]; then
    CMD_ARGS=("$@")
else
    CMD_ARGS=(
        scripts/qmix_torchrl_clusters.py
        --id qmix_clusters_alt10_ing2_s42
        --alg-conf config1
        --env-conf clusters_eta
        --task-conf altruistic_10
        --net ingolstadt_custom2
        --env-seed 42
        --torch-seed 42
    )
fi

singularity exec --nv \
    --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" \
    "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" \
    "${CMD_ARGS[@]}" \
    2>&1 | tee "$PRINTS_SAVE_PATH"
