#!/bin/bash
#SBATCH --job-name=urb_test_sa
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --mem=16G
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

# Run container by mounting repository to /app and running cmd_container.sh
singularity exec --nv \
    --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" \
    "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" \
    scripts/qmix_torchrl_clusters.py \
    --id test_qmix_sa \
    --alg-conf test \
    --env-conf test \
    --task-conf test \
    --net saint_arnoult \
    --env-seed 42 \
    --torch-seed 42 \
    2>&1 | tee "$PRINTS_SAVE_PATH"
