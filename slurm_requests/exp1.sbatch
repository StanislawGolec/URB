#!/bin/bash
#SBATCH --job-name=qmix_ing2_alt10_s42
#SBATCH --partition=rtx3080
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

nvidia-smi -L

cd $HOME/URB

# Activate virtual/conda environment (adjust name if different)
source activate URB 2>/dev/null || conda activate URB 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

python -u scripts/qmix_torchrl_clusters.py \
    --id qmix_clusters_alt10_ing2_s42 \
    --alg-conf config1 \
    --env-conf clusters_eta \
    --task-conf altruistic_10 \
    --net ingolstadt_custom2 \
    --env-seed 42 \
    --torch-seed 42
