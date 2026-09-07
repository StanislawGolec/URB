#!/bin/bash
#SBATCH --job-name=qmix_ing2_alt10_s43
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

nvidia-smi -L

cd $HOME/URB

# Activate virtual/conda environment (adjust name if different)
source activate URB 2>/dev/null || conda activate URB 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

# Abort job immediately if CUDA is not visible to PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch!'"

python -u scripts/qmix_torchrl_clusters.py \
    --id qmix_clusters_alt10_ing2_s43 \
    --alg-conf config1 \
    --env-conf clusters_eta \
    --task-conf altruistic_10 \
    --net ingolstadt_custom2 \
    --env-seed 43 \
    --torch-seed 43
