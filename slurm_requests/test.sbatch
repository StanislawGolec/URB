#!/bin/bash
#SBATCH --job-name=urb_test_sa
#SBATCH --partition=rtx3080
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

nvidia-smi -L

cd $HOME/URB

# Activate virtual/conda environment
source activate URB 2>/dev/null || conda activate URB 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

# Abort job immediately if CUDA is not visible to PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch!'"

python -u scripts/qmix_torchrl_clusters.py \
    --id test_qmix_sa \
    --alg-conf test \
    --env-conf test \
    --task-conf test \
    --net saint_arnoult \
    --env-seed 42 \
    --torch-seed 42
