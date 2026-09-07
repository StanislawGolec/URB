#!/bin/bash
#SBATCH --job-name=urb_test_sa
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

nvidia-smi -L

cd $HOME/URB

# Activate Conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate URB

export SUMO_HOME=$CONDA_PREFIX/share/sumo
export PATH=$CONDA_PREFIX/bin:$PATH

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
