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

# Activate Conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate URB

export SUMO_HOME=$CONDA_PREFIX/share/sumo
export PATH=$CONDA_PREFIX/bin:$PATH

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
