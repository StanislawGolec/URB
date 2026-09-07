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

CONTAINER="/shared/sets/singularity/sumo1_26_0patched.sif"

singularity exec --nv "$CONTAINER" /bin/bash -c '
    cd $HOME/URB
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate URB

    python -c "import torch; assert torch.cuda.is_available(), \"CUDA not available in PyTorch!\""

    python -u scripts/qmix_torchrl_clusters.py \
        --id test_qmix_sa \
        --alg-conf test \
        --env-conf test \
        --task-conf test \
        --net saint_arnoult \
        --env-seed 42 \
        --torch-seed 42
'
