#!/bin/bash
#SBATCH --job-name=qmix_ing2_alt10_s42
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

nvidia-smi -L

cd $HOME/URB

CONTAINER="/shared/sets/singularity/sumo1_26_0patched.sif"

singularity exec --nv "$CONTAINER" /bin/bash -c '
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
    cd $HOME/URB
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate URB

    python -c "import torch; assert torch.cuda.is_available(), \"CUDA not available in PyTorch!\""

    python -u scripts/qmix_torchrl_clusters.py \
        --id qmix_clusters_alt10_ing2_s42 \
        --alg-conf config1 \
        --env-conf clusters_eta \
        --task-conf altruistic_10 \
        --net ingolstadt_custom2 \
        --env-seed 42 \
        --torch-seed 42
'
