#!/bin/bash
#SBATCH --job-name=urb_gpu_probe
#SBATCH --partition=rknodes
#SBATCH --qos=big
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_requests/container_printouts/gpu-probe-%j.out

set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
