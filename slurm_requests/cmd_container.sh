#!/bin/bash
set -eo pipefail

# Prepend internal container libraries to avoid GLIBC mismatch with host libGL
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Navigate to the mounted application folder
cd /app

# Ensure user environment (such as WANDB_API_KEY) is available
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" 2>/dev/null || true
fi

# Activate Conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate URB

# Fail before simulation if cloud tracking is unavailable. Offline use is explicit.
export WANDB_ENTITY="${WANDB_ENTITY:-aintern26coexistence}"
export WANDB_PROJECT="${WANDB_PROJECT:-URB-Traffic-Routing}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_REQUIRE_ONLINE="${WANDB_REQUIRE_ONLINE:-1}"

# Verify GPU is accessible to PyTorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available in PyTorch!'"

if [[ "${1:-}" == *qmix_torchrl_clusters.py ]]; then
    python scripts/qmix_memory_preflight.py "${@:2}"
fi

# Run the passed Python script with arguments
echo "--- Starting: python -u $@ ---"
python -u "$@"
