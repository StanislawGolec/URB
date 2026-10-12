#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p container_printouts
sbatch exp1.sbatch
sbatch exp2.sbatch
