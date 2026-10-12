#!/bin/bash
# Review with --dry-run; --check-access asks Slurm without submitting.
# Optional: TASKS, ARRAY_LIMIT, PARTITION, QOS, WALLTIME.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ "$#" -le 1 && ( "${1:-}" == "" || "${1:-}" == "--dry-run" || "${1:-}" == "--check-access" ) ]] || {
    echo "Usage: $0 [--dry-run|--check-access]" >&2
    exit 2
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" analysis/archive_failed_sweeps.py
ARRAY="${TASKS:-1-15}"
[[ "$ARRAY" =~ ^([1-9]|1[0-5])(-([1-9]|1[0-5]))?$ ]] || {
    echo "TASKS must be a task number or range within 1-15" >&2
    exit 2
}
if [[ "$ARRAY" == *-* ]]; then
    START="${ARRAY%-*}"
    END="${ARRAY#*-}"
    (( START <= END )) || { echo "TASKS range must be ascending" >&2; exit 2; }
fi
if [[ -n "${ARRAY_LIMIT:-}" ]]; then
    [[ "$ARRAY_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "ARRAY_LIMIT must be positive" >&2; exit 2; }
    ARRAY="${ARRAY}%${ARRAY_LIMIT}"
fi
COMMAND=(sbatch --partition="${PARTITION:-rknodes}" --qos="${QOS:-big}"
         --time="${WALLTIME:-7-00:00:00}" --array="$ARRAY"
         slurm_requests/retry_failed.sbatch)
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "${1:-}" == "--dry-run" ]]; then
    exit 0
fi
if [[ "${1:-}" == "--check-access" ]]; then
    sbatch --test-only "${COMMAND[@]:1}"
    exit 0
fi

"$PYTHON_BIN" analysis/archive_failed_sweeps.py --execute
mkdir -p slurm_requests/container_printouts
"${COMMAND[@]}"
