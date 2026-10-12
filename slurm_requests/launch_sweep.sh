#!/bin/bash
# Preview: bash slurm_requests/launch_sweep.sh mixed --dry-run
# Optional overrides: PARTITION, QOS, WALLTIME, ARRAY_LIMIT, EXP_SUFFIX.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-}"
case "$MODE" in
    altruistic) SCRIPT=sweep_altruistic.sbatch; ARRAY=1-10 ;;
    mixed) SCRIPT=sweep_mixed_reward.sbatch; ARRAY=1-16 ;;
    *) echo "Usage: $0 {altruistic|mixed} [--dry-run]" >&2; exit 2 ;;
esac
[[ "$#" -le 2 && ( "${2:-}" == "" || "${2:-}" == "--dry-run" ) ]] || { echo "Unknown argument" >&2; exit 2; }
if [[ -n "${ARRAY_LIMIT:-}" ]]; then
    [[ "$ARRAY_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "ARRAY_LIMIT must be positive" >&2; exit 2; }
    ARRAY="${ARRAY}%${ARRAY_LIMIT}"
fi
export EXP_SUFFIX="${EXP_SUFFIX:-_bonk}"
COMMAND=(sbatch --partition="${PARTITION:-rknodes}" --qos="${QOS:-big}"
         --time="${WALLTIME:-7-00:00:00}" --array="$ARRAY" "$SCRIPT")
printf 'EXP_SUFFIX=%q ' "$EXP_SUFFIX"
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "${2:-}" != "--dry-run" ]]; then
    mkdir -p container_printouts
    "${COMMAND[@]}"
fi
