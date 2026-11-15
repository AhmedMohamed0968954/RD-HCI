#!/bin/bash -l
#SBATCH --job-name=probe-steps
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00

# Member 3: Member 2's frozen STA probe over all 360k bank clips + the STA clips, once (~1.8 GB out).
#   sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/probe-steps-%j.out" scripts/precompute_probe_steps.slurm
# Output: $EPIC_PROBE_STEPS (scripts/sta_memory_env.sh). Then the memory runs with these steps:
#   sbatch --dependency=afterok:<this job id> ... scripts/sta_memory_pretrain.slurm <seed> probe
set -euo pipefail
source scripts/sta_memory_env.sh
[[ -f "$EPIC_PROBE_CKPT" ]] || { echo "Missing $EPIC_PROBE_CKPT (Member 2's STA probe)"; exit 1; }
python scripts/precompute_probe_steps.py --features-dir "$EPIC_FEATURES" --probe-ckpt "$EPIC_PROBE_CKPT" \
    --out-dir "$EPIC_PROBE_STEPS"
