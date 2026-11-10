#!/bin/bash -l
#SBATCH --job-name=sta-memory
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00

# One STA memory run (Member 3), on CPU with the data environment. From the repository root:
#   sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-memory-%j.out" scripts/train_sta_memory.slurm \
#          --mode decoupled --steps 16
# Extra arguments go to scripts/train_sta_memory.py. Paths: scripts/sta_memory_env.sh.
# Optional GPU: add -p gpu_p100 --gpus-per-node=1 to sbatch, export EPIC_GPU_ENV=<CUDA venv>.
set -euo pipefail
source scripts/sta_memory_env.sh
RUN="${RUN_NAME:-run-${SLURM_JOB_ID}}"
python scripts/train_sta_memory.py \
    --features-dir "$EPIC_FEATURES" --manifest-dir "$EPIC_MANIFESTS" "${STA_TEST_ARGS[@]}" "${STA_INIT_ARGS[@]}" \
    --pool-cache "$EPIC_TEAM/cache" --num-workers 4 \
    --out-dir "$EPIC_TEAM/results/sta_memory/$RUN" "$@"
