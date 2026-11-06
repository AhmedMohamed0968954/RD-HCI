#!/bin/bash -l
#SBATCH --job-name=sta-memory
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00

# One STA memory run (Member 3). Submit from the repository root:
#   sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-memory-%j.out" scripts/train_sta_memory.slurm \
#          --mode decoupled --steps 16 --seed 42
# Extra arguments go straight to scripts/train_sta_memory.py.
# Needs in ~/.bashrc (see docs/MEMORY_INTERFACE.md):
#   EPIC_TEAM       team folder, e.g. /lustre1/scratch/394/vsc39484/epic-kitchens/team
#   EPIC_FEATURES   one feature version folder, e.g. $EPIC_TEAM/features/<version>
#   EPIC_MANIFESTS  shared split, e.g. $EPIC_PROJECT/manifests/data-<jobid>
# Optional: EPIC_GPU_ENV=<venv folder>; otherwise the course conda env "rd_lab" is used.
set -euo pipefail
: "${EPIC_TEAM:?set EPIC_TEAM}" "${EPIC_FEATURES:?set EPIC_FEATURES}" "${EPIC_MANIFESTS:?set EPIC_MANIFESTS}"
if [[ -n "${EPIC_GPU_ENV:-}" ]]; then source "$EPIC_GPU_ENV/bin/activate"; else conda activate rd_lab; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print(torch.cuda.get_device_name(0))"

TEST_ARGS=()
if [[ -f "$EPIC_PROJECT/annotations/EPIC_100_test.csv" ]]; then
    TEST_ARGS=(--test-csv "$EPIC_PROJECT/annotations/EPIC_100_test.csv")
fi
RUN="${RUN_NAME:-run-${SLURM_JOB_ID}}"
python scripts/train_sta_memory.py \
    --features-dir "$EPIC_FEATURES" --manifest-dir "$EPIC_MANIFESTS" "${TEST_ARGS[@]}" \
    --pool-cache "$EPIC_TEAM/cache" --num-workers 6 \
    --out-dir "$EPIC_TEAM/results/sta_memory/$RUN" "$@"
