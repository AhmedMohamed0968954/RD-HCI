#!/bin/bash -l
#SBATCH --job-name=sta-memory-ablation
#SBATCH --clusters=wice
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --array=0-9%3

# Member 3 ablation: AGA single vs decoupled x history length S, plus "no exact STA clip"
# and decoupled memory with Member 4's three-way gate (history + present; future added later).
# Run ONE normal job first so the pooled feature cache exists, then:
#   sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-ablation-%A_%a.out" scripts/sta_memory_ablation.slurm
# For more seeds: SEED=1 sbatch ... (each seed gets its own run folders).
# Afterwards: python scripts/report_sta_memory.py --runs-dir "$EPIC_TEAM/results/sta_memory" --out-dir <dir>
set -euo pipefail
: "${EPIC_TEAM:?set EPIC_TEAM}" "${EPIC_FEATURES:?set EPIC_FEATURES}" "${EPIC_MANIFESTS:?set EPIC_MANIFESTS}"
if [[ -n "${EPIC_GPU_ENV:-}" ]]; then source "$EPIC_GPU_ENV/bin/activate"; else conda activate rd_lab; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1

MODES=(single single single single decoupled decoupled decoupled decoupled decoupled decoupled)
STEPS=(4 8 16 30 4 8 16 30 16 16)
EXTRA=("" "" "" "" "" "" "" "" "--no-anchor" "--fusion three_way")
SUFFIX=("" "" "" "" "" "" "" "" "_noanchor" "_3way")
i=$SLURM_ARRAY_TASK_ID
SEED="${SEED:-42}"
NAME="${MODES[$i]}_S${STEPS[$i]}_seed${SEED}"
NAME="${NAME}${SUFFIX[$i]}"
OUT="$EPIC_TEAM/results/sta_memory/$NAME"
if [[ -f "$OUT/best.pt" ]]; then echo "already done: $OUT"; exit 0; fi
rm -rf "$OUT"            # an interrupted earlier attempt is restarted from scratch

TEST_ARGS=()
if [[ -f "$EPIC_PROJECT/annotations/EPIC_100_test.csv" ]]; then
    TEST_ARGS=(--test-csv "$EPIC_PROJECT/annotations/EPIC_100_test.csv")
fi
python scripts/train_sta_memory.py \
    --features-dir "$EPIC_FEATURES" --manifest-dir "$EPIC_MANIFESTS" "${TEST_ARGS[@]}" \
    --pool-cache "$EPIC_TEAM/cache" --num-workers 6 \
    --mode "${MODES[$i]}" --steps "${STEPS[$i]}" --seed "$SEED" ${EXTRA[$i]} --out-dir "$OUT"
