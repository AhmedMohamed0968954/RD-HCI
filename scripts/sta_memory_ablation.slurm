#!/bin/bash -l
#SBATCH --job-name=sta-memory-ablation
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --array=0-10%3

# Member 3 ablation (all on top of Member 2's STA factorized probe as the present encoder,
# initialised from its checkpoint and frozen, so every row is "fixed probe + memory"):
#   0-3  single memory,    S = 4, 8, 16, 30 s
#   4-7  decoupled memory, S = 4, 8, 16, 30 s
#   8    decoupled S=16 with the memory's token mean as present (shows what the probe adds)
#   9    decoupled S=16 without the exact STA clip (matches hidden-test conditions)
#   10   decoupled S=16 with Member 4's three-way gate (history + present; future later)
# Run ONE normal job first (builds the pooled bank cache), then:
#   sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-ablation-%A_%a.out" scripts/sta_memory_ablation.slurm
# More seeds: SEED=1 sbatch ... . Then: python scripts/report_sta_memory.py --runs-dir ... --out-dir ...
set -euo pipefail
source scripts/sta_memory_env.sh
MODES=(single single single single decoupled decoupled decoupled decoupled decoupled decoupled decoupled)
STEPS=(4 8 16 30 4 8 16 30 16 16 16)
EXTRA=("" "" "" "" "" "" "" "" "--present-encoder mean" "--no-anchor" "--fusion three_way")
SUFFIX=("" "" "" "" "" "" "" "" "_meanpresent" "_noanchor" "_3way")
i=$SLURM_ARRAY_TASK_ID
SEED="${SEED:-42}"
NAME="${MODES[$i]}_S${STEPS[$i]}_seed${SEED}${SUFFIX[$i]}"
OUT="$EPIC_TEAM/results/sta_memory/$NAME"
if [[ -f "$OUT/metrics.json" ]] && grep -q best_val "$OUT/metrics.json"; then echo "already done: $OUT"; exit 0; fi
rm -rf "$OUT"            # an interrupted earlier attempt is restarted from scratch
INIT=("${STA_INIT_ARGS[@]}")
[[ "${EXTRA[$i]}" == "--present-encoder mean" ]] && INIT=()
python scripts/train_sta_memory.py \
    --features-dir "$EPIC_FEATURES" --manifest-dir "$EPIC_MANIFESTS" "${STA_TEST_ARGS[@]}" "${INIT[@]}" \
    --pool-cache "$EPIC_TEAM/cache" --num-workers 4 \
    --mode "${MODES[$i]}" --steps "${STEPS[$i]}" --seed "$SEED" ${EXTRA[$i]} --out-dir "$OUT"
