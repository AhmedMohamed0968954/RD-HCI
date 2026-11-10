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
# More seeds: pass the seed as an argument (default 42), e.g. the decoupled rows with seed 1:
#   sbatch --array=4-7%3 -A "$EPIC_ACCOUNT" -o ... scripts/sta_memory_ablation.slurm 1
# (not SEED=1 sbatch ...: on wICE the job does not receive variables set on the command line,
# so every seed silently became 42). Then: python scripts/report_sta_memory.py --runs-dir ... --out-dir ...
set -euo pipefail
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '$SEED'"; exit 1; }
source scripts/sta_memory_env.sh
MODES=(single single single single decoupled decoupled decoupled decoupled decoupled decoupled decoupled)
STEPS=(4 8 16 30 4 8 16 30 16 16 16)
EXTRA=("" "" "" "" "" "" "" "" "--present-encoder mean" "--no-anchor" "--fusion three_way")
SUFFIX=("" "" "" "" "" "" "" "" "_meanpresent" "_noanchor" "_3way")
i=$SLURM_ARRAY_TASK_ID
NAME="${MODES[$i]}_S${STEPS[$i]}_seed${SEED}${SUFFIX[$i]}"
OUT="$EPIC_TEAM/results/sta_memory/$NAME"
echo "run $NAME"
# finished = the final metrics.json, which has a "best_val" entry (per-epoch files only have
# "best_val_action_mt5r", so the check must include the quote and colon)
if [[ -f "$OUT/metrics.json" ]] && grep -q '"best_val":' "$OUT/metrics.json"; then echo "already done: $OUT"; exit 0; fi
# a run written to in the last 20 minutes is probably still training in another job: leave it
if [[ -d "$OUT" && -n "$(find "$OUT" -type f -mmin -20 -print -quit)" ]]; then
    echo "$OUT was updated in the last 20 min (another job running it?); not touching it"; exit 1
fi
rm -rf "$OUT"            # an interrupted earlier attempt is restarted from scratch
INIT=("${STA_INIT_ARGS[@]}")
[[ "${EXTRA[$i]}" == "--present-encoder mean" ]] && INIT=()
python scripts/train_sta_memory.py \
    --features-dir "$EPIC_FEATURES" --manifest-dir "$EPIC_MANIFESTS" "${STA_TEST_ARGS[@]}" "${INIT[@]}" \
    --pool-cache "$EPIC_TEAM/cache" --num-workers 4 \
    --mode "${MODES[$i]}" --steps "${STEPS[$i]}" --seed "$SEED" ${EXTRA[$i]} --out-dir "$OUT"
