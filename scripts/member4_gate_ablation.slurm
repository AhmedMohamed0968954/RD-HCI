#!/bin/bash -l
#SBATCH --job-name=m4-gate-ablation
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --array=0-4%2

# Member 4: which branches of the three-way gate help? Every row is Member 3's best setting
# (decoupled memory, S = 16, Member 2's STA probe as the frozen present encoder) with the
# three-way gate; only the branches the gate may use change:
#   0  present only                     history is computed but switched off: the control
#                                       "same model with the history switched off"
#   1  history + present                = Member 3's "_3way" row (skipped if it already exists)
#   2  present + future (mean)          future = mean of the 17 predictor-only tokens
#   3  history + present + future (mean)
#   4  history + present + future (probe) future = Member 2's frozen factorized_pred pathway
# Seed as argument (default 42); one array per seed:
#   sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/m4-gate-%A_%a.out" scripts/member4_gate_ablation.slurm 42
# Then: python scripts/member4_gate_summary.py --runs-dir "$EPIC_TEAM/results/sta_memory" ...
set -euo pipefail
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '$SEED'"; exit 1; }
source scripts/sta_memory_env.sh
FUTURE_CKPT="${EPIC_FUTURE_CKPT:-$EPIC_TEAM/results/member2_probe_full/STA_factorized_pred/best.pt}"
SUFFIX=(_3way_P _3way _3way_PF_mean _3way_HPF_mean _3way_HPF_probe)
EXTRA=("--branches present"
       ""
       "--future-encoder mean --branches present,future"
       "--future-encoder mean"
       "--future-encoder probe --future-init $FUTURE_CKPT --freeze-future")
i=$SLURM_ARRAY_TASK_ID
NAME="decoupled_S16_seed${SEED}${SUFFIX[$i]}"
OUT="$EPIC_TEAM/results/sta_memory/$NAME"
echo "run $NAME"
if [[ $i == 4 && ! -r "$FUTURE_CKPT" ]]; then echo "Missing $FUTURE_CKPT (Member 2's factorized_pred probe)"; exit 1; fi
# same restart rules as Member 3's sta_memory_ablation.slurm
if [[ -f "$OUT/metrics.json" ]] && grep -q '"best_val":' "$OUT/metrics.json"; then echo "already done: $OUT"; exit 0; fi
if [[ -d "$OUT" && -n "$(find "$OUT" -type f -mmin -20 -print -quit)" ]]; then
    echo "$OUT was updated in the last 20 min (another job running it?); not touching it"; exit 1
fi
rm -rf "$OUT"            # an interrupted earlier attempt is restarted from scratch
python scripts/train_sta_memory.py \
    --features-dir "$EPIC_FEATURES" --manifest-dir "$EPIC_MANIFESTS" "${STA_TEST_ARGS[@]}" "${STA_INIT_ARGS[@]}" \
    --pool-cache "$EPIC_TEAM/cache" --num-workers 4 \
    --mode decoupled --steps 16 --seed "$SEED" --fusion three_way ${EXTRA[$i]} --out-dir "$OUT"
