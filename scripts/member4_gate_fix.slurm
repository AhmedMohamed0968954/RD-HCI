#!/bin/bash -l
#SBATCH --job-name=m4-gate-fix
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --array=0-9%3

# Member 4: does the three-way gate stop collapsing onto one branch (Spec 5), and which of Member 2's
# predictor probes makes the best future branch (Spec 2)?
# Every row: Member 3's best memory (decoupled, S = 16, Member 2's probe outputs as steps, pre-trained
# without the probe and kept FROZEN), Member 2's STA factorized probe frozen as present encoder, the
# three-way gate. Memory seed s goes with Member 2's probe seed s (Member 3's convention).
#    future branch (Member 2's probe, frozen)       gate fixes
#   0  factorized_pred  (24 x 1024)                 none: the reference (gate expected on future only)
#   1  factorized_pred                              --gate-dropout $GATE_DROPOUT
#   2  factorized_pred                              --gate-entropy $GATE_ENTROPY
#   3  factorized_pred                              --gate-uncertainty
#   4  factorized_pred                              all three
#   5  none (history + present)                     all three: the fallback for test events without tokens
#   6  pred_17_only     (17 x 1664, strict future)  none
#   7  pred_17_only                                 all three
#   8  factorized_pred_1664 (17 x 1664 + 7 obs)     none
#   9  factorized_pred_1664                         all three
# Seed as argument; one array per seed:
#   for s in 42 1 2; do sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/m4-fix-%A_%a.out" scripts/member4_gate_fix.slurm $s; done
#
# M4_SPLIT (default hybrid-v3, the team split since 22/11; split_seed42 = the earlier rows, tasks 0-5 only).
# Paths below may contain {seed}; on wICE pass variables with --export=ALL,NAME=value:
#   M4_PROBE_STEPS  Member 3's probe-steps folder computed with STA_factorized_seed<s>
#                   (default $EPIC_TEAM/cache/probe_steps/hybrid-v3_seed{seed})
#   M4_RESULTS      results folder (default $EPIC_TEAM/results/sta_memory_hybrid-v3)
#   M4_MEMORY       Member 3's stage-1 memory (default $M4_RESULTS/decoupled_S16_seed{seed}_psteps_memonly/best.pt)
#   M4_PROBE_DIR    Member 2's checkpoints (default $EPIC_TEAM/results/member2_probe_hybrid; files
#                   STA_<mode>_seed{seed}/best.pt)
#   GATE_DROPOUT (0.3), GATE_ENTROPY (0.1)
# Then: python scripts/member4_gate_summary.py --runs-dir "$M4_RESULTS" --out "$EPIC_TEAM/results/member4/gate_fix.md" \
#   --reference "three_way H+P+F (future: probe) [memory probe steps, pre-trained (frozen)]"
set -euo pipefail
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '$SEED'"; exit 1; }
TASK="${SLURM_ARRAY_TASK_ID:-0}"
source scripts/sta_memory_env.sh
seeded() { echo "${1//\{seed\}/$SEED}"; }
S="{seed}"                                       # placeholder, filled in by seeded
SPLIT="${M4_SPLIT:-hybrid-v3}"
if [[ "$SPLIT" == split_seed42 ]]; then          # the earlier split and checkpoints (Member 3's defaults)
    MANIFESTS="$EPIC_MANIFESTS"
    RESULTS="${STA_RESULTS:-$EPIC_TEAM/results/sta_memory}"
    PROBE="$EPIC_PROBE_CKPT"
    STEPS="$EPIC_PROBE_STEPS"
    TEST_ARGS=("${STA_TEST_ARGS[@]}")
    future_ckpt() { [[ "$1" == factorized_pred ]] && echo "$EPIC_TEAM/results/member2_probe_full/STA_factorized_pred/best.pt" || echo ""; }
else
    MANIFESTS="$EPIC_TEAM/manifests/$SPLIT"
    RESULTS="$(seeded "${M4_RESULTS:-$EPIC_TEAM/results/sta_memory_$SPLIT}")"
    PROBE_DIR="$(seeded "${M4_PROBE_DIR:-$EPIC_TEAM/results/member2_probe_hybrid}")"
    PROBE="$PROBE_DIR/STA_factorized_seed${SEED}/best.pt"
    STEPS="$(seeded "${M4_PROBE_STEPS:-$EPIC_TEAM/cache/probe_steps/${SPLIT}_seed${S}}")"
    future_ckpt() { echo "$PROBE_DIR/STA_$1_seed${SEED}/best.pt"; }
    TEST_ARGS=()                                # the split's own test_sta.jsonl (hidden test)
fi
MEMORY="$(seeded "${M4_MEMORY:-$RESULTS/decoupled_S16_seed${S}_psteps_memonly/best.pt}")"
GATE_DROPOUT="${GATE_DROPOUT:-0.3}"
GATE_ENTROPY="${GATE_ENTROPY:-0.1}"
FIX=(--gate-dropout "$GATE_DROPOUT" --gate-entropy "$GATE_ENTROPY" --gate-uncertainty)
BASE="decoupled_S16_seed${SEED}_psteps"
MODE=factorized_pred; TAG=HPF_probe
case "$TASK" in
    0) GATE=(); SUFFIX="" ;;
    1) GATE=(--gate-dropout "$GATE_DROPOUT"); SUFFIX=_gdrop ;;
    2) GATE=(--gate-entropy "$GATE_ENTROPY"); SUFFIX=_gent ;;
    3) GATE=(--gate-uncertainty); SUFFIX=_gunc ;;
    4) GATE=("${FIX[@]}"); SUFFIX=_gfix ;;
    5) GATE=("${FIX[@]}"); SUFFIX=_gfix; MODE=""; TAG="" ;;
    6) GATE=(); SUFFIX=""; MODE=pred_17_only; TAG=HPF_p17 ;;
    7) GATE=("${FIX[@]}"); SUFFIX=_gfix; MODE=pred_17_only; TAG=HPF_p17 ;;
    8) GATE=(); SUFFIX=""; MODE=factorized_pred_1664; TAG=HPF_p1664 ;;
    9) GATE=("${FIX[@]}"); SUFFIX=_gfix; MODE=factorized_pred_1664; TAG=HPF_p1664 ;;
    *) echo "unknown array task $TASK"; exit 1 ;;
esac
FUTURE=()
if [[ -n "$MODE" ]]; then
    CKPT="$(future_ckpt "$MODE")"
    [[ -n "$CKPT" && -r "$CKPT" ]] || { echo "No $MODE checkpoint for split $SPLIT seed $SEED (${CKPT:-none}); skipping task $TASK"; exit 0; }
    FUTURE=(--future-encoder probe --future-init "$CKPT" --freeze-future)
fi
for need in "$MANIFESTS/train_sta.jsonl" "$PROBE" "$STEPS/meta.json" "$MEMORY"; do
    [[ -r "$need" ]] || { echo "Missing $need (Member 3's probe steps / stage-1 memory for this split and seed?)"; exit 1; }
done
NAME="${BASE}_3way${TAG:+_$TAG}_warm_frozenmem${SUFFIX}"
OUT="$RESULTS/$NAME"
echo "=== run $NAME (split $SPLIT, $(date +%H:%M))"
# same restart rules as Member 3's sta_memory_pretrain.slurm
if [[ -f "$OUT/metrics.json" ]] && grep -q '"best_val":' "$OUT/metrics.json"; then echo "already done: $OUT"; exit 0; fi
if [[ -d "$OUT" && -n "$(find "$OUT" -type f -mmin -20 -print -quit)" ]]; then
    echo "$OUT was updated in the last 20 min (another job running it?); not touching it"; exit 1
fi
rm -rf "$OUT"            # an interrupted earlier attempt is restarted from scratch
python scripts/train_sta_memory.py \
    --features-dir "$EPIC_FEATURES" --manifest-dir "$MANIFESTS" "${TEST_ARGS[@]}" \
    --pool-cache "$EPIC_TEAM/cache" --num-workers 4 --step-features "$STEPS" \
    --mode decoupled --steps 16 --seed "$SEED" \
    --present-init "$PROBE" --freeze-present --memory-init "$MEMORY" --freeze-memory \
    --fusion three_way "${FUTURE[@]}" "${GATE[@]}" --out-dir "$OUT"
echo "=== done ($(date +%H:%M))"
