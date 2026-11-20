#!/bin/bash -l
#SBATCH --job-name=sta-memory-next
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --array=0-4

# Member 3: follow-ups on the best configuration of 15/11 (AGA gate, memory on Member 2's probe
# steps, pre-trained and frozen: decoupled_S16_seed<N>_psteps_warm_frozenmem, 8.42 ± 0.02).
# Each array task runs both pre-training stages (memory only, then frozen memory on the frozen
# probe) for one variant; run names end in _memonly and _warm_frozenmem.
#   0  S = 8    decoupled_S8_seed<N>_psteps_*         history length (S had no effect while the
#   1  S = 30   decoupled_S30_seed<N>_psteps_*        gate ignored the history; it uses it now)
#   2  S = 60   decoupled_S60_seed<N>_psteps_*
#   3  --no-anchor   decoupled_S16_seed<N>_psteps_noanchor_*   test-like: no exact STA clip, as on the
#                    hidden test set; its test predictions are the ones to submit
#   4  --step-keys features   decoupled_S16_seed<N>_psteps_fkeys_*   plain-attention baseline: keys
#                    from the probe's embeddings instead of its predictions, nothing else changed
# One job (5 tasks) per seed:
#   for s in 42 1 2; do sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-next-%A_%a.out" scripts/sta_memory_next.slurm $s; done
# Then: scripts/sta_memory_summary.py (header says how to run it on a compute node).
set -euo pipefail
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '$SEED'"; exit 1; }
TASK="${SLURM_ARRAY_TASK_ID:-0}"
source scripts/sta_memory_env.sh
RESULTS="${STA_RESULTS:-$EPIC_TEAM/results/sta_memory}"
[[ -f "$EPIC_PROBE_CKPT" ]] || { echo "Missing $EPIC_PROBE_CKPT (Member 2's STA probe)"; exit 1; }
[[ -f "$EPIC_PROBE_STEPS/meta.json" ]] || {
    echo "Missing $EPIC_PROBE_STEPS (run scripts/precompute_probe_steps.slurm first)"; exit 1; }

S=16; TAG=""; EXTRA=()
case "$TASK" in
    0) S=8 ;;
    1) S=30 ;;
    2) S=60 ;;
    3) TAG="_noanchor"; EXTRA=(--no-anchor) ;;
    4) TAG="_fkeys"; EXTRA=(--step-keys features) ;;
    *) echo "unknown array task $TASK"; exit 1 ;;
esac
BASE="decoupled_S${S}_seed${SEED}_psteps${TAG}"

run() {   # run NAME [train_sta_memory.py arguments...]; same restart rules as sta_memory_pretrain.slurm
    local name="$1"; shift
    local out="$RESULTS/$name"
    echo "=== run $name ($(date +%H:%M))"
    if [[ -f "$out/metrics.json" ]] && grep -q '"best_val":' "$out/metrics.json"; then
        echo "already done: $out"; return 0
    fi
    if [[ -d "$out" && -n "$(find "$out" -type f -mmin -20 -print -quit)" ]]; then
        echo "$out was updated in the last 20 min (another job running it?); stopping"; exit 1
    fi
    rm -rf "$out"            # an interrupted earlier attempt is restarted from scratch
    python scripts/train_sta_memory.py \
        --features-dir "$EPIC_FEATURES" --manifest-dir "$EPIC_MANIFESTS" "${STA_TEST_ARGS[@]}" \
        --pool-cache "$EPIC_TEAM/cache" --num-workers 4 --step-features "$EPIC_PROBE_STEPS" \
        --mode decoupled --steps "$S" --seed "$SEED" "${EXTRA[@]}" "$@" --out-dir "$out"
}

run "${BASE}_memonly" --present-encoder none
run "${BASE}_warm_frozenmem" --present-init "$EPIC_PROBE_CKPT" --freeze-present \
    --memory-init "$RESULTS/${BASE}_memonly/best.pt" --freeze-memory
echo "=== done ($(date +%H:%M))"
