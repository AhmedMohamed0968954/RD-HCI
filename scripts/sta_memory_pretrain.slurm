#!/bin/bash -l
#SBATCH --job-name=sta-memory-pretrain
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=06:00:00

# Member 3: memory pre-training, both stages for one seed, one run after the other (each run is
# skipped when it already finished, so a resubmitted job continues where it stopped).
# Every run: decoupled memory, S = 16, same split, clips and seed as the earlier rows.
#   stage 1  decoupled_S16_seed<N>_memonly            memory only: predicts the STA action from the
#                                                      retrieved history (no probe, no gate)
#   stage 2  (Member 2's probe frozen as the present encoder, memory loaded from stage 1)
#            decoupled_S16_seed<N>_warm               AGA 2-way gate, memory fine-tuned
#                                                      compare: decoupled_S16_seed<N> (scratch)
#            decoupled_S16_seed<N>_warm_frozenmem     AGA 2-way gate, memory frozen
#            decoupled_S16_seed<N>_3way_warm          Member 4's three-way gate, history + present
#                                                      compare: _3way (scratch) and _3way_P (history off)
#            decoupled_S16_seed<N>_3way_HPF_probe_warm  three-way gate with Member 2's frozen future
#                                                      probe; compare: _3way_HPF_probe (scratch)
# Seed as argument (default 42); one job per seed:
#   for s in 42 1 2; do
#     sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-pretrain-%j.out" scripts/sta_memory_pretrain.slurm $s
#   done
# Second argument "probe": the same rows with Member 2's probe outputs as memory steps
# (--step-features "$EPIC_PROBE_STEPS", run names with _psteps), plus decoupled_S16_seed<N>_psteps
# (memory from scratch on probe steps). Needs scripts/precompute_probe_steps.slurm first:
#   J=$(sbatch --parsable -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/probe-steps-%j.out" scripts/precompute_probe_steps.slurm)
#   for s in 42 1 2; do sbatch --dependency=afterok:${J%%;*} -A "$EPIC_ACCOUNT" \
#     -o "$EPIC_PROJECT/logs/sta-pretrain-%j.out" scripts/sta_memory_pretrain.slurm $s probe; done
# Then: python scripts/member4_gate_summary.py --runs-dir "$EPIC_TEAM/results/sta_memory" --out ...
# Other split: sbatch --export=ALL,EPIC_MANIFESTS=<dir>,STA_RESULTS=<new results dir> ... (on wICE,
# variables set in front of sbatch are not passed on; --export=ALL,... is).
set -euo pipefail
SEED="${1:-42}"
STEPS="${2:-mean}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '$SEED'"; exit 1; }
[[ "$STEPS" == mean || "$STEPS" == probe ]] || { echo "second argument must be mean or probe, got '$STEPS'"; exit 1; }
source scripts/sta_memory_env.sh
RESULTS="${STA_RESULTS:-$EPIC_TEAM/results/sta_memory}"
FUTURE_CKPT="${EPIC_FUTURE_CKPT:-$EPIC_TEAM/results/member2_probe_full/STA_factorized_pred/best.pt}"
BASE="decoupled_S16_seed${SEED}"
if [[ ! -f "$EPIC_PROBE_CKPT" ]]; then echo "Missing $EPIC_PROBE_CKPT (Member 2's STA probe)"; exit 1; fi
STEP_ARGS=()
if [[ "$STEPS" == probe ]]; then
    [[ -f "$EPIC_PROBE_STEPS/meta.json" ]] || {
        echo "Missing $EPIC_PROBE_STEPS (run scripts/precompute_probe_steps.slurm first)"; exit 1; }
    STEP_ARGS=(--step-features "$EPIC_PROBE_STEPS")
    BASE="${BASE}_psteps"
fi

run() {   # run NAME [train_sta_memory.py arguments...]; same restart rules as sta_memory_ablation.slurm
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
        --pool-cache "$EPIC_TEAM/cache" --num-workers 4 \
        --mode decoupled --steps 16 --seed "$SEED" "${STEP_ARGS[@]}" "$@" --out-dir "$out"
}

if [[ "$STEPS" == probe ]]; then        # memory from scratch on probe steps: separates steps from pre-training
    run "$BASE" --present-init "$EPIC_PROBE_CKPT" --freeze-present
fi
run "${BASE}_memonly" --present-encoder none
MEMORY="$RESULTS/${BASE}_memonly/best.pt"
WARM=(--present-init "$EPIC_PROBE_CKPT" --freeze-present --memory-init "$MEMORY")
run "${BASE}_warm" "${WARM[@]}"
run "${BASE}_warm_frozenmem" "${WARM[@]}" --freeze-memory
run "${BASE}_3way_warm" "${WARM[@]}" --fusion three_way
if [[ -r "$FUTURE_CKPT" ]]; then
    run "${BASE}_3way_HPF_probe_warm" "${WARM[@]}" --fusion three_way \
        --future-encoder probe --future-init "$FUTURE_CKPT" --freeze-future
else
    echo "Missing $FUTURE_CKPT (Member 2's factorized_pred probe): skipping ${BASE}_3way_HPF_probe_warm"
fi
echo "=== done ($(date +%H:%M))"
