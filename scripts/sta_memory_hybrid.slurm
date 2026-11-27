#!/bin/bash -l
#SBATCH --job-name=sta-memory-hybrid
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00

# Member 3 on the team split (Member 1's hybrid-v3, seen + unseen participants), one job per seed.
# Memory seed s goes with Member 2's probe seed s (STA_factorized_seed<s>), as agreed with Member 4.
#   0  probe steps of STA_factorized_seed<s> (train/val + hidden-test STA clips), ~3 min, once
#   1  decoupled_S16_seed<s>_psteps_memonly          stage 1: memory only   <- Member 4's warm start
#   2  decoupled_S16_seed<s>_psteps_warm_frozenmem   stage 2: AGA gate, frozen memory (the main row)
#   3  decoupled_S16_seed<s>_psteps_3way_P           present-only control (same correction head, no history)
#   4  decoupled_S16_seed<s>_psteps                  AGA gate, memory from scratch (effect of pre-training)
#   5  decoupled_S16_seed<s>_psteps_fkeys_{memonly,warm_frozenmem}   plain-attention baseline (Spec 4)
# Rows 2-5 read the frozen probe's stored outputs for the present clip (--present-encoder cached, the same
# as running the frozen probe, see tests), so the job needs only the probe-steps folder: no 17 GB bank.
# The hidden-test predictions (test_outputs.npz) use the exact test STA clips (Member 2's *_test files).
#   for s in 42 1 2; do sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-hybrid-%j.out" scripts/sta_memory_hybrid.slurm $s; done
# Then: python scripts/sta_memory_summary.py --runs-dir "$EPIC_TEAM/results/sta_memory_hybrid-v3" (compute node).
# Paths (override with sbatch --export=ALL,NAME=value): STA_SPLIT (hybrid-v3), STA_MANIFESTS, STA_PROBE_DIR,
# STA_PROBE_STEPS_ROOT, STA_RESULTS. Defaults match scripts/member4_gate_fix.slurm.
set -euo pipefail
SEED="${1:-42}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer, got '$SEED'"; exit 1; }
source scripts/sta_memory_env.sh
SPLIT="${STA_SPLIT:-hybrid-v3}"
MANIFESTS="${STA_MANIFESTS:-$EPIC_TEAM/manifests/$SPLIT}"
[[ -f "$MANIFESTS/train_sta.jsonl" ]] || MANIFESTS="$PWD/manifests/$SPLIT"     # Member 1's copy in the repo
PROBE="${STA_PROBE_DIR:-$EPIC_TEAM/results/member2_probe_hybrid}/STA_factorized_seed${SEED}/best.pt"
STEPS="${STA_PROBE_STEPS_ROOT:-$EPIC_TEAM/cache/probe_steps}/${SPLIT}_seed${SEED}"
RESULTS="${STA_RESULTS:-$EPIC_TEAM/results/sta_memory_$SPLIT}"
for need in "$MANIFESTS/train_sta.jsonl" "$MANIFESTS/test_sta.jsonl" "$PROBE"; do
    [[ -r "$need" ]] || { echo "Missing $need"; exit 1; }
done
echo "split $SPLIT ($MANIFESTS), probe $PROBE, steps $STEPS, results $RESULTS"

echo "=== probe steps ($(date +%H:%M))"
python scripts/precompute_probe_steps.py --features-dir "$EPIC_FEATURES" --probe-ckpt "$PROBE" --out-dir "$STEPS"

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
    python scripts/train_sta_memory.py --manifest-dir "$MANIFESTS" --step-features "$STEPS" \
        --num-workers 4 --mode decoupled --steps 16 --seed "$SEED" "$@" --out-dir "$out"
}

B="decoupled_S16_seed${SEED}_psteps"
run "${B}_memonly" --present-encoder none
run "${B}_warm_frozenmem" --present-encoder cached --memory-init "$RESULTS/${B}_memonly/best.pt" --freeze-memory
run "${B}_3way_P" --present-encoder cached --fusion three_way --branches present
run "$B" --present-encoder cached
run "${B}_fkeys_memonly" --present-encoder none --step-keys features
run "${B}_fkeys_warm_frozenmem" --present-encoder cached --step-keys features \
    --memory-init "$RESULTS/${B}_fkeys_memonly/best.pt" --freeze-memory
echo "=== done ($(date +%H:%M))"
