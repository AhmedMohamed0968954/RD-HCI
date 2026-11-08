#!/bin/bash -l
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_edu_rdlab
#SBATCH --job-name=m2-full-abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre1/scratch/394/vsc39450/RD-HCI/logs/%x-%j.out

set -euo pipefail

export VSC_DATA=/data/leuven/394/vsc39450
export XDG_CACHE_HOME=/data/leuven/394/vsc39450/.cache
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens

FEAT_DIR="$EPIC_WORK/team/features/vjepa21_vitl384_f16"
MANIFEST_DIR="$EPIC_WORK/team/manifests/split_seed42"
RESULTS_ROOT="$EPIC_WORK/team/results/member2_probe_full"

module load Python/3.13.5-GCCcore-14.3.0
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python3.13
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python3
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python

source /data/leuven/394/vsc39450/venvs/procgen_rl/bin/activate
cd /lustre1/scratch/394/vsc39450/RD-HCI

rm -rf /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda \
       /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda_bindings* \
       /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda_pathfinder*

echo "=== Running Full 67,217-Event Probe Ablations (AR & STA) ==="
for MODE in mean_pool full_attn factorized; do
    python -X faulthandler scripts/train_probe.py \
      --manifest-dir "$MANIFEST_DIR" \
      --feature-dir "$FEAT_DIR" \
      --out-dir "$RESULTS_ROOT/AR_${MODE}" \
      --task AR --mode "$MODE" --epochs 20 --batch-size 256 --lr 3e-4 --num-workers 0
done

for MODE in mean_pool full_attn factorized factorized_pred; do
    python -X faulthandler scripts/train_probe.py \
      --manifest-dir "$MANIFEST_DIR" \
      --feature-dir "$FEAT_DIR" \
      --out-dir "$RESULTS_ROOT/STA_${MODE}" \
      --task STA --mode "$MODE" --epochs 20 --batch-size 256 --lr 3e-4 --num-workers 0
done

echo "=== OFFICIAL FULL-DATASET SUMMARY TABLE FOR MID-TERM PRESENTATION (700 VIDEOS / 67,217 EVENTS) ==="
python -c "
import json
from pathlib import Path
root = Path('$RESULTS_ROOT')
print('| Task | Probe Mode | Verb Top-1 | Noun Top-1 | Action Top-1 | Action MT5R | Tail Top-1 | Tail MT5R |')
print('|---|---|---|---|---|---|---|---|')
for task, modes in [('AR', ['mean_pool', 'full_attn', 'factorized']), ('STA', ['mean_pool', 'full_attn', 'factorized', 'factorized_pred'])]:
    for mode in modes:
        m = json.loads((root / f'{task}_{mode}' / 'best_metrics.json').read_text())
        a, t = m['all'], m['tail']
        print(f'| {task} | {mode} | {a[\"verb_top1\"]:.2f}% | {a[\"noun_top1\"]:.2f}% | {a[\"action_top1\"]:.2f}% | {a[\"action_mt5r\"]:.2f}% | {t[\"action_top1\"]:.2f}% | {t[\"action_mt5r\"]:.2f}% |')
"
