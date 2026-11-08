#!/bin/bash -l
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_edu_rdlab
#SBATCH --job-name=vjepa-con
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre1/scratch/394/vsc39450/RD-HCI/logs/%x-%j.out

set -euo pipefail

export VSC_DATA=/data/leuven/394/vsc39450
export XDG_CACHE_HOME=/data/leuven/394/vsc39450/.cache
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens
OUT_DIR="$EPIC_WORK/team/features/vjepa21_vitl384_f16"

module load Python/3.13.5-GCCcore-14.3.0
source /data/leuven/394/vsc39450/venvs/procgen_rl/bin/activate
cd /lustre1/scratch/394/vsc39450/RD-HCI

SHARD_COUNT=$(ls -1 "$OUT_DIR/shards/"*.npz | wc -l)
echo "Found $SHARD_COUNT video shards in $OUT_DIR/shards/"

python -X faulthandler scripts/extract_vjepa21_tokens.py consolidate \
  --out-dir "$OUT_DIR" \
  --img-size 384

python -X faulthandler -c "
from epic_starter.models.factorized_probe import TokenBank
from epic_starter.memory.bank import FeatureBank, Anchors
tb = TokenBank('$OUT_DIR')
fb = FeatureBank(root='$OUT_DIR', pooled=False)
an = Anchors(root='$OUT_DIR', pooled=False)
print(f'CONSOLIDATION VERIFIED: {len(tb.row)} events, {len(fb.index)} videos in FeatureBank')
"
