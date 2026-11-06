#!/bin/bash -l
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_deeplearning_gt_2026b
#SBATCH --job-name=vjepa-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre1/scratch/394/vsc39450/RD-HCI/logs/%x-%j.out

set -euo pipefail

export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens
VJEPA_REPO=/lustre1/scratch/394/vsc39450/pretrained/vjepa2
CKPT=/lustre1/scratch/394/vsc39450/pretrained/vjepa2_1_vitl_384.pt
OUT_DIR="$EPIC_WORK/team/features/vjepa21-smoke-${SLURM_JOB_ID}"

cd /lustre1/scratch/394/vsc39450/RD-HCI
module load Python/3.13.5-GCCcore-14.3.0
source /data/leuven/394/vsc39450/venvs/procgen_rl/bin/activate

test -r "$EPIC_WORK/EPIC-KITCHENS/annotations/EPIC_100_train.csv"
test -d "$EPIC_WORK/EPIC-KITCHENS/videos_640x360"
test -r "$CKPT"
test -d "$VJEPA_REPO"
mkdir -p "$OUT_DIR"

python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python scripts/extract_vjepa21_tokens.py extract \
  --train-csv "$EPIC_WORK/EPIC-KITCHENS/annotations/EPIC_100_train.csv" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --vjepa-repo "$VJEPA_REPO" \
  --ckpt "$CKPT" \
  --out-dir "$OUT_DIR" \
  --img-size 384 \
  --batch-size 1 \
  --rank 0 \
  --world-size 1 \
  --max-videos 2 \
  --device cuda

python scripts/extract_vjepa21_tokens.py consolidate --out-dir "$OUT_DIR" --img-size 384
echo "VJEPA_SMOKE_COMPLETE: $OUT_DIR"