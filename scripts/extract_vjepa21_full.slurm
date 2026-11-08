#!/bin/bash -l
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_edu_rdlab
#SBATCH --job-name=vjepa-ext
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=/lustre1/scratch/394/vsc39450/RD-HCI/logs/%x-%A_%a.out

set -euo pipefail

export VSC_DATA=/data/leuven/394/vsc39450
export XDG_CACHE_HOME=/data/leuven/394/vsc39450/.cache
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens

VJEPA_REPO=/lustre1/scratch/394/vsc39450/pretrained/vjepa2
CKPT=/lustre1/scratch/394/vsc39450/pretrained/vjepa2_1_vitl_384.pt
OUT_DIR="$EPIC_WORK/team/features/vjepa21_vitl384_f16"

module load Python/3.13.5-GCCcore-14.3.0
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python3.13
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python3
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python

source /data/leuven/394/vsc39450/venvs/procgen_rl/bin/activate
cd /lustre1/scratch/394/vsc39450/RD-HCI

rm -rf /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda \
       /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda_bindings* \
       /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda_pathfinder*

TEST_ARGS=()
if [[ -f "$EPIC_WORK/EPIC-KITCHENS/annotations/EPIC_100_test.csv" ]]; then
    TEST_ARGS=(--test-csv "$EPIC_WORK/EPIC-KITCHENS/annotations/EPIC_100_test.csv")
fi

echo "=== [Rank ${SLURM_ARRAY_TASK_ID}/8] Resuming V-JEPA 2.1 extraction on $(hostname) ==="
python -X faulthandler scripts/extract_vjepa21_tokens.py extract \
  --train-csv "$EPIC_WORK/EPIC-KITCHENS/annotations/EPIC_100_train.csv" \
  "${TEST_ARGS[@]}" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --vjepa-repo "$VJEPA_REPO" \
  --ckpt "$CKPT" \
  --out-dir "$OUT_DIR" \
  --img-size 384 \
  --batch-size 32 \
  --rank "${SLURM_ARRAY_TASK_ID}" \
  --world-size 8 \
  --include-all-disk-videos \
  --device cuda

echo "=== [Rank ${SLURM_ARRAY_TASK_ID}/8] FINISHED ==="
