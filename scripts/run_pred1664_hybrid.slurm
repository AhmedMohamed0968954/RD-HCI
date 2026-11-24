#!/bin/bash -l
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_edu_rdlab
#SBATCH --job-name=m2-pred1664-hyb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre1/scratch/394/vsc39450/RD-HCI/logs/%x-%j.out

set -euo pipefail

export VSC_DATA=/data/leuven/394/vsc39450
export XDG_CACHE_HOME=/data/leuven/394/vsc39450/.cache
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens

FEAT_DIR="$EPIC_WORK/team/features/vjepa21_vitl384_f16"
MANIFEST_DIR="manifests/hybrid-v3"
RESULTS_ROOT="$EPIC_WORK/team/results/member2_probe_hybrid"

module load Python/3.13.5-GCCcore-14.3.0
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python3.13
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python3
ln -sf "$(which python3.13)" /data/leuven/394/vsc39450/venvs/procgen_rl/bin/python

source /data/leuven/394/vsc39450/venvs/procgen_rl/bin/activate
cd /lustre1/scratch/394/vsc39450/RD-HCI

rm -rf /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda \
       /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda_bindings* \
       /data/leuven/394/vsc39450/venvs/procgen_rl/lib/python3.13/site-packages/cuda_pathfinder*

for SEED in 42 1 2; do
    for MODE in pred_17_only factorized_pred_1664; do
        echo "=== Running STA_${MODE} (hybrid-v3, seed=${SEED}) ==="
        python -X faulthandler scripts/train_probe.py \
          --manifest-dir "$MANIFEST_DIR" \
          --feature-dir "$FEAT_DIR" \
          --out-dir "$RESULTS_ROOT/STA_${MODE}_seed${SEED}" \
          --task STA --mode "$MODE" --epochs 20 --batch-size 256 --lr 3e-4 --num-workers 0 --seed "$SEED"
    done
done
