#!/bin/bash -l
#SBATCH --job-name=epic-data-check
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --account=intro_vsc39484
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=/data/leuven/394/vsc39484/epic-kitchens/logs/data-check-%j.out
#SBATCH --error=/data/leuven/394/vsc39484/epic-kitchens/logs/data-check-%j.err

# Submit from the repository root. Each job writes to a fresh output directory.
# The account and log paths above belong to the current dataset owner.
set -euo pipefail
umask 077
export EPIC_PROJECT=/data/leuven/394/vsc39484/epic-kitchens
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens
export PIP_CACHE_DIR="$EPIC_PROJECT/cache/pip"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
module load Python/3.13.1-GCCcore-14.2.0
module load FFmpeg/7.1.1-GCCcore-14.2.0
DATA_ENV="$EPIC_PROJECT/envs/epic-data-20261009"
if [[ ! -x "$DATA_ENV/bin/python" ]]; then
    python -m venv "$DATA_ENV"
fi
source "$DATA_ENV/bin/activate"
# CPU-only PyTorch is sufficient for data preparation; GPU training comes later.
python -m pip install --disable-pip-version-check 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --disable-pip-version-check -r requirements-data.txt
RUN_ID="data-${SLURM_JOB_ID}"
python -m pip freeze > "$EPIC_PROJECT/logs/requirements-$RUN_ID.txt"
python -m pytest epic_starter/tests tests -q
HEALTH_REPORT="$EPIC_PROJECT/logs/video-health-$RUN_ID.json"
HEALTH_FAILED=0
if ! python scripts/probe_videos.py \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --annotation-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --out "$HEALTH_REPORT" --workers 4; then
    test -s "$HEALTH_REPORT" || exit 1
    HEALTH_FAILED=1
    echo "WARNING: unreadable videos found; manifests retain all events. Inspection uses readable videos only."
fi
python -m epic_starter.epic_pipeline prepare \
  --train-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --out-dir "$EPIC_PROJECT/manifests/$RUN_ID" --seed 42 --context 5
python scripts/check_data.py \
  --manifest-dir "$EPIC_PROJECT/manifests/$RUN_ID" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --out-dir "$EPIC_PROJECT/previews/$RUN_ID" \
  --annotation-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --legacy-manifest EPIC_100_train_manifest.csv \
  --health-report "$HEALTH_REPORT" \
  --samples 16 --workers 0 2 4 --batch-size 4
echo "DATA_PIPELINE_CHECK_COMPLETE"

if [[ "$HEALTH_FAILED" == "1" ]]; then
    echo "DATASET_HEALTH_BLOCKED: inspect $HEALTH_REPORT before full training."
    exit 2
fi
