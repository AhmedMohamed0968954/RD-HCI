#!/bin/bash -l
#SBATCH --job-name=epic-preprocess-check
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --account=intro_vsc39484
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --export=NONE
# Working directory defaults to the directory where sbatch was called.
#SBATCH --output=/data/leuven/394/vsc39484/epic-kitchens/logs/preprocess-%j.out
#SBATCH --error=/data/leuven/394/vsc39484/epic-kitchens/logs/preprocess-%j.err
# Stop on errors instead of continuing with incomplete outputs.
set -euo pipefail
umask 077
export EPIC_PROJECT=/data/leuven/394/vsc39484/epic-kitchens
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens
export PIP_CACHE_DIR="$EPIC_PROJECT/cache/pip"
export RUN_ID="smoke-${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
module load Python/3.13.1-GCCcore-14.2.0
ENV_DIR="$EPIC_PROJECT/envs/epic-preprocess-20261009"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --disable-pip-version-check --only-binary=:all: 'av==19.0.1' 'numpy>=2.1,<3' 'pytest>=8,<10'
python -m pip freeze > "$EPIC_PROJECT/logs/requirements-$RUN_ID.txt"
python -m pytest tests -q
python epic_pipeline.py prepare \
 --train-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
 --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
 --out-dir "$EPIC_PROJECT/manifests/$RUN_ID" \
 --seed 42 --val-fraction 0.2 --context 5
python - <<'PY'
import json,os,subprocess,sys
from pathlib import Path
from epic_pipeline import load_events
root=Path(os.environ["EPIC_PROJECT"])
csv_path=root/"annotations/EPIC_100_train.csv"
events=load_events(csv_path)
first=next(e for e in events if e.sta_cutoff<0)
regular=next(e for e in events if e.video_id==first.video_id and e.start>6)
for event in [first,regular]:
    subprocess.run([sys.executable,"epic_pipeline.py","sample","--csv",str(csv_path),
        "--video-root",os.environ["EPIC_WORK"]+"/EPIC-KITCHENS/videos_640x360",
        "--narration-id",event.narration_id,"--out-dir",str(root/"previews"/os.environ["RUN_ID"])],check=True)
print("VSC_PREPROCESS_CHECK_COMPLETE")
PY
