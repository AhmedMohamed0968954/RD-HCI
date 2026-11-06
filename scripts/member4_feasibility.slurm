#!/bin/bash -l
#SBATCH --job-name=m4-feasibility
#SBATCH --clusters=wice
#SBATCH --partition=batch_icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --export=NONE

# Member 4: tests, then the predictor feasibility check on CPU for two backbones.
# Submit from the repository root with your own account and log path:
#   sbatch -M wice -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/m4-feasibility-%j.out" scripts/member4_feasibility.slurm
# The vjepa2 clone and the pretrained checkpoints (about 5 GB each) go to the shared team folder,
# so the whole team uses one copy. Reports go to $EPIC_TEAM_DIR/results/member4/.
set -euo pipefail
export EPIC_PROJECT="${EPIC_PROJECT:-${VSC_DATA:?VSC_DATA is not set}/epic-kitchens}"
export EPIC_WORK="${EPIC_WORK:-${VSC_SCRATCH_LUSTRE1:-${VSC_SCRATCH:?VSC_SCRATCH is not set}}/epic-kitchens}"
TEAM="${EPIC_TEAM_DIR:-$EPIC_WORK/team}"
ANNOTATIONS="${EPIC_ANNOTATIONS:-$EPIC_PROJECT/annotations/EPIC_100_train.csv}"
VIDEOS="$EPIC_WORK/EPIC-KITCHENS/videos_640x360"
VJEPA="$TEAM/code/vjepa2"
CKPT="$TEAM/checkpoints/pretrained"
OUT="$TEAM/results/member4"
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# Official Meta checkpoints: torch >= 2.6 loads with weights_only=True by default, which can reject
# non-tensor entries saved in them; the vjepa2 loader does not pass weights_only itself.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

module load Python/3.13.1-GCCcore-14.2.0 FFmpeg/7.1.1-GCCcore-14.2.0
source "$EPIC_PROJECT/envs/epic-data-20261009/bin/activate"
# The V-JEPA model code needs timm and einops; keep the CPU build of torch (torchvision from the same index).
if ! python -c "import timm, einops, yaml" 2>/dev/null; then
    TORCH_VERSION=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
    pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu \
        "torch==$TORCH_VERSION" torchvision
    pip install --disable-pip-version-check timm einops pyyaml
fi

python -m pytest tests -q -p no:cacheprovider

[ -d "$VJEPA/.git" ] || git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "$VJEPA"
mkdir -p "$CKPT" "$OUT"
fetch() {  # download once, verify the size, rename only when complete
    local name=$1 url="https://dl.fbaipublicfiles.com/vjepa2/$1" expected
    expected=$(curl -sIL "$url" | awk 'tolower($1)=="content-length:" {v=$2} END {print v+0}')
    if [[ -s "$CKPT/$name" && $(stat -c %s "$CKPT/$name") == "$expected" ]]; then return 0; fi
    curl -sfL "$url" -o "$CKPT/$name.part"
    [[ $(stat -c %s "$CKPT/$name.part") == "$expected" ]] || { echo "size mismatch for $name"; exit 1; }
    mv "$CKPT/$name.part" "$CKPT/$name"
}
fetch vjepa2_1_vitl_dist_vitG_384.pt
fetch vitl.pt

STAMP=$(date +%Y%m%d-%H%M%S)
run() {  # run <tag> <config relative to the vjepa2 clone> <checkpoint>
    python scripts/predictor_feasibility.py --backend vjepa --vjepa-root "$VJEPA" \
        --config "$VJEPA/$2" --checkpoint "$CKPT/$3" --csv "$ANNOTATIONS" --video-root "$VIDEOS" \
        --num-events 8 --out "$OUT/feasibility_${1}_${STAMP}.json" > "$OUT/feasibility_${1}_${STAMP}.log" 2>&1 \
        || { echo "FAILED: $1 (see $OUT/feasibility_${1}_${STAMP}.log)"; tail -20 "$OUT/feasibility_${1}_${STAMP}.log"; }
}
run vjepa21_vitl384 configs/eval_2_1/vitl-384/ek100.yaml vjepa2_1_vitl_dist_vitG_384.pt
run vjepa2_vitl256 configs/eval/vitl/ek100.yaml vitl.pt
python scripts/gate_demo.py --steps 300 > "$OUT/gate_demo_${STAMP}.txt"
ls -la "$OUT"
echo "MEMBER4_FEASIBILITY_DONE"
