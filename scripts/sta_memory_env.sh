# Shared setup for the STA memory Slurm jobs (Member 3). Sourced, not run.
# Defaults point at the team folders; override any of them in ~/.bashrc.
export EPIC_PROJECT="${EPIC_PROJECT:-${VSC_DATA:?VSC_DATA is not set}/epic-kitchens}"
export EPIC_TEAM="${EPIC_TEAM:-/lustre1/scratch/394/vsc39484/epic-kitchens/team}"
export EPIC_FEATURES="${EPIC_FEATURES:-$EPIC_TEAM/features/vjepa21_vitl384_f16}"
export EPIC_MANIFESTS="${EPIC_MANIFESTS:-$EPIC_TEAM/manifests/split_seed42}"
export EPIC_PROBE_CKPT="${EPIC_PROBE_CKPT:-$EPIC_TEAM/results/member2_probe_full/STA_factorized/best.pt}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
for need in "$EPIC_FEATURES/bank_tokens.npy" "$EPIC_FEATURES/events_index.csv" "$EPIC_MANIFESTS/train_sta.jsonl"; do
    [[ -r "$need" ]] || { echo "Missing $need (check EPIC_FEATURES / EPIC_MANIFESTS)"; exit 1; }
done
if [[ -n "${EPIC_GPU_ENV:-}" ]]; then
    source "$EPIC_GPU_ENV/bin/activate"            # optional CUDA environment for GPU partitions
else
    module load Python/3.13.1-GCCcore-14.2.0       # the data environment from check_data.slurm (CPU)
    source "$EPIC_PROJECT/envs/epic-data-20261009/bin/activate"
fi
# CPU threads: the training script uses the cores Slurm gave this job (minus data workers).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
STA_TEST_ARGS=()
if [[ -f "$EPIC_PROJECT/annotations/EPIC_100_test.csv" ]]; then
    STA_TEST_ARGS=(--test-csv "$EPIC_PROJECT/annotations/EPIC_100_test.csv")
fi
# The probe starts from Member 2's checkpoint and stays FROZEN by default: training it further
# overfits (first full run: best 7.86 at epoch 0, then down to 6.8), while a frozen probe lets
# the memory improve on it. sbatch --export=ALL,EPIC_TRAIN_PRESENT=1 keeps it trainable.
STA_INIT_ARGS=()
if [[ -f "$EPIC_PROBE_CKPT" ]]; then
    STA_INIT_ARGS=(--present-init "$EPIC_PROBE_CKPT")
    [[ "${EPIC_TRAIN_PRESENT:-0}" == "1" ]] || STA_INIT_ARGS+=(--freeze-present)
else
    echo "NOTE: $EPIC_PROBE_CKPT not found; the present probe is trained from scratch."
fi
python -c "import torch, numpy; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
