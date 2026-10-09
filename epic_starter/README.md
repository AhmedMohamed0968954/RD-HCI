# AR / STA data pipeline: step-by-step guide

The pipeline is a sequence of small steps, not a requirement to keep everything
in one Python file. The original Dataset and clip-export entry points remain at
the repository root. Shared rules are implemented once in `data/`.

## Read the code in this order

| Step | File | Responsibility |
|---|---|---|
| 1 | `epic_starter/data/events.py` | Parse CSV; represent one event; calculate AR/STA windows |
| 2 | `epic_starter/data/manifests.py` | Index videos; split by video; save manifests, statistics and priors |
| 3 | `epic_starter/data/sampling.py` | Decode original video and sample legal frames using PTS |
| 4 | `dataset.py` | Convert one event into a Tensor sample; create DataLoader batches |
| 5 | `scripts/check_data.py` | Check complete batches, export contact sheets, benchmark workers |
| Optional | `extract_clips.py` | Export standalone AR clips for inspection or a chosen AR cache |

`epic_pipeline.py` remains a small CLI and compatibility layer. Existing
`from epic_pipeline import sample_event` imports still work when run from its
folder; package users can import from `epic_starter.data.sampling`.

## Environment

Python 3.13 is used on VSC. Install an appropriate PyTorch build first (CPU works
for this entire data stage), then run:

```bash
python -m pip install -r requirements-data.txt
python -m pytest epic_starter/tests tests -q
```

The optional exporter also requires the `ffmpeg` executable. PyAV handles direct
video reading; torchvision is no longer required for the Dataset. Full clip
export is not required before training.

On the current VSC account, the complete check job creates a separate environment
at `$EPIC_PROJECT/envs/epic-data-20261009`, preserving the earlier environment.

## 1. Prepare one shared split

Run these commands from the repository root on VSC. Decoding and benchmarks belong
on allocated compute resources. Paths below refer to the current dataset owner;
other members need access to that storage or their own data paths.

```bash
export EPIC_PROJECT=/data/leuven/394/vsc39484/epic-kitchens
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
module load Python/3.13.1-GCCcore-14.2.0
source "$EPIC_PROJECT/envs/epic-data-20261009/bin/activate"
python -m epic_starter.epic_pipeline prepare \
  --train-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --out-dir "$EPIC_PROJECT/manifests/$RUN_ID" \
  --seed 42 --val-fraction 0.2 --context 5
```

Outputs: `train_ar.jsonl`, `val_ar.jsonl`, `train_sta.jsonl`, `val_sta.jsonl`,
`summary.json`, `class_counts.json`, `priors.json`. JSONL has one event per line.
The summary records the annotation SHA256 and exact video split; class statistics
retain all 97/300 class IDs, including zero counts. Priors use only training labels.
An existing output set is never overwritten. Freeze this split for team comparisons.

The course test CSV is still missing. When available, pass `--test-csv PATH` to
prepare unlabeled test manifests. No public test labels are used as a substitute.

## 2. Read samples and batches

```python
from dataset import EpicKitchensDataset, make_dataloader

# Use train_sta.jsonl for STA; the manifest carries its observation window.
dataset = EpicKitchensDataset(
    "/path/to/manifests/train_ar.jsonl",
    num_frames=16,
    video_root="/path/to/videos_640x360",
    image_size=224,
    normalize=False,
)
loader = make_dataloader(dataset, batch_size=4, num_workers=0, shuffle=True)
batch = next(iter(loader))
print(batch["frames"].shape)  # [B, T, 3, H, W]
```

Start with zero workers for clearer error messages. When using worker processes,
put script execution behind `if __name__ == "__main__":` because workers use spawn.
Use `shuffle=False` for validation/test. The final short batch is retained.

| Field | Meaning |
|---|---|
| `frames` | float32 [B,T,3,H,W], default RGB in [0,1] |
| `video` | Alias of frames, retained for existing teammate code |
| `verb_label`, `noun_label` | int64 [B], original class IDs; -1 for unlabeled events |
| `has_label` | Whether labels are available; never train using -1 as a real class |
| `narration_id`, `video_id`, `task` | Event identity and task |
| `valid_mask` | bool [B,T], marks real frames |
| `frame_times_seconds` | Original-video timestamps; NaN for empty history or legacy clips |
| `use_prior` | No usable STA history: bypass the visual model and apply a training-only fallback |
| `timing_verified` | True for direct original-video sampling; False for legacy clips |
| `observation_start/end`, `action_start/stop` | Boundaries for inspection |

`transform` keeps the original convention: it receives uint8 [T,C,H,W]. Return
uint8 or float in [0,1] at the configured size, without normalization. Optional
`normalize=True` then applies configurable mean/std once. Default mean/std are
ImageNet values, not a claim that every video backbone uses them. Set the values
required by the selected model. No random augmentation is applied automatically;
any spatial augmentation should remain consistent across frames in one clip.

The resize currently stretches to a square. Aspect-preserving crop/resize remains
an experimental choice for the team. Empty-history placeholders remain zero even
when normalization is enabled and are never passed through random transforms.

## 3. Probe video health first

```bash
python scripts/probe_videos.py \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --annotation-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --out "$EPIC_PROJECT/logs/health-$RUN_ID.json" --workers 4
```

This checks all headers and first frames, not every frame in every video. Failures
produce a report and nonzero exit code. They are never silently removed from
training. See [validation status](DATA_VALIDATION.md) for known source-file issues.

## 4. Inspect and benchmark

```bash
python scripts/check_data.py \
  --manifest-dir "$EPIC_PROJECT/manifests/$RUN_ID" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --out-dir "$EPIC_PROJECT/previews/$RUN_ID" \
  --annotation-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --legacy-manifest EPIC_100_train_manifest.csv \
  --health-report "$EPIC_PROJECT/logs/health-$RUN_ID.json" \
  --samples 16 --workers 0 2 4 --batch-size 4
```

Allocate at least five CPU cores for workers 0/2/4. The checker verifies split
isolation, AR/STA event alignment, labels, shapes, finite inputs, timestamp bounds,
empty-history handling and exact sample coverage. PNG contact sheets and JSON
metadata make the sampled frames reviewable. It audits the old partial clip CSV
without requiring those exported clips to be present on every machine.

Worker timings use the same events and include process startup and decoding.
Later passes may benefit from filesystem caches, so this small benchmark is a
smoke measurement, not proof of the best full-training configuration. Worker
count and prefetching are bounded; each decoder uses one thread.

For a single repeatable VSC job, run from the repository root:

```bash
sbatch -M wice scripts/check_data.slurm
```

If an existing OnDemand session triggers an interactive node limit, submit with
`sbatch -M wice --partition=batch_icelake scripts/check_data.slurm` instead.
This uses the configured compute account; it does not stop your interactive session.

The job requests five CPUs, 12 GB RAM and 30 minutes, installs CPU dependencies,
runs both test suites, generates a fresh split and checks real batches. Edit the
account/log paths for your VSC account. It does not train a model. If health checks found unreadable videos, it completes
inspection on the explicitly identified readable subset, then exits with code 2
and DATASET_HEALTH_BLOCKED. This is a data blocker, not a passing full-dataset
validation. The old
`epic_starter/check_on_vsc.slurm` remains a smaller preprocessing-only check.

## 5. Optional AR clip export

The original CLI is preserved:

```bash
python extract_clips.py \
  --csv_path "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --video_dir "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --output_dir "$EPIC_WORK/ar_preview_clips" \
  --output_csv "$EPIC_PROJECT/manifests/ar_preview_clips.csv" \
  --manifest_jsonl "$EPIC_PROJECT/manifests/$RUN_ID/train_ar.jsonl" \
  --num_workers 1 --limit 8
```

The exporter uses the shared AR window rule, writes temporary files before final
rename, and verifies an existing clip against a source/time sidecar before reuse.
Failed events remain in the output with an error and cause a nonzero exit status.
It never silently creates a smaller supposedly complete dataset.

Legacy usage still works: `EpicKitchensDataset("clip_manifest.csv", 16, transform)`.
Relative clip paths now resolve against the CSV directory, or an explicit
`clip_root`, rather than the process working directory. Output `video` is now
float32 [0,1] by default. Original source timestamps cannot be certified from
legacy re-encoded clips. STA explicitly rejects this mode; use original videos.

## Timing rules and remaining work

AR uses [start,stop). STA uses [max(0,start-1-context),start-1]. Real timestamps are
filtered before RGB conversion; neither target frames nor one-second-gap frames
may enter features. A decoder may internally decode outside the requested window;
these frames are not returned as input. Negative cutoffs return no history, while
cutoff zero can use the frame at time zero. Short windows repeat legal frames.

This completes a usable data-input foundation. It does not implement models,
training loops, model-specific augmentation, or full hidden-test inference.
`epic_metrics.py` remains available for future integration; its Cartesian product
action ranking and tail/unseen definitions still need alignment with the course
scorer. Keep model selection and experiments as team work.
