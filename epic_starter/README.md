# EPIC-KITCHENS AR / STA starter

A small, runnable video preprocessing and evaluation foundation for a four-person course project. Both tasks use video only. Model architectures, training loops and research experiments are left for the team.

## Start here

1. Read the `Event` structure and `observation()` in `epic_pipeline.py`.
2. Run the tests: `python -m pytest tests -q`.
3. Run `prepare` to create train/validation manifests.
4. Run `sample` to inspect AR and STA inputs from the same event.
5. Add a Dataset/DataLoader and your models using the interfaces below.

Install processing dependencies with `python -m pip install -r requirements.txt`. Python 3.13 was tested on VSC; exact tested versions are in `requirements-vsc.lock.txt`. PyTorch is optional for submission packaging and is not installed by these requirements.

## Files

| File | Purpose |
|---|---|
| `epic_pipeline.py` | Read annotations, split videos, compute observation windows, sample RGB frames |
| `epic_metrics.py` | Local Top-1/Top-5 and MT5R metrics; prediction validation and ZIP packaging |
| `tests/test_epic.py` | Time boundaries, variable frame rates, grouped splits and metric checks |
| `check_on_vsc.slurm` | Set up the processing environment and run a complete preprocessing smoke test |
| `TEAM_TASKS.md` | Suggested ownership and extension points |
| `GITHUB.md` | Upload workflow and file cleanup guidance |

## Run on VSC

The paths below match the current dataset owner. Other members must use directories they can access. Run decoding on allocated compute resources, not the login node.

```bash
export EPIC_PROJECT=/data/leuven/394/vsc39484/epic-kitchens
export EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens
module load Python/3.13.1-GCCcore-14.2.0
source "$EPIC_PROJECT/envs/epic-preprocess-20261009/bin/activate"
# Enter the directory containing epic_pipeline.py before running commands.
python -m pytest tests -q
```

Create a new output directory for each run; existing results are never overwritten:

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
python epic_pipeline.py prepare \
  --train-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --out-dir "$EPIC_PROJECT/manifests/$RUN_ID" \
  --seed 42 --val-fraction 0.2 --context 5
python epic_pipeline.py sample \
  --csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --video-root "$EPIC_WORK/EPIC-KITCHENS/videos_640x360" \
  --narration-id P01_01_10 --task both \
  --out-dir "$EPIC_PROJECT/previews/$RUN_ID"
```

The complete CPU smoke test can also be submitted with `sbatch -M wice check_on_vsc.slurm`. Check its account, working directory and log paths first: Slurm header values are specific to the current VSC user. Each job uses a fresh output directory. This is a preprocessing smoke test, not model training.

## Data and timing rules

- AR reads `[action_start, action_stop)`.
- STA reads `[max(0, start - 1 - context), start - 1]`. Default context: five seconds. For an action at 20 seconds, this is [14, 19]. Gap and target frames must never enter model inputs or extracted features.
- Sampling uses presentation timestamps (PTS), not frame numbers divided by average FPS. Decoders can internally return frames outside the interval; these are filtered before RGB conversion.
- Each video belongs to exactly one internal split. The default split holds out 20% of videos with seed 42; it does not hold out participants.
- Training priors use internal training labels only. Negative STA cutoffs produce placeholders with `use_prior=True`; use the saved priors rather than treating black placeholders as video. A cutoff of exactly zero permits a frame at time zero.
- Missing/corrupt videos are errors, not silently discarded samples. Short windows can repeat valid frames.
- The course hidden test CSV is still needed. Supply it using `prepare --test-csv PATH`; do not substitute public test labels.

## Interfaces for future training code

`prepare` writes `train_ar.jsonl`, `val_ar.jsonl`, `train_sta.jsonl`, `val_sta.jsonl`, `priors.json`, and `summary.json`. JSONL stores one event per line without copying videos.

```python
from pathlib import Path
from epic_pipeline import load_events, video_index, sample_event

events = load_events(Path("EPIC_100_train.csv"))
index = video_index(Path("/path/to/videos_640x360"))
clip = sample_event(events[0], index, task="STA", context=5, num_frames=8)
# clip["frames"]: uint8 [T, H, W, 3], RGB, default [8, 224, 224, 3].
# clip["valid_mask"]: boolean [T]; clip["frame_times_seconds"]: actual sampled times.
# clip["use_prior"]: True when no usable STA history exists.
```

The current resize is a simple square resize. Add model-specific normalization, tensor layout and augmentation in your Dataset. Suggested model outputs: `verb_logits [B,97]` and `noun_logits [B,300]`, retaining original class IDs.

## Metrics and submission utilities

```python
from epic_metrics import evaluate
scores = evaluate(verb_logits, noun_logits, verb_labels, noun_labels,
                  task="STA", score_type="logits")
print(scores["all"]["primary_score"])
```

Scores are percentages. AR primary score is action Top-1; STA primary score is action MT5R. MT5R averages Top-5 recall equally over classes present in the evaluated subset.

Local action scores use the product of verb/noun probabilities over all 97 x 300 pairs; ties favor smaller IDs. This convention must be checked against the course scorer. Supply official tail/unseen masks through `subsets`; their definitions are not guessed here.

`save_submission(ar_predictions, sta_predictions, test_ids, out_dir)` requires PyTorch and produces a ZIP containing `submission.pt` and `submission_sta.pt`. It checks exact ID coverage, finite floating scores and dimensions 97/300. It does not upload anything. Model training and full test inference remain team tasks.

## Validation

13 tests passed locally. On VSC, 12 passed and the PyTorch packaging test was skipped because PyTorch is not installed there. Real AR/STA video sampling also passed. This does not establish model accuracy.

References: [PyAV seeking](https://pyav.org/docs/stable/api/container.html), [presentation timestamps](https://pyav.org/docs/stable/api/frame.html).
