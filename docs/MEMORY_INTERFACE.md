# STA memory: interface and checkpoint (Member 3)

## Handoff formats this module expects

**Events** — `epic_starter.data.events.Event` (narration_id, participant_id, video_id,
start, stop, sta_cutoff = start − 1, verb, noun). Loaded from the course CSV or the
shared JSONL manifests; same split as everyone else.

**Feature bank (from Member 2)** — the team's consolidated layout, one folder per
feature version (`team/features/<version>/`). This module reads:

| file | content |
|---|---|
| `bank_tokens.npy` | `[M, T, D]` float16, dense memory bank for all videos (T = 24) |
| `bank_index.csv` | `video_id, t_end_s, row`; `t_end_s` = end time of the clip in that row |
| `sta_obs_tokens.npy` | `[N_events, T, D]`, exact clip ending at each event's STA cutoff |
| `events_index.csv` | `narration_id, row, video_id, ar_clip_end_s, sta_clip_end_s` (`sta_clip_end_s` required) |
| `meta.json` | backbone, resolution, frames/fps, crop, stride, git commit |

Arrays are opened with `mmap_mode="r"` inside each DataLoader worker, so a sample reads only
its own rows. `bank_index.csv` rows of a video may be in any order; the loader sorts them and
rejects duplicate end times. The STA clip is refused (error) if its end time is after the
cutoff, and a legacy ambiguous `clip_end_s` column is rejected (MEMBER4_FORMAT_CONTRACT.md). Rows cover all events; train/val membership comes from the manifests, not the files.

**Model output (to Member 1's evaluator)** — `verb_logits [B, 97]`, `noun_logits [B, 300]`,
accepted directly by `epic_metrics.evaluate(..., task="STA")`.

## What the loader guarantees

- The exact STA clip (`sta_obs_tokens.npy`) replaces the last step when available.
- Steps are `tau_k = cutoff − (S − k)·stride`, k = 1..S; the last step is the cutoff.
- Each step takes the latest clip ending **at or before** `tau_k` (snap down), so no step
  ever reads video after its own time or after the cutoff. `build_history` raises if it would.
- Steps with `tau_k < 0`, no clip yet, or a clip older than `tolerance` (default = stride)
  are masked (`valid = False`) and contribute nothing to the model.
- Events with no valid step get `use_prior = True` (cold start; Member 1's prior handles them).
- Auxiliary step labels: the first action starting in `[tau + 0.5, tau + 2.5]` s of the
  same video, from training events only; `-1` (ignored) otherwise.

## Model (`epic_starter/models/aga.py`)

AGA memory with `mode="single"` (keys/query over the 397-dim verb+noun prediction) or
`mode="decoupled"` (separate verb and noun memories). Predictions used as keys are the
model's own, detached, in training and testing. Hooks for teammates:

- `model.pool` — mean over tokens; Member 2's factorized probe replaces it (train with `--no-pool`).
- `model.encode(feats, valid)` — returns `verb_history`, `noun_history`, `present` (`[B, d]` each)
  and `history_available` / `present_available` (`[B]` bool): exactly the inputs of Member 4's
  `ThreeWayGate`.
- `GatedAGAModel(memory, GatedVerbNounHead(in_dim=d))` — memory + Member 4's gate. The future
  branch is zeros and marked unavailable until predictor features exist; then pass
  `future=(f_verb, f_noun)`. Train it with `--fusion three_way`.

## Team decisions this module follows

- Backbone V-JEPA 2.1 at 384 px; T = 24 compact tokens per clip; D is read from the arrays.
- Memory bank stride **1 s**, memory steps 1 s apart, exact STA clip for the last step.
- By default the bank is **token-averaged once** (`[M,24,D] -> [M,1,D]`, about 1 GB, kept in RAM
  and cached in `$EPIC_TEAM/cache/<version>/`). The current model averages tokens anyway, so
  results are identical and training does not reread ~25 GB from Lustre each epoch.
  Use `--no-pool` when a token-level probe replaces `model.pool`.
- Plain cross-entropy by default (`--class-weights none`), so Member 1's post-hoc logit
  adjustment stays valid (Menon et al. assume an unweighted loss).

## One-time setup (`~/.bashrc`)

```bash
export EPIC_TEAM=/lustre1/scratch/394/vsc39484/epic-kitchens/team
export EPIC_FEATURES=$EPIC_TEAM/features/<version>          # the folder Member 2 publishes
export EPIC_MANIFESTS=$EPIC_PROJECT/manifests/data-<jobid>  # the agreed split (train_sta.jsonl, val_sta.jsonl)
```

GPU environment: the course conda env `rd_lab` (`vsc_intro/install_libs.slurm`), or set
`EPIC_GPU_ENV` to a venv with CUDA PyTorch, NumPy and Matplotlib.

## Run

```bash
# unit tests (any node with the env)
python -m pytest tests/test_history.py tests/test_sta_training.py -q

# quick checkpoint demo on mock features (no GPU needed)
python scripts/demo_memory.py --train-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv"

# 1) one real run (also builds the pooled cache used by later runs)
sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-memory-%j.out" scripts/train_sta_memory.slurm \
       --mode single --steps 16
# 2) the ablation: single/decoupled x S in {4,8,16,30} + no exact STA clip + Member 4's 3-way gate
sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-ablation-%A_%a.out" scripts/sta_memory_ablation.slurm
SEED=1 sbatch ...   # optional extra seeds for error bars
# 3) report material
python scripts/report_sta_memory.py --runs-dir "$EPIC_TEAM/results/sta_memory" --out-dir report_sta
```

## Outputs per run (`$EPIC_TEAM/results/sta_memory/<run>/`)

| file | for whom | content |
|---|---|---|
| `metrics.json` | everyone | per-epoch validation scores (Overall, unseen participants, tail actions), best epoch |
| `config.json` | report | arguments, git commit, feature `meta.json`, split sizes, cold-start counts |
| `best.pt` | Member 4 | weights + constructor arguments of the best epoch |
| `val_outputs.npz` | Member 1 | raw verb/noun logits, labels, `use_prior`, training prior; tune (τ_v, τ_n) here |
| `test_outputs.npz` | Member 1 | raw hidden-test logits (calibrate, then package with AR) |
| `submission_sta.pt` | Member 1 | uncalibrated STA predictions in Codabench format, every test event exactly once |

Cold-start events (cutoff before the video start) get the training-prior scores. Tail actions
here are the action classes outside the head that covers 80% of training instances; Member 1's
evaluator remains the reference for the official tail/unseen definitions.

## Still needed from the team

- **Member 2** (`scripts/extract_vjepa21_tokens.py`, checked against this loader):
  - run the full extraction with `--test-csv EPIC_100_test.csv`, otherwise there are no STA
    features (and no bank rows for test videos) for the hidden test;
  - add `ar_clip_end_s` to `events_index.csv` and `bank_stride_s` to `meta.json` (format contract;
    `bank_stride_seconds` is also accepted here);
  - already compatible: `bank_index.csv` rows with `t_end_s = -1` (undecodable clip) and repeated
    end times are skipped and counted; events with `sta_use_prior = True` / `sta_clip_end_s = -1`
    never use their zero block as the present clip.
- **Member 1:** the final split manifests, and calibration of `val_outputs.npz` / `test_outputs.npz`.
- **Member 4:** future latents `f_t` projected to `[B, d]` per field, to pass as `future=` to `GatedAGAModel`.

With the mock bank the scores are near chance: the demo only proves the pipeline runs end to end.
