# STA memory (Member 3): interface, model and how to run it

The STA model is **Member 2's factorized probe on the present clip + an AGA memory over the
last S seconds**. The probe (initialised from Member 2's STA checkpoint) gives the base
prediction; the memory retrieves relevant past seconds and adds a correction, so every result
is directly "probe alone" vs "probe + memory".

## Inputs this module reads

**Events** — `epic_starter.data.events.Event` (narration_id, participant_id, video_id,
start, stop, sta_cutoff = start − 1, verb, noun), from the shared manifests
(`$EPIC_TEAM/manifests/split_seed42`, the same split Member 2's probe used) or a CSV.

**Features (Member 2)** — `$EPIC_TEAM/features/vjepa21_vitl384_f16/` (V-JEPA 2.1 ViT-L/384,
24 tokens × 1024 per clip, bank stride 1 s):

| file | content |
|---|---|
| `bank_tokens.npy` + `bank_index.csv` | dense memory bank, all 700 videos (`video_id, t_end_s, row`) |
| `sta_obs_tokens.npy` + `events_index.csv` | exact clip ending at each event's STA cutoff (`sta_clip_end_s`, `sta_use_prior`) |
| `meta.json` | provenance; `bank_stride_s` is checked against the memory stride |

Checks on read: an STA clip ending after the cutoff raises an error; a legacy ambiguous
`clip_end_s` column is rejected (MEMBER4_FORMAT_CONTRACT.md); bank rows with `t_end_s = -1`
(undecodable) or a repeated end time (video end) are skipped and counted; events with
`sta_use_prior = True` never use their zero block.

**Probe checkpoint (Member 2)** — `$EPIC_TEAM/results/member2_probe_full/STA_factorized/best.pt`.

## What the loader guarantees (`epic_starter/memory/history.py`)

- Memory steps `tau_k = cutoff − (S − k)·stride`, k = 1..S; the last step is the cutoff.
- Each step takes the latest bank clip ending **at or before** `tau_k` (snap down), so no step
  reads video after its own time or the cutoff; `build_history` raises if it would.
- Steps before the video start, without a clip, or with a clip older than `tolerance`
  (= stride) are masked and contribute nothing. Events with no valid step get `use_prior`
  (training prior scores).
- **Present clip** (all 24 tokens, for the probe): the exact STA clip when it was extracted,
  otherwise the bank clip used for the last step (ends ≤ 1 s before the cutoff). Hidden-test
  events therefore always get a legal present clip, even without extracted STA clips.
- Auxiliary step labels: the first action starting in `[tau + 0.5, tau + 2.5]` s, training only.
- The bank is token-averaged once for the memory (`[M,24,D] → [M,1,D]`, ~0.7 GB in RAM, cached
  in `$EPIC_TEAM/cache/<version>/`, keyed by the source file so regenerated features are never
  mixed with an old cache). The present clip keeps all 24 tokens.

## Model (`epic_starter/models/aga.py`)

- Memory: AGA attention whose queries/keys are the model's own detached verb/noun predictions
  (EMA query, α = 0.8) and whose values are the past steps' features.
  `--mode single` (one memory over the 397-dim verb+noun prediction) or `decoupled`
  (separate verb and noun memories).
- Present: `--present-encoder probe` (default, Member 2's `FactorizedAttentiveProbe`, separate
  verb/noun embeddings) or `mean` (ablation: the memory's own last-step token mean).
- Fusion: per field a 2-way gate (history vs present), or Member 4's three-way gate
  (`--fusion three_way`; future branch unavailable until predictor features are wired in via
  `GatedAGAModel(..., future=(f_v, f_n))`).
- With the probe, output = probe logits + memory correction (zero-initialised): with
  `--present-init` the model starts exactly at Member 2's probe score. Every run is scored once
  before training (`init` in `metrics.json`); for these runs that is the probe alone in this
  same pipeline, and the report prints it as the baseline row. The Slurm jobs also **freeze**
  the probe (`--freeze-present`). Decoupled S = 16 on the full data (val action MT5R; probe
  alone 7.63):

  | probe | best | epochs after the best | early stop |
  |---|---|---|---|
  | trainable | 7.86 (epoch 0) | falls to 6.6–7.1 (overfits) | epoch 8 |
  | frozen | **8.20** (epoch 2) | stays at 7.75–8.04 | epoch 10 |

  `sbatch --export=ALL,EPIC_TRAIN_PRESENT=1 ...` keeps it trainable (wICE jobs do not receive
  variables set only on the command line, hence `--export`).
- `model.encode(...)` returns `verb_history`, `noun_history`, `verb_present`, `noun_present`
  (`[B, d]`), `history_available`, `present_available` — the inputs of Member 4's gate.
- Loss: cross-entropy with label smoothing 0.1 (as Member 2) + 0.5 × auxiliary step loss; no
  class reweighting by default, so Member 1's post-hoc logit adjustment stays valid.

## Member 1 integration (works before and after `member1-split-eval` is merged)

- **Split:** any manifest folder in the `prepare` / `prepare-hybrid` JSONL format. Today the
  default is `$EPIC_TEAM/manifests/split_seed42` (the split Member 2's probe used); when the team
  adopts the hybrid split, set `EPIC_MANIFESTS` to that folder (e.g. `.../manifests/hybrid-v1`).
- **Cold start:** events without frames get Member 1's `cold_start_prior.json` when the split
  folder has it (training actions in the first 5 s), else the training class prior. They are
  flagged `use_prior` in `val_outputs.npz` / `test_outputs.npz`, so Member 1 can recalibrate them.
- **Official numbers:** at the end of each run, `epic_metrics.evaluate_breakdown` (overall / seen /
  unseen / tail, tail classes from `tail_classes.json`) is printed and saved as
  `best_val_breakdown` in `metrics.json`; the report's Unseen / Tail columns use it. Per-epoch
  model selection uses Overall Action MT5R (same as Member 2).
- **Leakage cross-check:** a test checks that this loader and Member 1's `leakage.history_rows`
  pick exactly the same bank clips (skipped until Member 1's branch is merged).

## Run (CPU, the data environment from `check_data.slurm`; no GPU needed)

Paths default to the team folders (`scripts/sta_memory_env.sh`); nothing to set up. Submit
from the repository root:

```bash
# 1) one run (also builds the pooled bank cache, a few minutes the first time)
sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-memory-%j.out" scripts/train_sta_memory.slurm \
       --mode decoupled --steps 16
# 2) the ablation (11 runs, 3 at a time): see the header of the script for the rows
sbatch -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-ablation-%A_%a.out" scripts/sta_memory_ablation.slurm
sbatch --array=4-7%3 -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/sta-ablation-%A_%a.out" \
       scripts/sta_memory_ablation.slurm 1      # extra seed (argument, default 42) for error bars
# 3) report material (tables always; figures where matplotlib is installed)
python scripts/report_sta_memory.py --runs-dir "$EPIC_TEAM/results/sta_memory" --out-dir report_sta
```

Measured cost: data loading ~0.25 ms/event; with the probe ~1.4 s per batch of 256 on 2 cores,
i.e. about a minute per epoch on an 18-core Ice Lake node. Optional GPU: add
`-p gpu_p100 --gpus-per-node=1` to `sbatch` and `export EPIC_GPU_ENV=<CUDA venv>`.

## Outputs per run (`$EPIC_TEAM/results/sta_memory/<run>/`)

| file | for whom | content |
|---|---|---|
| `metrics.json` | everyone | per-epoch scores (all / `unseen` / `tail` as Member 2), best epoch, Member 1's official breakdown |
| `config.json` | report | arguments, git commit, feature `meta.json`, probe init, split sizes |
| `best.pt` | Member 4 | weights + constructor arguments of the best epoch |
| `val_outputs.npz` | Member 1 | raw verb/noun logits, labels, `use_prior`, prior, attention, gates, present source |
| `test_outputs.npz` | Member 1 | raw hidden-test logits (calibrate, then package with AR) |
| `submission_sta.pt` | Member 1 | uncalibrated STA predictions in Codabench format, every test event exactly once |

## Verification

- 28 unit tests (`tests/test_history.py`, `tests/test_sta_training.py`): no clip after the
  cutoff, masking, extractor edge rows, present-clip source, probe initialisation (model starts
  exactly at the probe's logits), frozen probe unchanged by training, full training runs,
  agreement with Member 1's `history_rows`. With Member 1's branch merged: 71 tests pass in total.
- Synthetic check with a signal placed **only 5–8 s before** each action (nothing in the
  present clip): S = 4 stays at chance (action MT5R 2.7%), S = 16 reaches 92.9%. The memory
  really retrieves information from the past, and the horizon matters.

## Still needed from the team

- **Member 2 — hidden-test STA clips.** The extraction has the 67,217 training events only
  (`EPIC_100_test.csv` was not in `$EPIC_WORK/EPIC-KITCHENS/annotations/`). The bank covers all
  700 videos, so test predictions work (present = last bank clip, ≤ 1 s before the cutoff), but
  training uses the exact clip. To match: copy `EPIC_100_test.csv` there, delete the shards of
  the videos that contain test events, rerun the extraction (other shards are skipped) and
  consolidate. Until then, the `_noanchor` ablation row is the model trained like the test.
- **Member 2 — provenance check:** confirm only the self-supervised encoder/predictor weights of
  `vjepa2_1_vitl_dist_vitG_384.pt` are loaded (the `ek100.yaml` config must not bring EK-100 probe weights).
- **Member 1:** merge `member1-split-eval`; calibrate `val_outputs.npz` / `test_outputs.npz`
  (tau on one half of the val videos, as in MEMBER1_SPLIT_EVAL.md) and package with AR.
- **Member 4:** future latents per field (`[B, d]`) for `GatedAGAModel(future=...)`.
