# Member 1: hybrid split, evaluation, calibration and leakage checks

Owner: Member 1 (data, calibration and evaluation). Everything here needs only
numpy and pandas. No videos, GPU, torch or PyAV are required, so it runs on a laptop.

| File | What it does |
|---|---|
| `epic_starter/data/hybrid_split.py` | Hybrid seen + unseen validation split, coverage and tail classes |
| `epic_starter/epic_metrics.py` | `evaluate_breakdown` (overall / seen / unseen / tail) and a fast, exact action top-5 |
| `epic_starter/calibration.py` | Logit adjustment, tau grid search, cold-start prior |
| `epic_starter/leakage.py` | Rejects manifests, feature indexes and history rows that pass the STA cutoff |
| `scripts/member1_demo.py` | One runnable demo of all of the above on synthetic scores |
| `tests/test_member1.py` | 13 unit tests |

## Quick start

```bash
python scripts/member1_demo.py --train-csv annotations/EPIC_100_train.csv --out-dir runs/member1_demo
python -m pytest -q tests/test_member1.py
```

## 1. The hybrid split (blueprint 3.2)

```bash
python -m epic_starter.epic_pipeline prepare-hybrid \
  --train-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv" \
  --test-csv  "$EPIC_PROJECT/annotations/EPIC_100_test.csv" \
  --out-dir   "$EPIC_PROJECT/manifests/hybrid-v1"
```

The output uses the same JSONL format as `prepare`, so the Dataset and
`read_manifest` accept it unchanged. Each row also carries
`subset` = `val_unseen`, `val_seen`, `train` or `test`. The command also writes:

- `split.csv`: one row per event (narration_id, participant_id, video_id, subset)
- `tail_classes.json`, `priors.json`, `cold_start_prior.json`: computed from **train only**
- `summary.json`: parameters, chosen IDs, sizes and coverage

**Rules:**

- **Unseen:** whole participants chosen from the middle 50% by size. The combination
  closest to 6% of events wins, preferring the one that covers the most classes.
  This is deterministic, and you can override it with `--unseen-participants P20,P29`.
- **Seen:** for every other participant, the latest videos by number
  (`PXX_1xx` were recorded after `PXX_xx`). Videos are added until about 8% of that
  participant's events are reached. A participant never gives more than 15%, and
  always keeps at least one training video. Only whole videos are moved.

**Result on EPIC_100_train.csv (defaults):**

| | events | share |
|---|---|---|
| train | 57,751 | 85.9% |
| val unseen (P08, P11, P29) | 4,699 | 7.0% |
| val seen (33 videos) | 4,767 | 7.1% |
| **val total** | **9,466** | **14.1%** |

**Coverage:** 86% of the training verb classes and 71% of the training noun classes
appear in val. Weighted by instances, that's 99.4% and 96.0%. The blueprint's "95% of
classes" goal is not reachable with a 14% split, because many nouns have fewer
than 10 examples in total. The instance-weighted figure is the honest one to report.

Twelve val classes (verbs 93, 95; ten nouns) never occur in train. No model can
get those right. Keep that in mind when reading tail MT5R.

P01 contributes no seen-val videos, because its last video (P01_109, about 1 hour) alone
exceeds 15% of its events.

## 2. Evaluation

```python
from epic_starter.epic_metrics import evaluate_breakdown, format_breakdown
r = evaluate_breakdown(verb_logits, noun_logits, verb_labels, noun_labels, task="STA",
                       unseen_mask=subset == "val_unseen",
                       tail_verbs=tail["verbs"], tail_nouns=tail["nouns"])
print(format_breakdown(r))
```

**Handoff contract:** a model returns `[N, 97]` verb scores and `[N, 300]` noun scores
(logits by default; pass `score_type="probabilities"` otherwise), in the same
order as the manifest rows.

- **Action:** the (verb, noun) pair, ranked by p_verb × p_noun.
- **Tail:** verb metrics on tail-verb events, noun metrics on tail-noun events,
  and action metrics on events where either one is tail. A tail class is outside the
  head that covers 80% of **training** instances.
- **MT5R:** averages Top-5 recall over the classes present in the evaluated subset.

`action_top5` searches the top-10 × top-10 pairs and not all 29,100 pairs. That is
exact, because a pair in the joint top 5 must have its verb and its noun in their
own top 5. A test checks it against brute force. It's about 300 times faster, which
makes tau grid searches practical.

Still to confirm against the course scorer on Codabench: the tail definition, and
whether MT5R is averaged over action classes or over verb and noun separately.

## 3. Calibration (blueprint 6.2)

```python
from epic_starter.calibration import class_log_prior, grid_search, adjust, split_halves
log_pi_v = class_log_prior(train_verbs, 97); log_pi_n = class_log_prior(train_nouns, 300)
half_a = split_halves(val_video_ids)          # tune on A, report on B
g = grid_search(zv, zn, yv, yn, log_pi_v, log_pi_n, "STA",
                taus_v=np.arange(0, 1.21, .1), taus_n=np.arange(0, 1.21, .1), mask=half_a)
zv_cal = adjust(zv, log_pi_v, g["tau_v"])
```

Tune tau on one half of the val videos and report on the other half. Tuning and
reporting on the same events makes the gain look larger than it is.

On the demo's synthetic head-biased scores, STA MT5R goes from 13.0 to 16.3 on the
held-out half. The AR Top-1 changes very little. That's expected, and matches the
blueprint's narrower AR range.

## 4. Cold-start prior (blueprint 6.3)

STA events with cutoff < 0 have no frames: 65 in train, 30 in val, 27 in test. The
prior is the verb/noun distribution of training actions that start in the first 5 s
(681 actions). `cold_start_logits(prior, log_pi_v, log_pi_n, tau_v, tau_n)` turns it
into submission scores.

**Caution:** with the STA tau (about 0.4–0.6), the `-tau·log(pi)` term pushes very rare
nouns (e.g. 157, 114, 235) into the cold-start top 5, because 681 events are too few
to outweigh it. Pick the cold-start tau separately on the 30 val cold-start events;
tau = 0 may well be best. 30 events is a small sample, so treat the result as rough.

## 5. Leakage checks

| Check | Where to call it |
|---|---|
| `check_manifest_rows(rows)` | after loading any STA manifest |
| `python -m epic_starter.epic_pipeline check-features --feature-dir <bank> --train-csv ... [--test-csv ...]` | after `extract_vjepa21_tokens.py consolidate`, before anyone trains on it |
| `history_rows(bank_video_df, cutoff, steps=16, spacing=1.0)` | Member 3's loader: picks the last bank clip ending ≤ each anchor, pads with -1 |
| `check_batch(clip_end_times, cutoffs)` | cheap guard inside the training loop |

`check-features` fails if any STA clip ends after its cutoff, if an event with a
negative cutoff is not marked `sta_use_prior`, if an event that has history is
marked as prior-only (this means frames are missing), or if any event has no features.
Use `--partial` for dummy banks.
