# Team ownership after the shared data stage

All four members can now review and extend one common pipeline.

| Member | Data-stage review | Next extension |
|---|---|---|
| A | Check annotations, fixed split and class counts in epic_starter/data/manifests.py | Sampling balance or alternative split experiments |
| B | Inspect AR/STA windows and contact sheets in epic_starter/data/sampling.py | AR model and controlled fine-tuning experiments |
| C | Review dataset.py batch contract, transforms and empty-history handling | STA model and history-length experiments |
| D | Reproduce scripts/check_data.py and inspect the benchmark report | Model evaluation, official scoring alignment and prediction integration |

Completed functionality: validated annotations, video index, reproducible grouped
split, class counts, training priors, timestamp-safe AR/STA sampling, Dataset,
DataLoader, legacy AR exporter compatibility, batch checks, contact sheets and
worker throughput measurement. Validation results are recorded separately.

Remaining team research: model choice, training, augmentation design, GPU
throughput experiments, official tail/unseen evaluation, ablations, hidden-test
inference and the report. The course hidden test CSV is still required.

Avoid four separate data pipelines. Review changes in small branches and retain
one agreed validation split. At least two members should review STA timing and
submission coverage.
