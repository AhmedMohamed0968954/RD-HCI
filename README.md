# RD-HCI
This is the repository used by one of the teams within the project of HCI in the R&amp;D Course in KU Leuven

## Known data issue

The full video header/first-frame check found 20 unreadable files affecting 5,923
annotated events. Code tests pass, but full-dataset training needs corrected data.
See [validation details](epic_starter/DATA_VALIDATION.md). No events were removed.

## Runnable data pipeline

The shared data stage now supports both Action Recognition (AR) and Short-Term
Action Anticipation (STA). Read [the step-by-step guide](epic_starter/README.md).

- `dataset.py`: the existing `EpicKitchensDataset`, now connected to the shared sampler.
- `extract_clips.py`: the existing optional AR exporter; original arguments retained.
- `epic_starter/data/`: small modules for annotations, manifests and frame sampling.
- `scripts/check_data.py`: batch validation, sample images and worker benchmarks.

```bash
# Run from this repository root, inside the project's Python environment.
python -m pytest epic_starter/tests tests -q
python -m epic_starter.epic_pipeline --help
python scripts/check_data.py --help
```

On VSC, `sbatch -M wice scripts/check_data.slurm` prepares a CPU environment and
checks real data. Read its account/path settings before use.

The committed legacy `EPIC_100_train_manifest.csv` is a partial AR clip export,
not the canonical training split. Generate shared train/validation JSONL manifests
with `prepare`; do not divide classes or videos independently between team members.

No model architecture or training loop is included in this data-stage integration.
See [integration notes](epic_starter/INTEGRATION.md) for preserved interfaces and
[remaining team work](epic_starter/TEAM_TASKS.md).
