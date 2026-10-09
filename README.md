# RD-HCI: EPIC-KITCHENS AR and STA

Shared video-only data pipeline for the KU Leuven R&D course project.
AR reads the target action segment. STA reads only frames at least one second
before the action starts. Model training remains the next team task.

## Start here

Read [the data pipeline guide](docs/DATA_PIPELINE.md) for setup and commands,
[team tasks](docs/TEAM_TASKS.md) for ownership, and
[integration notes](docs/INTEGRATION.md) for the existing Dataset interface.

```text
RD-HCI/
  epic_starter/
    epic_pipeline.py         # prepare/sample CLI
    epic_metrics.py          # Metric helpers; course scorer alignment still required
    data/
      dataset.py            # Dataset and DataLoader
      events.py             # Annotation parsing and AR/STA windows
      manifests.py          # Shared split and metadata
      sampling.py           # Timestamp-based frame sampling
      __init__.py
  scripts/
    extract_clips.py         # Optional AR clip export
    probe_videos.py          # Header/first-frame health audit
    check_data.py            # Batch checks, contact sheets and worker measurements
    check_data.slurm         # Complete CPU data check on VSC
  tests/                    # All project tests
    test_epic.py            # Core pipeline and metric tests
    test_data_integration.py # Dataset/exporter integration tests
  docs/                     # Guides, validation evidence and team workflow
  requirements-data.txt
  requirements-data-vsc.lock.txt
```

Run commands from this repository root, inside the project environment:

```bash
python -m pytest tests -q
python -m epic_starter.epic_pipeline --help
python scripts/check_data.py --help
# On VSC, review the account and path settings first:
sbatch -M wice scripts/check_data.slurm
```

## Code and data locations

The working repository on the current VSC account is:
`/data/leuven/394/vsc39484/epic-kitchens/code/RD-HCI`.
Work here and push this repository; do not copy edits between two starter folders.

- Course annotations: `$EPIC_PROJECT/annotations/`
- Videos: `$EPIC_WORK/EPIC-KITCHENS/videos_640x360/`
- Generated manifests, logs and previews: outside this repository
- Temporary download/repair utilities: outside this repository

Here `EPIC_PROJECT=/data/leuven/394/vsc39484/epic-kitchens` and
`EPIC_WORK=/lustre1/scratch/394/vsc39484/epic-kitchens`.
Other team members must configure paths they can access.

Course CSVs and generated clip manifests are not part of the source distribution.
Obtain annotations from the course and generate the shared grouped split with
`prepare`. The old partial clip manifest is optional for diagnostic comparison.

## Validation status

The initial check found 20 unreadable videos affecting 5,923 course events.
Recovery is in progress; P22_110 passed complete decode and timing checks.
The remaining repair array and dependent full-pipeline check must finish before
claiming the data blocker resolved. See [validation details](docs/DATA_VALIDATION.md).
No course labels or split membership were changed.

See [Git workflow](docs/GITHUB.md) before contributing. Keep videos, annotations,
credentials, generated results and environments out of Git.
