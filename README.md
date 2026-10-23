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

## First-time setup on VSC

Do steps 1–3 once. Then choose route A (one job does everything) or route B (by hand).

> Run project Python only on **compute nodes** (an OnDemand session, `srun` or `sbatch`).
> The login node you reach right after `ssh` uses an older CPU: the data environment
> crashes there with exit code 132 ("illegal instruction"). Editing files, `git`,
> `sbatch` and `squeue` on the login node are fine.

### 1. Personal settings

Add to `~/.bashrc` (values are personal; never commit them), then open a new shell:

```bash
export EPIC_PROJECT="$VSC_DATA/epic-kitchens"              # code, annotations, envs, manifests, logs
export EPIC_WORK="$VSC_SCRATCH_LUSTRE1/epic-kitchens"      # videos (large, not backed up)
export EPIC_ACCOUNT="<your Slurm compute account>"          # list yours with: sam-balance
```

### 2. Code

```bash
mkdir -p "$EPIC_PROJECT/code" && cd "$EPIC_PROJECT/code"
git clone https://github.com/AhmedMohamed0968954/RD-HCI.git
cd RD-HCI
```

Work in this clone and push from it; do not copy edits between code folders.

### 3. Data (required before any check can pass)

Access to GitHub does not grant access to another member's VSC storage. Each member
needs a readable copy of the course data in exactly this layout:

```text
$EPIC_PROJECT/annotations/EPIC_100_train.csv               # course annotations (from ManGO)
$EPIC_WORK/EPIC-KITCHENS/videos_640x360/P01/P01_01.MP4     # one folder per participant
```

Course CSVs and videos are never committed. If the videos come straight from ManGO,
20 of them are unreadable (see [validation details](docs/DATA_VALIDATION.md)); agree
within the team whether to use a shared, repaired copy before downloading 50 GB again.

### Route A — one job (recommended)

```bash
cd "$EPIC_PROJECT/code/RD-HCI"
mkdir -p "$EPIC_PROJECT/logs"
sbatch -M wice -A "$EPIC_ACCOUNT" -o "$EPIC_PROJECT/logs/data-check-%j.out" scripts/check_data.slurm
squeue -M wice -u "$USER"                                   # wait until it disappears
```

The job runs these steps in order:

1. Creates `$EPIC_PROJECT/envs/epic-data-20261009` on first use and installs CPU PyTorch
   plus `requirements-data.txt` (a few extra minutes the first time; reused afterwards).
2. Runs the tests (`python -m pytest tests -q`).
3. Checks that every video opens (`scripts/probe_videos.py`).
4. Writes the shared split to `$EPIC_PROJECT/manifests/data-<jobid>/`.
5. Loads real AR/STA batches and saves contact sheets to `$EPIC_PROJECT/previews/data-<jobid>/`.

Read the log `$EPIC_PROJECT/logs/data-check-<jobid>.out`:

| Last lines / exit code | Meaning |
|---|---|
| `DATA_PIPELINE_CHECK_COMPLETE`, exit 0 | Everything passed |
| `DATASET_HEALTH_BLOCKED`, exit 2 | Code works; some videos are unreadable (listed in `logs/video-health-data-<jobid>.json`) |
| Error before that, e.g. `Video root does not exist` | Data is missing or not in the layout of step 3 |

### Route B — by hand

From a compute-node shell, inside the clone:

```bash
module load Python/3.13.1-GCCcore-14.2.0 FFmpeg/7.1.1-GCCcore-14.2.0
python -m venv "$EPIC_PROJECT/envs/epic-data-20261009"      # first time only
source "$EPIC_PROJECT/envs/epic-data-20261009/bin/activate"
pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu   # first time only
pip install -r requirements-data.txt                        # first time only
python -m pytest tests -q                                   # expect: all passed
```

The tested package versions are in `requirements-data-vsc.lock.txt`. Then follow the
[data pipeline guide](docs/DATA_PIPELINE.md) step by step. GPU training will need a
separate environment with a CUDA build of PyTorch; keep this one for the data stage.

## Validation status

The initial check found 20 unreadable videos affecting 5,923 course events.
Recovery is in progress; P22_110 passed complete decode and timing checks.
The remaining repair array and dependent full-pipeline check must finish before
claiming the data blocker resolved. See [validation details](docs/DATA_VALIDATION.md).
No course labels or split membership were changed.

See [Git workflow](docs/GITHUB.md) before contributing. Keep videos, annotations,
credentials, generated results and environments out of Git.
