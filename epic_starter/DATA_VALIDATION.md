# Data-stage validation: 2026-10-09

## Code validation

- VSC job 62273213: **23 tests passed**, including real FFmpeg clip export and reuse.
- Original-video Dataset checks completed for train/validation and AR/STA with
  0, 2 and 4 workers: 12 configurations, 16 selected events per configuration.
- Contact sheets and timestamp JSON were generated. Empty-history STA samples
  remain present and marked for prior fallback.
- Full grouped split is retained: 55,136 training events / 396 videos and
  12,081 validation events / 99 videos. No video overlap.
- The old 1,105-row clip manifest has unique, recognized IDs and matching
  annotations. Its 1,105 referenced clip files are not present on this VSC
  checkout; that does not indicate whether they exist on the author's machine.

## Data blocker: do not start unqualified full-dataset training

All 700 files exist, but the header/first-frame probe found **20 unreadable
videos**, affecting **5,923 annotated events**. This was not detected by the
initial file-size verification. The remaining 680 passed header/first-frame
inspection; this does not certify every frame in every file.

Unreadable video IDs:

```text
P06_101
P09_103
P09_104
P22_102
P22_103
P22_104
P22_106
P22_107
P22_109
P22_110
P22_111
P22_113
P22_114
P22_115
P22_117
P25_104
P25_105
P25_106
P25_107
P28_103
```

`P22_110.MP4` was downloaded again from ManGO into a separate repair directory.
Its SHA256 is identical to the earlier download:

```text
e0dd46e55b3c43701fe8c5c1e3e65795ae146c567e1f8a31df836ed4226877fd
```

This confirms that redownloading that specific current source file does not fix
it. The other 19 failures were not individually redownloaded. Ask the course data
maintainer for corrected files or an agreed handling policy before full training.
No affected events, labels or original videos were removed or replaced.

The benchmark explicitly selected from readable videos for inspection only.
Manifests and priors still include all annotated events, and Dataset fails if an
unreadable file is requested. Job 62273213 intentionally exited **2** after
`DATA_PIPELINE_CHECK_COMPLETE`, with `DATASET_HEALTH_BLOCKED`. This is not a claim
that full-dataset validation passed.

## Small-sample loading measurements

| Split | Task | Workers | Events | Seconds |
|---|---|---:|---:|---:|
| train | ar | 0 | 16 | 5.98 |
| train | ar | 2 | 16 | 8.96 |
| train | ar | 4 | 16 | 11.60 |
| train | sta | 0 | 16 | 6.87 |
| train | sta | 2 | 16 | 8.61 |
| train | sta | 4 | 16 | 11.79 |
| val | ar | 0 | 16 | 5.53 |
| val | ar | 2 | 16 | 5.87 |
| val | ar | 4 | 16 | 8.88 |
| val | sta | 0 | 16 | 5.69 |
| val | sta | 2 | 16 | 7.07 |
| val | sta | 4 | 16 | 8.97 |

These measurements include worker startup. Later runs can benefit from filesystem
caching. Zero workers was fastest for these short checks; this is not a conclusion
about full-epoch training throughput. Start debugging with zero workers and
benchmark a longer run on the intended training hardware before increasing it.

## VSC artifacts

Under `/data/leuven/394/vsc39484/epic-kitchens`:

- `logs/data-check-62273213.out` and `.err`
- `logs/video-health-data-62273213.json` (includes affected event IDs)
- `manifests/data-62273213/` (canonical split and class statistics)
- `previews/data-62273213/report.json` and sample contact sheets
- `envs/epic-data-20261009/` (dedicated CPU data environment)

Tested package versions are recorded in `requirements-data-vsc.lock.txt` at the
repository root. The hidden course test CSV is still unavailable; no fake test
list or model predictions were created.
