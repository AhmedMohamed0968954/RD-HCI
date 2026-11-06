# Member 4 VSC result - 2026-11-06

Execution commit: 8af084d. Slurm job: 62588399, COMPLETED, exit 0.
Account: intro_vsc39484; interactive partition; CPU only, 8 threads, 32 GB requested.
An isolated worktree was used, so switching the development branch does not affect the job.

## Completed

- Fixed synthetic-video uint8 overflow without wrapping the intensity sequence.
- All 32 tests passed (36.08 seconds in the final run).
- V-JEPA 2.1 distilled ViT-L/384 ran on 8 real course STA events.
- Explicit protocol: 16 frames, 8 fps, 384 x 384, full-frame square resize and ImageNet normalization.
- Encoder and predictor checkpoint keys all matched.
- All tested outputs were finite; helper and official wrapper output maximum absolute difference: 0.0.
- Latest decoded frame was 0.05557 seconds before the allowed cutoff in the closest case.
- Mean encoder-plus-predictor forward time: 6.023 seconds per clip on this CPU allocation.
  This excludes video decode, checkpoint loading and the extra equivalence forward pass;
  it is not end-to-end extraction throughput or a GPU performance estimate.
- Cross-event prediction cosine: 0.8144; mean standard deviation: 0.1619.
  Outputs varied across these 8 events; this small check does not establish general non-collapse or task accuracy.
- Gate demo finished 300 steps. Informative-branch weight approached 1.0;
  its weight was 0.0 when unavailable. Training accuracy reached 1.0 on synthetic inputs.
  This is a demonstration, not held-out accuracy or evidence of gains on EPIC-KITCHENS.

## Integration constraints

Encoder features: 1024 dimensions, 4608 tokens. Raw predictor: 1664 dimensions,
576 predicted tokens. The wrapper exposes a 1024-channel slice of the predictor;
that slice is still not the student's encoder feature space. Direct predictor
fidelity against student encoder tokens was therefore skipped. Future-branch utility
must be established through compatible-space analysis or downstream ablations.
The owner selected 2.1 only; no further 256 run is scheduled.

The execution used the diagnostic full-frame resize, while the blueprint's extraction
baseline uses the official evaluation transform. Choose and record the bank transform
before extraction; do not share caches across those preprocessing definitions.

Timestamp columns and the proposed 1.0-second bank protocol are documented in
MEMBER4_FORMAT_CONTRACT.md. The teammate's events_index.csv loader was not available
in this checkout and has not been modified or jointly validated.

## Artifacts

Root: /lustre1/scratch/394/vsc39484/epic-kitchens/team/results/member4/
- feasibility_vjepa21_vitl384_20261106-150716-62588399.json
- feasibility_vjepa21_vitl384_20261106-150716-62588399.log
- gate_demo_20261106-150716-62588399.txt

Job log: /data/leuven/394/vsc39484/epic-kitchens/logs/m4-feasibility-62588399.out
Pretrained checkpoint: team/checkpoints/pretrained/vjepa2_1_vitl_dist_vitG_384.pt
Upstream V-JEPA code commit: 204698b.

Earlier canceled runs are retained for traceability and must not be mistaken for
the final result. Changes stay on member4/predictor-gate; no push or merge was made.
