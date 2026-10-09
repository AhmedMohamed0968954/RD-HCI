# Integration notes for existing contributors

## What stayed familiar

- `dataset.py` is still the Dataset entry point.
- `EpicKitchensDataset(manifest_csv, num_frames=16, transform=None)` keeps the same first three arguments.
- `__len__`, `__getitem__` and `_sample_indices` remain recognizable.
- Samples still contain `video`, `verb_label`, `noun_label` and `narration_id`.
- `extract_clips.py`, `find_source_video`, `trim_segment` and the original command-line arguments remain.
- Existing `epic_pipeline.py prepare/sample` commands remain available.

## Deliberate changes

- `frames` is the canonical tensor key; `video` is a compatibility alias.
- Default Tensor output is float32 [0,1], not uint8. Remove any extra `/255` in caller code.
- PyAV replaces torchvision video decoding so original-video timestamps can be checked.
- Prepared JSONL preserves the common video split and task-specific window.
- Legacy clip mode supports AR only and reports timing_verified=False.
- Relative clip paths are resolved from the CSV directory (override with clip_root).
- Empty clips, missing sources and malformed labels fail visibly.
- Export failures remain in the CSV; existing clips require a matching sidecar for reuse.
- The optional exporter defaults to one worker and one FFmpeg encoding thread.

## Why small modules?

Annotation parsing, split creation, frame sampling and Dataset batching have
separate responsibilities. The Dataset calls the shared sampler rather than
copying its rules. Entry scripts remain short and sequential; teammates can edit
one stage without navigating a single large file or implementing timing twice.

## Files not rewritten

The original committed annotation CSV and partial clip manifest remain unchanged.
The downloader and authentication files stay outside the repository. No full clip
export or model training is triggered by this integration.
