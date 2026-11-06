# Member 4 integration contract proposal

Backbone decision: the project owner selected V-JEPA 2.1 ViT-L/384 on 2026-11-06.
The timestamp-column and 1.0-second bank proposals still need adoption by the other members.

## Event timestamps

Use separate ar_clip_end_s and sta_clip_end_s columns in events_index.csv.
The former is the action stop; the latter is action start minus 1 second.
Both are seconds from the original video origin. Keep action_start_s and
 action_stop_s as source metadata. Do not silently reinterpret legacy clip_end_s;
loaders must know the task or reject ambiguous rows. Negative STA cutoffs represent
no available history, not permission to read a frame at zero.

## Memory bank

Start the shared extraction at bank_stride_s=1.0 as a compute baseline.
For each memory timestamp select the last bank end <= that timestamp. Retain
an exact event-specific STA clip at the cutoff. Earlier memory steps may be
less than 1 second stale; the exact final step does not remove that error.
A 0.5-second bank is a later accuracy/cost ablation, not proven unnecessary.

The approximate 18 GB (1 s) and 35 GB (0.5 s) estimates assume 100 video hours,
24 tokens x 1024 channels in float16 per bank clip. They exclude exact event
clips, predictor caches, indices and filesystem overhead. Measure actual video
durations before reserving final storage. These are pooled bank estimates,
not raw encoder token sizes.

## Backbone and preprocessing

Selected production backbone:
- Selected: V-JEPA 2.1 distilled ViT-L/384: vjepa2_1_vitl_dist_vitG_384.pt,
  configs/eval_2_1/vitl-384/ek100.yaml.

The first bank must use the selected 2.1 checkpoint at 384 x 384.
Do not run or extract the V-JEPA 2/256 model. The owner explicitly excluded it.
Never mix the two candidates in one bank or label a 384 checkpoint as 256.

The CPU feasibility run explicitly uses 16 frames at 8 fps (2 seconds), matching
the blueprint. Official eval YAML defaults use 32 frames; our override is deliberate.
The current preprocessing resizes the full frame to a square with ImageNet
normalization, rather than the official center crop. Record this difference.

meta.json must identify schema version, model family, exact checkpoint and hash,
upstream code commit, config, resolution, frames_per_clip, fps, clip_duration_s,
bank_stride_s, bank time origin, frame timestamp selection policy, resize/crop,
normalization, pooling rule, token count, feature dimension, dtype, and predictor
output space/dimension when cached. Record exact event-clip boundary conventions.
A metadata mismatch must trigger rejection or a separate bank directory.

The distilled predictor operates in the teacher feature space; a matching sliced
width does not prove alignment with the student encoder. Separate trainable
projections can fuse branches, but direct fidelity comparisons require compatible
feature spaces. All retained observed frames must remain at/before the STA cutoff.
