# Member 2: V-JEPA 2.1 Feature Bank & Factorized Spatiotemporal Attentive Probe

## 1. Consolidated Feature Bank (`$EPIC_WORK/team/features/vjepa21_vitl384_f16`)

All 700 videos (`67,217` annotated events and `360,451` 1.0s-strided memory bank clips) are extracted using frozen **V-JEPA 2.1 ViT-L/384** (`vjepa2_1_vitl_dist_vitG_384.pt`, 16 frames at 8 fps, aspect-preserving resize + bottom-centered 384x384 crop) and consolidated into memory-mapped `.npy` arrays:

| File | Shape & Dtype | Description |
|---|---|---|
| `ar_tokens.npy` | `[67217, 24, 1024] float16` | Action Recognition clips (`[start, stop)`) |
| `sta_obs_tokens.npy` | `[67217, 24, 1024] float16` | Exact STA clips ending at `sta_cutoff = start - 1.0s` |
| `sta_pred_tokens.npy` | `[67217, 24, 1024] float16` | Frozen V-JEPA 2.1 predictor tokens (`1.0s` ahead of `sta_cutoff`) |
| `bank_tokens.npy` | `[360451, 24, 1024] float16` | Dense `1.0s`-strided video memory bank across all 700 videos |
| `events_index.csv` | `67,217 rows` | `narration_id, row, video_id, action_start_s, action_stop_s, ar_clip_end_s, sta_clip_end_s, sta_use_prior` |
| `bank_index.csv` | `360,451 rows` | `video_id, t_end_s, row` |
| `meta.json` | JSON | Full provenance, preprocessing, token layout, and git commit |

### Compact 24-Token Layout (`[24, 1024]`)
- **Tokens `0..7` (`E_temp`):** 8 temporal tubelet spatial-mean tokens capturing frame-to-frame motion progression.
- **Tokens `8..23` (`E_spat`):** $4 \times 4$ adaptive spatial grid pooled over the last 2 temporal tubelets (final `0.5s`) capturing active hand-object interactions.

## 2. Full 67,217-Event Ablation Results (Seed 42 Split: 55,136 Train / 12,081 Val)

Checkpoints (`best.pt`), raw validation logits (`val_outputs.npz`), and metrics (`best_metrics.json`) are stored in `/lustre1/scratch/394/vsc39484/epic-kitchens/team/results/member2_probe_full/`.

| Task | Probe Mode | Verb Top-1 | Noun Top-1 | Action Top-1 (AR Primary) | Action MT5R (STA Primary) | Tail Top-1 | Tail MT5R |
|---|---|---|---|---|---|---|---|
| **AR** | `mean_pool` | 48.52% | 27.77% | 16.14% | 8.08% | 1.76% | 3.24% |
| **AR** | `full_attn` *(JFAA baseline)* | 64.66% | 43.65% | 32.60% | 23.28% | 7.09% | 14.37% |
| **AR** | **`factorized` *(Ours)*** | **65.47%** | **45.07%** | **34.58%** | **24.80%** | **7.67%** | **16.45%** |
| **STA** | `mean_pool` | 30.92% | 15.95% | 4.95% | 3.71% | 0.39% | 1.47% |
| **STA** | `full_attn` *(JFAA baseline)* | **33.29%** | 19.90% | 8.50% | 7.39% | 1.25% | **4.15%** |
| **STA** | **`factorized` *(Ours)*** | 32.82% | **20.97%** | **8.92%** | **7.63%** | **1.86%** | 3.67% |
| **STA** | `factorized_pred` *(Teacher slice)* | 32.73% | 18.74% | 8.16% | 7.23% | 1.61% | 3.98% |

## 3. Reproducing Extraction & Probe Training

```bash
# 1. Extract shards in parallel (8x B200 GPUs) & consolidate
sbatch scripts/extract_vjepa21_full.slurm
sbatch scripts/consolidate_vjepa21.slurm

# 2. Run full AR & STA probe ablations
sbatch scripts/run_probe_ablations_full.slurm

# 3. Generate attention routing figures
python scripts/plot_member2_probe.py
