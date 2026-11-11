# Member 4: gate ablation results

Runs finished on 11 November 2026 (wICE CPU, 13-18 min each). Setup and commands:
[MEMBER4_PREDICTOR_GATE.md](MEMBER4_PREDICTOR_GATE.md#gate-ablation). Table from
`scripts/member4_gate_summary.py`; raw numbers in `$EPIC_TEAM/results/member4/gate_ablation.{md,json}`.

Every row: decoupled memory, S = 16, Member 2's STA probe frozen as the present encoder, validation
`split_seed42`, seeds 42, 1, 2 (mean ± std). Δ is against history + present **with the same seed**.

| Gate branches | Action MT5R | Δ vs H+P | Δ per seed (1 / 2 / 42) | Verb MT5R | Noun MT5R | Tail action |
|---|---|---|---|---|---|---|
| AGA 2-way gate (Member 3's best row) | 8.09 ± 0.13 | +0.09 ± 0.17 | -0.10 / +0.21 / +0.15 | 27.52 ± 1.40 | 25.35 ± 0.57 | 4.36 ± 0.09 |
| present only (history off) | 8.02 ± 0.08 | +0.02 ± 0.02 | +0.02 / +0.04 / -0.01 | 27.05 ± 1.24 | 24.94 ± 0.35 | 4.28 ± 0.23 |
| history + present | 8.00 ± 0.09 | – | – | 26.67 ± 1.09 | 24.80 ± 0.15 | 4.25 ± 0.23 |
| present + future (mean) | 7.97 ± 0.13 | -0.03 ± 0.04 | +0.01 / -0.07 / -0.02 | 26.82 ± 0.87 | 25.22 ± 0.52 | 4.32 ± 0.37 |
| history + present + future (mean) | 7.96 ± 0.04 | -0.04 ± 0.06 | -0.07 / +0.03 / -0.09 | 26.26 ± 1.38 | 24.76 ± 0.22 | 4.23 ± 0.16 |
| **history + present + future (probe, frozen)** | **8.20 ± 0.10** | **+0.20 ± 0.07** | **+0.13 / +0.22 / +0.27** | **32.23 ± 0.58** | **26.22 ± 0.48** | 4.45 ± 0.19 |

Mean gate weights on validation events (history / present / future; 346 of 348 validation events
have predictor tokens, so the "with tokens" columns are the same):

| Gate branches | Verb gate | Noun gate |
|---|---|---|
| present only | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 |
| history + present | 0.00 / 1.00 / 0.00 | 0.01 / 0.99 / 0.00 |
| present + future (mean) | 0.00 / 0.95 / 0.04 | 0.00 / 0.98 / 0.02 |
| history + present + future (mean) | 0.03 / 0.94 / 0.03 | 0.01 / 0.98 / 0.01 |
| history + present + future (probe) | 0.00 / 0.00 / 1.00 | 0.00 / 0.00 / 1.00 |

## What this shows

1. **The three-way gate does not use the history.** With history + present the gate puts ≥ 0.98 of
   its weight on the present branch, and switching history off changes the score by +0.02 ± 0.02.
   So within the three-way gate the memory adds nothing measurable; the AGA 2-way row (+0.09 ± 0.17)
   is within seed noise of it. This answers Member 3's "history switched off" control.
2. **The mean of the predictor tokens does not help** (-0.03 and -0.04, all within noise). The gate
   gives it 1-4 % of the weight: a linear projection of the averaged teacher-space tokens is not
   informative enough to compete with the trained present probe.
3. **Member 2's trained predictor pooling helps on every seed** (+0.13, +0.22, +0.27; 8.20 ± 0.10,
   the best row). The gate moves all its weight to the future branch, and the gain is mostly in
   verbs (32.2 vs 26.7 verb MT5R). The predicted latents are therefore useful when they are pooled
   with attention that was trained on them, not when they are averaged.

## Caveats

- **The gate collapses onto one branch** in every setting (present, or future with the probe), so
  "gating" here is effectively branch selection learned in the first 0-2 epochs. Gate entropy
  regularisation or branch dropout would be needed to get a real mixture.
- **Hidden test.** Predictor tokens exist only for the training events. In the probe row the gate
  never learned to rely on history + present, so falling back to them on test events without
  predictor tokens is untested and may be worse than the H+P model. Before using this row for a
  submission: extract predictor tokens for the test events, or train with random future dropout.
- The verb jump (+5.5 verb MT5R) matches Member 2's probes on their own
  (`member2_probe_full/*/best_metrics.json`): `factorized_pred` has verb 27.82 / noun 23.78 against
  23.82 / 25.50 for `factorized` (observed clip). The predicted latents carry more verb information
  and less noun information than the observed clip; the gated model keeps the verb gain without
  the noun loss. No leakage path is known: predictor inputs are the same clip ending at the cutoff
  as the present branch.
- Validation scores on our split (348 events), 3 seeds; differences below ~0.1 are noise.
