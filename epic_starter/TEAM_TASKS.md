# Suggested four-person ownership

The starter provides data preparation, legal frame sampling and local metric utilities. The team owns model selection, training, experiments and the course report.

| Member | First deliverable | Possible improvements |
|---|---|---|
| A: data | Dataset/DataLoader returning one working batch | Sampling, augmentation, caching and throughput |
| B: action recognition | AR encoder, verb/noun heads, small-sample training | Temporal pooling, fine-tuning, class imbalance |
| C: anticipation | STA model and small-sample training with legal history | Context length, temporal architecture, missing-history fallback |
| D: evaluation/integration | Shared evaluation and prediction entry points | Error analysis, official scorer alignment, calibration |

These are suggested roles, not implemented training modules. Each member should own a research question and review another member's changes.

## Shared interfaces

- Input batch: frames, valid_mask, use_prior, verb_label, noun_label, narration_id.
- Agree on tensor layout and normalization before connecting models.
- Model outputs: verb_logits [B,97], noun_logits [B,300]. Keep original class IDs.
- Keep all no-history STA events and apply a documented fallback using training information only.
- Freeze one validation split before comparing experiments; version any changed split.
- Two members should review STA timing boundaries and final submission coverage.

First milestone: A supplies a working batch, B/C each overfit a small sample, and D evaluates both through one interface. Architecture improvements and full experiments come afterwards.
