"""Member 1 checkpoint demo: split -> evaluator -> calibration -> leakage checks.

Runs on a laptop in seconds with numpy + pandas only (no videos, GPU, torch or av).
Model scores are SYNTHETIC: a head-biased fake model, so the numbers show the
pipeline works, not how good any real model is.

    python scripts/member1_demo.py --train-csv annotations/EPIC_100_train.csv --out-dir runs/member1_demo
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.calibration import (adjust, class_log_prior, cold_start_logits, grid_search,
                                      split_halves)
from epic_starter.data.events import load_events
from epic_starter.data.hybrid_split import prepare_hybrid
from epic_starter.data.manifests import read_manifest
from epic_starter.epic_metrics import evaluate_breakdown, format_breakdown
from epic_starter.leakage import LeakageError, check_event_features, check_manifest_rows, history_rows


def fake_model(y, log_pi, rng, signal=2.5, head_bias=1.0, noise=1.5):
    """Logits that know the answer a bit but over-predict frequent classes."""
    z = rng.standard_normal((len(y), len(log_pi))) * noise + head_bias * log_pi[None, :]
    z[np.arange(len(y)), y] += signal
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    rng = np.random.default_rng(0)

    print("== 1. Hybrid split ==")
    summary = prepare_hybrid(args.train_csv, out / "split")
    print(json.dumps({k: summary[k] for k in ("unseen_participants", "events", "fractions")}, indent=1))

    train = read_manifest(out / "split" / "train_sta.jsonl")
    val = read_manifest(out / "split" / "val_sta.jsonl")
    check_manifest_rows(train + val)
    print("Manifest STA windows: OK (all end at start - 1 s)")

    tail = json.loads((out / "split" / "tail_classes.json").read_text())
    log_pi_v = class_log_prior([r["verb"] for r in train], 97)
    log_pi_n = class_log_prior([r["noun"] for r in train], 300)
    yv = np.array([r["verb"] for r in val])
    yn = np.array([r["noun"] for r in val])
    unseen = np.array([r["subset"] == "val_unseen" for r in val])
    zv, zn = fake_model(yv, log_pi_v, rng), fake_model(yn, log_pi_n, rng)

    print("\n== 2. Evaluator (synthetic STA scores, tau = 0) ==")
    base = evaluate_breakdown(zv, zn, yv, yn, "STA", unseen_mask=unseen,
                              tail_verbs=tail["verbs"], tail_nouns=tail["nouns"])
    print(format_breakdown(base))

    print("\n== 3. Logit adjustment: tune on half A of val videos, report on half B ==")
    half_a = split_halves([r["video_id"] for r in val])
    grid_ar = np.round(np.arange(-0.2, 0.31, 0.1), 2)
    grid_sta = np.round(np.arange(0.0, 1.21, 0.2), 2)
    results = {}
    for task, grid in (("AR", grid_ar), ("STA", grid_sta)):
        g = grid_search(zv, zn, yv, yn, log_pi_v, log_pi_n, task, grid, grid, mask=half_a)
        b = ~half_a
        tuned = evaluate_breakdown(adjust(zv[b], log_pi_v, g["tau_v"]), adjust(zn[b], log_pi_n, g["tau_n"]),
                                   yv[b], yn[b], task)["overall"]["primary_score"]
        plain = evaluate_breakdown(zv[b], zn[b], yv[b], yn[b], task)["overall"]["primary_score"]
        results[task] = g
        print(f"{task}: tau_v={g['tau_v']:+.1f} tau_n={g['tau_n']:+.1f}  "
              f"{g['metric']} on half B: {plain:.2f} -> {tuned:.2f}")
    (out / "calibration.json").write_text(json.dumps(results, indent=2))

    print("\n== 4. Cold-start prior (events with cutoff < 0) ==")
    prior = json.loads((out / "split" / "cold_start_prior.json").read_text())
    cv, cn = cold_start_logits(prior, log_pi_v, log_pi_n, results["STA"]["tau_v"], results["STA"]["tau_n"])
    print(f"built from {prior['events_used']} training actions starting in the first "
          f"{prior['window_seconds']:.0f} s; top verbs {np.argsort(-cv)[:5].tolist()}, "
          f"top nouns {np.argsort(-cn)[:5].tolist()}")

    print("\n== 5. Leakage checks on a mock feature index ==")
    events = load_events(Path(args.train_csv))
    index = pd.DataFrame({"narration_id": [e.narration_id for e in events],
                          "row": np.arange(len(events)),
                          "sta_clip_end_s": [e.sta_cutoff if e.sta_cutoff >= 0 else -1.0 for e in events],
                          "sta_use_prior": [e.sta_cutoff < 0 for e in events]})
    check_event_features(index, events)
    print(f"legal index ({len(index)} rows): accepted")
    leaky = index.copy()
    i = int(np.flatnonzero(leaky.sta_clip_end_s >= 0)[0])
    leaky.loc[i, "sta_clip_end_s"] += 0.5
    try:
        check_event_features(leaky, events)
        raise SystemExit("leaky index was NOT rejected")
    except LeakageError as err:
        print(f"leaky index (one clip 0.5 s past cutoff): rejected -> {err}")
    bank = pd.DataFrame({"t_end_s": np.arange(1.0, 60.0), "row": np.arange(59)})
    rows, ends, anchors = history_rows(bank, cutoff=20.5, steps=16)
    print(f"history for cutoff 20.5 s: clip ends {ends[-4:].tolist()} (all <= 20.5)")


if __name__ == "__main__":
    main()
