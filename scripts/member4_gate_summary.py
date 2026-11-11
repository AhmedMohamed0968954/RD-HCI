"""Member 4: summarise the gate-branch ablation (scripts/member4_gate_ablation.slurm).

For every finished STA memory run whose gate setting is listed below, reads metrics.json and
val_outputs.npz and writes one Markdown table (mean +- std over seeds) plus the per-seed change
against "history + present" (same seed), and the mean gate weights per branch:
    python scripts/member4_gate_summary.py --runs-dir "$EPIC_TEAM/results/sta_memory" \
        --out "$EPIC_TEAM/results/member4/gate_ablation.md"
A JSON file with the same numbers is written next to the Markdown file.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_sta_memory import fusion_label, prefer_official, scores   # Member 3's report helpers

BRANCH_NAMES = ("history", "present", "future")
REFERENCE = "three_way H+P"
ORDER = ["aga", "three_way P", "three_way H+P", "three_way P+F (future: mean)",
         "three_way H+P+F (future: mean)", "three_way H+P+F (future: probe)"]
DESCRIPTION = {"aga": "AGA 2-way gate (Member 3's best row)",
               "three_way P": "present only (history off)",
               "three_way H+P": "history + present",
               "three_way P+F (future: mean)": "present + future (mean of predictor tokens)",
               "three_way H+P+F (future: mean)": "history + present + future (mean)",
               "three_way H+P+F (future: probe)": "history + present + future (Member 2's pred pathway, frozen)"}


def load(runs_dir: Path, steps: int, any_present: bool = False):
    runs = []
    for metrics in sorted(runs_dir.glob("*/metrics.json")):
        run = metrics.parent
        if not (run / "config.json").is_file():
            continue
        m, cfg = json.loads(metrics.read_text()), json.loads((run / "config.json").read_text())
        a = cfg["args"]
        if "best_val" not in m or a.get("mode") != "decoupled" or a.get("steps") != steps:
            continue
        same_present = a.get("present_encoder", "probe") == "probe" and a.get("freeze_present")
        if not any_present and (not same_present or a.get("no_anchor")):
            continue                      # same present encoder and clips as the gate rows only
        label = fusion_label(a)
        if label not in ORDER:
            continue
        row = {"run": run.name, "label": label, "seed": a["seed"], "best_epoch": m["best_epoch"],
               "seconds": round(sum(h.get("seconds", 0) for h in m.get("history", [])))}
        row.update(scores(m["best_val"], m.get("best_val_breakdown")))
        row["unseen"] = prefer_official(row, "unseen_action_mt5r")
        row["tail"] = prefer_official(row, "tail_action_mt5r")
        row.update(gate_weights(run / "val_outputs.npz"))
        if cfg.get("events_with_future"):
            row["val_events_with_future"] = cfg["events_with_future"].get("val")
        runs.append(row)
    return runs


def gate_weights(path: Path):
    """Mean three-way gate weight per branch over validation events that use the model (not the
    prior); for future-branch runs also over the events that have predictor tokens."""
    if not path.is_file():
        return {}
    z = np.load(path)
    if "verb_gates" not in z.files:
        return {}
    used = ~z["use_prior"].astype(bool)
    out = {}
    for field in ("verb", "noun"):
        g = z[f"{field}_gates"][used]
        out[f"{field}_gates"] = dict(zip(BRANCH_NAMES, np.round(g.mean(0), 3).tolist()))
        if "future_available" in z.files:
            has = z["future_available"].astype(bool)[used]
            if has.any():
                out[f"{field}_gates_with_future"] = dict(zip(BRANCH_NAMES, np.round(g[has].mean(0), 3).tolist()))
    return out


def mean_std(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None, None, 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def fmt(m, s, n):
    if m is None:
        return "–"
    return f"{m:.2f} ± {s:.2f}" if n > 1 else f"{m:.2f}"


def summarise(runs):
    by = defaultdict(list)
    for r in runs:
        by[r["label"]].append(r)
    ref = {r["seed"]: r for r in by.get(REFERENCE, [])}
    table = []
    for label in [x for x in ORDER if x in by]:
        rows = sorted(by[label], key=lambda r: r["seed"])
        entry = {"label": label, "description": DESCRIPTION[label], "seeds": [r["seed"] for r in rows],
                 "runs": [r["run"] for r in rows]}
        for key in ("action_mt5r", "verb_mt5r", "noun_mt5r", "unseen", "tail"):
            entry[key] = mean_std(r.get(key) for r in rows)
        deltas = [r["action_mt5r"] - ref[r["seed"]]["action_mt5r"] for r in rows if r["seed"] in ref]
        entry["delta_vs_HP"] = mean_std(deltas) if label != REFERENCE else (None, None, 0)
        entry["delta_vs_HP_per_seed"] = {r["seed"]: round(r["action_mt5r"] - ref[r["seed"]]["action_mt5r"], 2)
                                         for r in rows if r["seed"] in ref and label != REFERENCE}
        for field in ("verb", "noun"):
            for key in (f"{field}_gates", f"{field}_gates_with_future"):
                gs = [r[key] for r in rows if key in r]
                if gs:
                    entry[key] = {b: round(float(np.mean([g[b] for g in gs])), 3) for b in BRANCH_NAMES}
        entry["minutes"] = round(np.mean([r["seconds"] for r in rows]) / 60, 1)
        table.append(entry)
    return table


def markdown(table, runs_dir):
    lines = [f"Runs: `{runs_dir}` (decoupled memory, S = 16, frozen STA probe as present encoder; "
             "validation split_seed42; mean ± std over seeds).", "",
             "| Gate branches | Seeds | Action MT5R | Δ vs H+P (same seed) | Verb MT5R | Noun MT5R | "
             "Unseen action | Tail action |",
             "|---|---|---|---|---|---|---|---|"]
    for e in table:
        lines.append(f"| {e['description']} | {','.join(map(str, e['seeds']))} | {fmt(*e['action_mt5r'])} | "
                     f"{fmt(*e['delta_vs_HP'])} | {fmt(*e['verb_mt5r'])} | {fmt(*e['noun_mt5r'])} | "
                     f"{fmt(*e['unseen'])} | {fmt(*e['tail'])} |")
    lines += ["", "Mean gate weights on validation events (history / present / future):", "",
              "| Gate branches | Verb gate | Noun gate | Verb gate, events with predictor tokens | Noun gate, same |",
              "|---|---|---|---|---|"]
    show = lambda g: "–" if not g else " / ".join(f"{g[b]:.2f}" for b in BRANCH_NAMES)
    for e in table:
        if "verb_gates" in e:
            lines.append(f"| {e['description']} | {show(e.get('verb_gates'))} | {show(e.get('noun_gates'))} | "
                         f"{show(e.get('verb_gates_with_future'))} | {show(e.get('noun_gates_with_future'))} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out", required=True, help="Markdown file to write (JSON next to it)")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--any-present", action="store_true",
                   help="also keep runs with another present encoder (tests / exploratory runs)")
    a = p.parse_args(argv)
    runs = load(Path(a.runs_dir), a.steps, a.any_present)
    table = summarise(runs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = markdown(table, a.runs_dir)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps({"table": table, "runs": runs}, indent=2))
    print(text)
    return table


if __name__ == "__main__":
    main()
