"""Member 3: one row per configuration, mean ± sd over seeds, with the gate weights.

Groups the finished runs of a results folder by name without the seed (decoupled_S16_seed42_psteps_warm
and decoupled_S16_seed1_psteps_warm -> decoupled_S16_psteps_warm) and prints a markdown table:
action / verb / noun MT5R, tail action MT5R (Member 1's breakdown), and how much the gate gives the
history (AGA gate: mean sigmoid gate; three-way gate: history share), verb / noun.

    python scripts/sta_memory_summary.py --runs-dir "$EPIC_TEAM/results/sta_memory" [--match psteps]

On wICE run it on a compute node (the project environment does not run on the login nodes):
    srun --clusters=wice --partition=batch_icelake -A "$EPIC_ACCOUNT" --ntasks=1 --cpus-per-task=1 \
         --time=00:10:00 bash -lc 'source scripts/sta_memory_env.sh >/dev/null && python scripts/sta_memory_summary.py'
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
import numpy as np


def run_row(run: Path) -> dict | None:
    """Scores and gate weights of one finished run, or None."""
    try:
        m = json.loads((run / "metrics.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if "best_val" not in m:
        return None                                   # still running or interrupted
    allv = m["best_val"].get("all", {})
    bd = m.get("best_val_breakdown") or {}
    tail = bd.get("tail", {}).get("action_mt5r")
    seen, unseen = bd.get("seen", {}).get("action_mt5r"), bd.get("unseen", {}).get("action_mt5r")
    row = {"action": m["best_val_action_mt5r"], "verb": allv.get("verb_mt5r"),
           "noun": allv.get("noun_mt5r"), "tail": tail, "seen": seen, "unseen": unseen,
           "hist_v": None, "hist_n": None}
    npz = run / "val_outputs.npz"
    if npz.is_file():
        z = np.load(npz)
        keep = ~z["use_prior"].astype(bool) if "use_prior" in z.files else slice(None)
        if "gate_verb" in z.files:                    # AGA 2-way gate
            row["hist_v"], row["hist_n"] = float(z["gate_verb"][keep].mean()), float(z["gate_noun"][keep].mean())
        elif "verb_gates" in z.files:                 # Member 4's three-way gate: history share
            row["hist_v"] = float(z["verb_gates"][keep][:, 0].mean())
            row["hist_n"] = float(z["noun_gates"][keep][:, 0].mean())
    return row


def group_name(name: str) -> tuple[str, int | None]:
    m = re.search(r"_seed(\d+)", name)
    return (name[:m.start()] + name[m.end():], int(m.group(1))) if m else (name, None)


def fmt(values, sd=True):
    v = [x for x in values if x is not None]
    if not v:
        return "–"
    if len(v) == 1 or not sd:
        return f"{np.mean(v):.2f}"
    return f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=os.path.join(os.environ.get("EPIC_TEAM", "."), "results", "sta_memory"))
    p.add_argument("--match", help="only configurations whose name contains this text")
    p.add_argument("--out", help="also write the table to this markdown file")
    a = p.parse_args(argv)
    groups = defaultdict(dict)
    for run in sorted(Path(a.runs_dir).iterdir()):
        if not run.is_dir():
            continue
        g, seed = group_name(run.name)
        if a.match and a.match not in g:
            continue
        row = run_row(run)
        if row is not None:
            groups[g][seed] = row
    lines = ["| Configuration | Seeds | Action MT5R | Seen | Unseen | Verb MT5R | Noun MT5R | Tail action "
             "| History (verb / noun) |", "|---|---|---|---|---|---|---|---|---|"]
    for g, runs in sorted(groups.items()):
        rows = list(runs.values())
        col = lambda k: [r[k] for r in rows]
        seeds = ",".join(str(s) for s in sorted(runs, key=lambda s: (s is None, s)) if s is not None) or "–"
        hist = f"{fmt(col('hist_v'), sd=False)} / {fmt(col('hist_n'), sd=False)}" \
            if any(r["hist_v"] is not None for r in rows) else "–"
        lines.append(f"| {g} | {seeds} | {fmt(col('action'))} | {fmt(col('seen'))} | {fmt(col('unseen'))} | "
                     f"{fmt(col('verb'), sd=False)} | "
                     f"{fmt(col('noun'), sd=False)} | {fmt(col('tail'), sd=False)} | {hist} |")
    table = "\n".join(lines)
    print(table)
    if a.out:
        Path(a.out).write_text(table + "\n")
    return groups


if __name__ == "__main__":
    main()
