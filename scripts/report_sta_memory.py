"""Collect STA memory runs into report material (Member 3, paper section on temporal memory).

    python scripts/report_sta_memory.py --runs-dir "$EPIC_TEAM/results/sta_memory" --out-dir report_sta

Writes to --out-dir:
  results.csv / results.md   one row per run: mode, S, seed, best epoch, Overall / Unseen / Tail scores
  horizon.png                validation Action MT5R vs history length S, single vs decoupled (mean ± std over seeds)
  <run>_attention.png        where the memory looks: mean attention per second before the cutoff
  <run>_gates.png            how much each stream relies on history vs the present (gate values)
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def load_runs(root: Path):
    runs = []
    for metrics in sorted(root.glob("*/metrics.json")):
        run = metrics.parent
        cfg_path = run / "config.json"
        if not cfg_path.is_file():
            continue
        m, cfg = json.loads(metrics.read_text()), json.loads(cfg_path.read_text())
        if "best_val" not in m:                       # still training or crashed before the end
            continue
        a = cfg["args"]
        row = {"run": run.name, "mode": a["mode"], "steps": a["steps"], "seed": a["seed"],
               "fusion": a.get("fusion", "aga"),
               "anchor": not a.get("no_anchor", False), "best_epoch": m["best_epoch"]}
        for subset, res in m["best_val"].items():
            prefix = {"all": "", "unseen_participants": "unseen_", "tail_actions": "tail_"}.get(subset, subset + "_")
            for key in ("action_mt5r", "verb_mt5r", "noun_mt5r", "action_top1"):
                row[prefix + key] = res.get(key)
        runs.append((row, run))
    return runs


def fmt(x):
    return "–" if x is None else f"{x:.2f}"


def write_tables(rows, out: Path):
    cols = ["run", "mode", "fusion", "steps", "seed", "anchor", "best_epoch", "action_mt5r", "verb_mt5r", "noun_mt5r",
            "unseen_action_mt5r", "tail_action_mt5r", "action_top1"]
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| Mode | Fusion | S | Seed | Action MT5R | Verb MT5R | Noun MT5R | Unseen action | Tail action |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["mode"], r["fusion"], r["steps"], r["seed"])):
        lines.append(f"| {r['mode']}{'' if r['anchor'] else ' (no anchor)'} | {r['fusion']} | {r['steps']} | {r['seed']} | "
                     f"{fmt(r.get('action_mt5r'))} | {fmt(r.get('verb_mt5r'))} | {fmt(r.get('noun_mt5r'))} | "
                     f"{fmt(r.get('unseen_action_mt5r'))} | {fmt(r.get('tail_action_mt5r'))} |")
    (out / "results.md").write_text("\n".join(lines) + "\n")


def plot_horizon(rows, out: Path, plt):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["anchor"] and r["fusion"] == "aga" and r.get("action_mt5r") is not None:
            groups[r["mode"]][r["steps"]].append(r["action_mt5r"])
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for mode, marker in (("single", "o"), ("decoupled", "s")):
        if mode not in groups:
            continue
        s = sorted(groups[mode])
        mean = [np.mean(groups[mode][k]) for k in s]
        std = [np.std(groups[mode][k]) for k in s]
        ax.errorbar(s, mean, yerr=std, marker=marker, capsize=3, label=f"{mode} AGA")
    ax.set_xlabel("History length S (seconds before the cutoff)")
    ax.set_ylabel("Validation action MT5R (%)")
    ax.set_xticks(sorted({k for g in groups.values() for k in g}))
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "horizon.png", dpi=200)
    plt.close(fig)


def plot_run(run: Path, out: Path, plt):
    path = run / "val_outputs.npz"
    if not path.is_file():
        return
    z = np.load(path, allow_pickle=False)
    cfg = json.loads((run / "config.json").read_text())["args"]
    keys = [k for k in ("attn", "attn_verb", "attn_noun") if k in z.files]
    if keys:
        valid = z["step_valid"][:, :-1].astype(float)            # keys = all steps before the last
        L = valid.shape[1]
        x = -np.arange(L, 0, -1) * cfg["stride"]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for k in keys:
            w = z[k]
            mean = (w * valid).sum(0) / np.maximum(valid.sum(0), 1)
            ax.plot(x, mean, marker=".", label={"attn": "shared memory", "attn_verb": "verb memory",
                                                "attn_noun": "noun memory"}[k])
        ax.axhline(1 / L, color="grey", lw=.8, ls="--", label="uniform")
        ax.set_xlabel("Memory step (seconds before the STA cutoff)")
        ax.set_ylabel("Mean attention weight")
        ax.grid(alpha=.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{run.name}_attention.png", dpi=200)
        plt.close(fig)
    gates = [k for k in ("gate", "gate_verb", "gate_noun") if k in z.files]
    if gates:
        keep = ~z["use_prior"].astype(bool)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bins = np.linspace(0, 1, 31)
        for k in gates:
            ax.hist(z[k][keep], bins=bins, alpha=.55, density=True,
                    label=f"{k.replace('_', ' ')} (mean {z[k][keep].mean():.2f})")
        ax.set_xlabel("Gate value (1 = relies on history, 0 = on the present clip)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{run.name}_gates.png", dpi=200)
        plt.close(fig)
    three = [k for k in ("verb_gates", "noun_gates") if k in z.files]
    if three:                                                  # Member 4's gate: [N,3] weights
        keep = ~z["use_prior"].astype(bool)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        x = np.arange(3)
        for j, k in enumerate(three):
            ax.bar(x + (j - .5) * .38, z[k][keep].mean(0), width=.38, label=k.split("_")[0])
        ax.set_xticks(x, ["history", "present", "future"])
        ax.set_ylabel("Mean gate weight")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{run.name}_three_way_gates.png", dpi=200)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(a.runs_dir))
    if not runs:
        raise SystemExit(f"No finished runs (metrics.json with best_val) under {a.runs_dir}")
    rows = [r for r, _ in runs]
    write_tables(rows, out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_horizon(rows, out, plt)
    for _, run in runs:
        plot_run(run, out, plt)
    print((out / "results.md").read_text())
    print("figures in", out)


if __name__ == "__main__":
    main()
