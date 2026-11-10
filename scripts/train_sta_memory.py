"""Train and evaluate the STA memory model (Member 3): AGA memory over V-JEPA features.

One run = one configuration (mode, history length S, seed). It
  1. reads the team split (manifest JSONL from `epic_pipeline prepare`) or a CSV + grouped split,
  2. builds S one-second history steps per event from the consolidated feature folder,
  3. trains AGAMemoryModel with early stopping on validation Action MT5R,
  4. writes everything teammates need into --out-dir:
       config.json          arguments, git commit, feature meta.json, split sizes
       metrics.json         per-epoch validation scores and the best epoch
       best.pt              model weights + constructor arguments
       val_outputs.npz      raw verb/noun logits, labels, attention and gate values (Member 1
                            calibrates logits; report_sta_memory.py plots attention/gates)
       test_outputs.npz     raw logits for the hidden test (if --test-csv / test manifest given)
       submission_sta.pt    uncalibrated STA predictions in Codabench format, every event covered
Events whose cutoff is before the video start get the training-prior scores (cold start).

    python scripts/train_sta_memory.py --features-dir "$EPIC_FEATURES" \
        --manifest-dir "$EPIC_MANIFESTS" --test-csv "$EPIC_PROJECT/annotations/EPIC_100_test.csv" \
        --out-dir "$EPIC_TEAM/results/sta_memory/decoupled_S16_seed42" --mode decoupled --steps 16
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from epic_starter.data.events import Event, load_events
from epic_starter.data.manifests import grouped_split, read_manifest, training_priors
from epic_starter.epic_metrics import evaluate, validate_predictions
from epic_starter.memory import Anchors, AuxLabeler, FeatureBank, HistoryDataset, make_mock_bank
from epic_starter.models.aga import N_NOUN, N_VERB, AGAMemoryModel, GatedAGAModel, aga_loss


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data")
    src.add_argument("--manifest-dir", help="folder with train_sta.jsonl / val_sta.jsonl (team split)")
    src.add_argument("--train-csv", help="alternative to --manifest-dir: CSV + grouped video split")
    src.add_argument("--val-fraction", type=float, default=.2)
    src.add_argument("--split-seed", type=int, default=42)
    src.add_argument("--test-csv", help="hidden test CSV (unlabelled) to predict after training")
    src.add_argument("--features-dir", help="consolidated feature version folder; omit for a mock bank")
    src.add_argument("--pool-cache", help="folder to cache token-averaged arrays between runs")
    src.add_argument("--no-pool", action="store_true", help="keep all T tokens (token-level probe)")
    src.add_argument("--no-anchor", action="store_true", help="ignore sta_obs_tokens.npy (ablation)")
    src.add_argument("--limit-videos", type=int, help="smoke tests: keep only the first N videos per split")
    m = p.add_argument_group("model")
    m.add_argument("--mode", choices=["single", "decoupled"], default="decoupled")
    m.add_argument("--steps", type=int, default=16, help="history length S (one step per --stride s)")
    m.add_argument("--stride", type=float, default=1.0)
    m.add_argument("--d-model", type=int, default=512)
    m.add_argument("--heads", type=int, default=8)
    m.add_argument("--alpha", type=float, default=0.8, help="EMA weight of the newest prediction")
    m.add_argument("--dropout", type=float, default=0.5)
    m.add_argument("--present-encoder", choices=["probe", "mean"], default="probe",
                   help="probe: Member 2's factorized probe on the present clip's 24 tokens (default); "
                        "mean: the memory's own token-mean embedding of the last step (ablation)")
    m.add_argument("--present-init", help="Member 2's STA factorized probe checkpoint (best.pt) to start from")
    m.add_argument("--freeze-present", action="store_true", help="keep the initialised probe fixed")
    m.add_argument("--fusion", choices=["aga", "three_way"], default="aga",
                   help="aga: AGA's 2-way gate (baseline); three_way: Member 4's ThreeWayGate "
                        "(history + present now, future branch once predictor features exist)")
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--patience", type=int, default=8, help="stop after N epochs without improvement")
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--aux-weight", type=float, default=0.5)
    t.add_argument("--label-smoothing", type=float, default=0.1, help="same as Member 2's probe training")
    t.add_argument("--class-weights", choices=["none", "sqrt", "inv"], default="none",
                   help="none keeps plain CE so Member 1's post-hoc logit adjustment stays valid")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--num-workers", type=int, default=4)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    if bool(a.manifest_dir) == bool(a.train_csv):
        p.error("give exactly one of --manifest-dir or --train-csv")
    if (a.present_init or a.freeze_present) and a.present_encoder != "probe":
        p.error("--present-init / --freeze-present need --present-encoder probe")
    if a.freeze_present and not a.present_init:
        p.error("--freeze-present needs --present-init")
    return a


def manifest_events(path: Path) -> list[Event]:
    return [Event(**{k: r[k] for k in Event.__dataclass_fields__}) for r in read_manifest(path)]


def load_splits(a):
    if a.manifest_dir:
        d = Path(a.manifest_dir)
        train, val = manifest_events(d / "train_sta.jsonl"), manifest_events(d / "val_sta.jsonl")
        test = manifest_events(d / "test_sta.jsonl") if (d / "test_sta.jsonl").is_file() else []
    else:
        train, val = grouped_split(load_events(Path(a.train_csv)), a.val_fraction, a.split_seed)
        test = []
    if a.test_csv:
        test = load_events(Path(a.test_csv), labeled=False)
    if a.limit_videos:
        def cut(es):
            keep = set(sorted({e.video_id for e in es})[: a.limit_videos])
            return [e for e in es if e.video_id in keep]
        train, val, test = cut(train), cut(val), cut(test)
    if {e.video_id for e in train} & {e.video_id for e in val}:
        raise ValueError("Train and validation share videos: split leakage")
    return train, val, test


def class_weights(events, kind):
    if kind == "none":
        return None, None
    power = -1.0 if kind == "inv" else -0.5
    out = []
    for labels, n in ((np.array([e.verb for e in events]), N_VERB), (np.array([e.noun for e in events]), N_NOUN)):
        counts = np.bincount(labels, minlength=n).astype(np.float64) + 1.0
        w = counts ** power
        w *= counts.sum() / (w * counts).sum()          # average weight per training sample = 1
        out.append(torch.tensor(w, dtype=torch.float32))
    return out


def subsets(events, train):
    """Unseen participants and tail actions (classes outside the head covering 80% of training);
    same definitions and names as Member 2's train_probe.py, so tables line up."""
    seen = {e.participant_id for e in train}
    counts = defaultdict(int)
    for e in train:
        counts[(e.verb, e.noun)] += 1
    head, total, acc = set(), sum(counts.values()), 0
    for key, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        if acc >= 0.8 * total:
            break
        head.add(key)
        acc += c
    return {"unseen": np.array([e.participant_id not in seen for e in events]),
            "tail": np.array([(e.verb, e.noun) not in head for e in events])}


def head_tail(labels, n_classes, head_share=0.8):
    """Classes outside the most frequent ones covering head_share of the instances (EPIC-100 style,
    as Member 1's hybrid_split.tail_classes); from TRAINING labels only."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)
    total, running, tail = counts.sum(), 0, []
    for c in np.argsort(-counts, kind="stable"):
        if counts[c] == 0:
            continue
        if running >= head_share * total:
            tail.append(int(c))
        running += counts[c]
    return sorted(tail)


def official_breakdown(p, val, train, manifest_dir):
    """Member 1's overall / seen / unseen / tail table (epic_metrics.evaluate_breakdown), when
    available. Tail classes come from tail_classes.json of the split, else from training counts."""
    try:
        from epic_starter.epic_metrics import evaluate_breakdown, format_breakdown
    except ImportError:
        return None, None
    tail_file = Path(manifest_dir) / "tail_classes.json" if manifest_dir else None
    if tail_file is not None and tail_file.is_file():
        t = json.loads(tail_file.read_text())
        tv, tn = t["verbs"], t["nouns"]
    else:
        tv, tn = head_tail([e.verb for e in train], N_VERB), head_tail([e.noun for e in train], N_NOUN)
    seen = {e.participant_id for e in train}
    r = evaluate_breakdown(p["verb_logits"], p["noun_logits"], p["verb_label"], p["noun_label"], task="STA",
                           unseen_mask=np.array([e.participant_id not in seen for e in val]),
                           tail_verbs=tv, tail_nouns=tn)
    return r, format_breakdown(r)


def cold_start_scores(train, manifest_dir):
    """Scores for events with no observable frame. Member 1's cold_start_prior.json (actions in the
    first 5 s of training videos) when the split provides it, else the overall training prior."""
    path = Path(manifest_dir) / "cold_start_prior.json" if manifest_dir else None
    if path is not None and path.is_file():
        prior, source = json.loads(path.read_text()), str(path)
    else:
        prior, source = training_priors(train), "training class frequencies"
    return np.log(prior["verb_output"]), np.log(prior["noun_output"]), source


def run_model(model, b, device):
    """Forward pass for both present encoders (present tokens are only in the batch for 'probe')."""
    if "present_tokens" in b:
        return model(b["feats"].to(device, non_blocking=True), b["valid"].to(device, non_blocking=True),
                     b["present_tokens"].to(device, non_blocking=True),
                     (b["present_source"] > 0).to(device, non_blocking=True))
    return model(b["feats"].to(device, non_blocking=True), b["valid"].to(device, non_blocking=True))


@torch.no_grad()
def predict(model, loader, device, prior_v, prior_n, extras=False):
    model.eval()
    out = defaultdict(list)
    for b in loader:
        r = run_model(model, b, device)
        v, n = r["verb_logits"].float().cpu().numpy(), r["noun_logits"].float().cpu().numpy()
        cold = b["use_prior"].numpy().astype(bool)
        v[cold], n[cold] = prior_v, prior_n                 # cold start: training prior
        out["narration_id"] += list(b["narration_id"])
        for k, x in (("verb_logits", v), ("noun_logits", n), ("verb_label", b["verb_label"].numpy()),
                     ("noun_label", b["noun_label"].numpy()), ("use_prior", cold)):
            out[k].append(x)
        if extras:
            out["step_valid"].append(b["valid"].numpy())
            if "present_source" in b:
                out["present_source"].append(b["present_source"].numpy())
            for k in ("attn", "attn_verb", "attn_noun"):          # [B,H,S-1] -> mean over heads
                if k in r:
                    out[k].append(r[k].mean(1).float().cpu().numpy())
            for k in ("gate", "gate_verb", "gate_noun"):          # [B,d] -> share taken from history
                if k in r:
                    out[k].append(r[k].mean(1).float().cpu().numpy())
            for k in ("verb_gates", "noun_gates"):                # three-way weights [B,3]
                if k in r:
                    out[k].append(r[k].float().cpu().numpy())
    return {k: (v if k == "narration_id" else np.concatenate(v)) for k, v in out.items()}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=10).stdout.strip() or None
    except Exception:
        return None


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists():
        raise FileExistsError(f"{out} already holds a finished run; choose a new --out-dir")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    if device.type == "cpu":                       # the cores Slurm gave us, minus the data workers
        cores = int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))
        torch.set_num_threads(max(1, cores - a.num_workers))

    train, val, test = load_splits(a)
    pooled = not a.no_pool
    probe = a.present_encoder == "probe"
    if a.features_dir:
        bank = FeatureBank(root=a.features_dir, pooled=pooled, cache_dir=a.pool_cache)
        # the probe needs all 24 tokens of the exact STA clip; the memory uses their mean
        anchors = None if a.no_anchor else Anchors(a.features_dir, pooled=pooled and not probe,
                                                   cache_dir=a.pool_cache)
    else:
        print("WARNING: no --features-dir, using random MOCK features (pipeline test only)")
        bank, anchors = make_mock_bank(train + val + test, tokens=24, dim=64, encode_time=False), None
        if pooled:
            bank = FeatureBank({k: v for k, v in bank.entries.items()}, pooled=True)
    T, D = bank.shape
    D = bank.token_shape[1]
    if anchors is not None and test:
        missing = sum(e.sta_cutoff >= 0 and anchors.rows.get(e.narration_id) is None for e in test)
        if missing:
            print(f"WARNING: {missing}/{len(test)} test events have no extracted STA clip; their present "
                  "clip falls back to the last bank clip (<= 1 s earlier). Ask Member 2 to extract the "
                  "test events, or train with --no-anchor so training matches test.", flush=True)
    bank_stride = bank.meta.get("bank_stride_s", bank.meta.get("bank_stride_seconds"))
    if getattr(bank, "dropped", None) and any(bank.dropped.values()):
        print(f"feature bank: ignored rows {bank.dropped}", flush=True)
    if bank_stride is not None and float(bank_stride) > a.stride + 1e-9:
        print(f"WARNING: bank stride {bank_stride}s is coarser than memory stride {a.stride}s; "
              "consecutive memory steps may reuse the same clip", flush=True)
    print(f"events train {len(train)} / val {len(val)} / test {len(test)}; features [T,D]=[{T},{D}]; "
          f"mode {a.mode}, S={a.steps}, stride {a.stride}s, device {device}", flush=True)

    prior_v, prior_n, prior_source = cold_start_scores(train, a.manifest_dir)
    kw_ds = dict(anchors=anchors, present_tokens=probe)
    sets = {"train": HistoryDataset(train, bank, a.steps, a.stride, aux=AuxLabeler(train), **kw_ds),
            "val": HistoryDataset(val, bank, a.steps, a.stride, **kw_ds)}
    if test:
        sets["test"] = HistoryDataset(test, bank, a.steps, a.stride, **kw_ds)
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=a.num_workers > 0)
    loaders = {k: DataLoader(ds, shuffle=(k == "train"), generator=torch.Generator().manual_seed(a.seed), **kw)
               for k, ds in sets.items()}
    masks = subsets(val, train)

    ctor = dict(feat_dim=D, d=a.d_model, heads=a.heads, alpha=a.alpha, mode=a.mode, dropout=a.dropout,
                present_encoder=a.present_encoder, freeze_present=a.freeze_present)
    model = AGAMemoryModel(**ctor)
    present_init = None
    if a.present_init:
        epoch0, score0 = model.load_present(a.present_init)
        present_init = {"path": a.present_init, "epoch": epoch0, "val_action_mt5r": score0}
        print(f"present probe initialised from {a.present_init} (its val action MT5R: {score0})", flush=True)
    if a.fusion == "three_way":
        from epic_starter.models.gate import GatedVerbNounHead   # Member 4
        model = GatedAGAModel(model, GatedVerbNounHead(in_dim=a.d_model))
    model = model.to(device)
    wv, wn = class_weights(train, a.class_weights)
    wv, wn = (wv.to(device) if wv is not None else None), (wn.to(device) if wn is not None else None)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    config = {"args": vars(a), "git_commit": git_commit(), "feature_meta": bank.meta,
              "feature_shape": [T, D], "events": {k: len(ds) for k, ds in sets.items()},
              "cold_start_events": {k: int(sum(e.sta_cutoff < 0 for e in ds.events)) for k, ds in sets.items()},
              "present_init": present_init, "cold_start_prior": prior_source,
              "torch": torch.__version__, "python": platform.python_version(),
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # Score before any training. With --present-init (AGA fusion) the memory correction starts at
    # zero, so this is exactly Member 2's probe in this pipeline: same split, clips and evaluator.
    p0 = predict(model, loaders["val"], device, prior_v, prior_n)
    init = {"val": evaluate(p0["verb_logits"], p0["noun_logits"], p0["verb_label"], p0["noun_label"],
                            task="STA", subsets=masks),
            "breakdown": official_breakdown(p0, val, train, a.manifest_dir)[0],
            "probe_alone": bool(present_init) and a.fusion == "aga"}
    print(f"before training  val action MT5R {init['val']['all']['primary_score']:.2f}  "
          f"verb {init['val']['all']['verb_mt5r']:.2f}  noun {init['val']['all']['noun_mt5r']:.2f}"
          + ("  (= the probe alone)" if init["probe_alone"] else ""), flush=True)

    history, best, best_epoch, waited = [], -1.0, -1, 0
    for epoch in range(a.epochs):
        model.train()
        t0, losses = time.time(), []
        for b in loaders["train"]:
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            loss = aga_loss(run_model(model, b, device), b, a.aux_weight, wv, wn, a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        p = predict(model, loaders["val"], device, prior_v, prior_n)
        res = evaluate(p["verb_logits"], p["noun_logits"], p["verb_label"], p["noun_label"],
                       task="STA", subsets=masks)
        score = res["all"]["primary_score"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1),
               "lr": sched.get_last_lr()[0], "val": res}
        history.append(row)
        print(f"epoch {epoch:3d}  loss {row['train_loss']:.3f}  val action MT5R {score:.2f}  "
              f"verb {res['all']['verb_mt5r']:.2f}  noun {res['all']['noun_mt5r']:.2f}  ({row['seconds']}s)",
              flush=True)
        if score > best:
            best, best_epoch, waited = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "ctor": ctor, "fusion": a.fusion, "epoch": epoch,
                        "val_action_mt5r": score}, out / "best.pt")
        else:
            waited += 1
        (out / "metrics.json").write_text(json.dumps({"best_epoch": best_epoch, "best_val_action_mt5r": best,
                                                      "init": init, "history": history}, indent=2))
        if waited >= a.patience:
            print(f"early stop: no improvement for {a.patience} epochs")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    p = predict(model, loaders["val"], device, prior_v, prior_n, extras=True)
    np.savez_compressed(out / "val_outputs.npz", **{k: np.asarray(v) for k, v in p.items()},
                        prior_verb=prior_v, prior_noun=prior_n)
    best_res = history[best_epoch]["val"]
    breakdown, table = official_breakdown(p, val, train, a.manifest_dir)
    summary = {"best_epoch": best_epoch, "best_val_action_mt5r": best, "best_val": best_res,
               "best_val_breakdown": breakdown, "init": init, "history": history}
    if table:
        print("official breakdown (Member 1's evaluator):\n" + table, flush=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("best epoch", best_epoch, json.dumps({k: v for k, v in best_res["all"].items()
                                                if isinstance(v, (int, float))}))

    if "test" in loaders:
        q = predict(model, loaders["test"], device, prior_v, prior_n)
        np.savez_compressed(out / "test_outputs.npz", narration_id=np.asarray(q["narration_id"]),
                            verb_logits=q["verb_logits"], noun_logits=q["noun_logits"], use_prior=q["use_prior"])
        rows = [{"narration_id": i, "verb_output": v.astype(np.float32), "noun_output": n.astype(np.float32)}
                for i, v, n in zip(q["narration_id"], q["verb_logits"], q["noun_logits"])]
        rows = validate_predictions(rows, [e.narration_id for e in test])   # every test event exactly once
        torch.save([{"narration_id": r["narration_id"], "verb_output": torch.from_numpy(r["verb_output"]),
                     "noun_output": torch.from_numpy(r["noun_output"])} for r in rows],
                   out / "submission_sta.pt")
        print(f"wrote {len(rows)} STA test predictions ({int(q['use_prior'].sum())} cold start)")
    return summary


if __name__ == "__main__":
    main()
