"""Member 4 feasibility check: does the frozen V-JEPA predictor run on our STA clips, and is its output usable?

Run from the repository root on a compute node (scripts/member4_feasibility.slurm does this):
  python scripts/predictor_feasibility.py --backend vjepa --vjepa-root VJEPA2_CLONE \
      --config VJEPA2_CLONE/configs/eval_2_1/vitl-384/ek100.yaml --checkpoint CKPT.pt \
      --csv EPIC_100_train.csv --video-root VIDEOS --out REPORT.json
`--backend dummy` uses a tiny random model with the same interface (tests and dry runs).

Every clip ends at or before the event's STA cutoff (action start - 1 s); the script checks the
decoded frame timestamps. No labels are used. Preprocessing follows our pipeline: the full frame is
resized to RxR and normalized with ImageNet mean/std (the official eval centre-crops instead).
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from epic_starter.data.events import Window, load_events
from epic_starter.data.manifests import video_index
from epic_starter.data.sampling import decode_window
from epic_starter.models.future_latent import (DummyAnticipativeModel, anticipate, load_vjepa_model,
                                               same_feature_space)

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def legal_clip(event, path, end, frames, fps, size):
    """Return a normalized [3, T, H, W] clip of `frames` frames at `fps` ending at `end` (<= STA cutoff)."""
    if end > event.sta_cutoff + 1e-9:
        raise ValueError(f"{event.narration_id}: clip end {end:.3f}s is after the STA cutoff {event.sta_cutoff:.3f}s")
    start = end - frames / fps
    if start < 0:
        raise ValueError(f"{event.narration_id}: not enough history before {end:.3f}s")
    images, valid, times = decode_window(path, Window(start, end, True), frames, size)
    if not valid.all():
        raise ValueError(f"{event.narration_id}: missing frames in [{start:.3f}, {end:.3f}]")
    latest = float(times.max())
    if latest > event.sta_cutoff + 1e-6:
        raise AssertionError(f"{event.narration_id}: decoded a frame at {latest:.3f}s after the cutoff")
    x = (torch.from_numpy(images).float() / 255 - MEAN) / STD  # [T, H, W, 3]
    return x.permute(3, 0, 1, 2).contiguous(), latest


def select_events(events, index, needed_history, count, seed):
    """Spread `count` events over as many videos as possible, among events with enough history."""
    by_video = {}
    for e in sorted(events, key=lambda e: e.narration_id):
        if e.video_id in index and e.sta_cutoff >= needed_history:
            by_video.setdefault(e.video_id, []).append(e)
    rng = random.Random(seed)
    videos = sorted(by_video)
    rng.shuffle(videos)
    for v in videos:
        rng.shuffle(by_video[v])
    chosen = []
    while len(chosen) < count and any(by_video[v] for v in videos):
        for v in videos:  # round robin: one event per video before taking a second from any video
            if by_video[v] and len(chosen) < count:
                chosen.append(by_video[v].pop())
    return chosen


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _pairwise_offdiag_cosine(vectors):
    v = torch.nn.functional.normalize(vectors, dim=-1)
    sim = v @ v.T
    n = sim.shape[0]
    return float((sim.sum() - sim.diagonal().sum()) / max(n * (n - 1), 1))


def _retrieval_top1(queries, keys):
    """Fraction of events whose query is most similar to its own key (chance = 1 / #events)."""
    sim = torch.nn.functional.normalize(queries, dim=-1) @ torch.nn.functional.normalize(keys, dim=-1).T
    return float((sim.argmax(dim=1) == torch.arange(sim.shape[0])).float().mean())


def _vjepa_commit(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_model(args):
    if args.backend == "dummy":
        torch.manual_seed(0)
        return DummyAnticipativeModel(resolution=args.resolution, frames_per_second=args.fps)
    return load_vjepa_model(args.vjepa_root, args.config_dict, args.checkpoint,
                            args.frames, args.fps, args.resolution)


def run_feasibility(args):
    """Run the check and return the report dict (also written to args.out)."""
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new --out")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    model = build_model(args).to(device).eval()
    tubelet, grid = model.tubelet_size, model.grid_size
    slot = tubelet / args.fps  # seconds covered by one predicted latent slice
    steps = {h: int(h * args.fps / tubelet) for h in args.horizons}
    clip_seconds = args.frames / args.fps
    needed = clip_seconds + max((s + 1) * slot for s in steps.values())
    index = video_index(Path(args.video_root))
    events = select_events(load_events(Path(args.csv)), index, needed, args.num_events, args.seed)
    if len(events) < 2:
        raise ValueError("Need at least two events with enough history")

    pooled_pred, pooled_enc, fidelity_rows = [], [], {h: [] for h in args.horizons}
    latest_minus_cutoff, seconds, finite, wrapper_diff = -math.inf, [], True, None
    model_info = None
    with torch.no_grad():
        for i, event in enumerate(events):
            path = index[event.video_id]
            x_obs, latest = legal_clip(event, path, event.sta_cutoff, args.frames, args.fps, args.resolution)
            latest_minus_cutoff = max(latest_minus_cutoff, latest - event.sta_cutoff)
            x_obs = x_obs.unsqueeze(0).to(device)
            t = torch.full((1,), float(args.anticipation), device=device)
            start = time.perf_counter()
            enc, enc_full, pred, pred_full = anticipate(model, x_obs, t)
            seconds.append(time.perf_counter() - start)
            finite &= bool(torch.isfinite(enc).all() and torch.isfinite(pred).all())
            if i == 0:
                # Our re-implementation must reproduce the official wrapper output exactly.
                official = model(x_obs, t)
                mine = torch.cat([enc, pred], dim=1)
                wrapper_diff = float((official - mine).abs().max()) if official.shape == mine.shape else math.inf
                model_info = {"encoder_dim": int(model.encoder.embed_dim), "encoder_full_dim": int(enc_full.shape[-1]),
                              "predictor_full_dim": int(pred_full.shape[-1]), "grid": int(grid),
                              "tubelet": int(tubelet), "encoder_tokens": int(enc.shape[1]),
                              "predicted_tokens": int(pred.shape[1]),
                              "same_feature_space": same_feature_space(enc_full, pred_full, model.encoder.embed_dim)}
            pooled_pred.append(pred.mean(dim=1).float().cpu())
            pooled_enc.append(enc.mean(dim=1).float().cpu())
            if not model_info["same_feature_space"]:
                continue
            target = enc[:, -grid ** 2:]  # last observed slice, which ends at the cutoff
            for h, k in steps.items():
                # Predict that same last slice from a context that stops (k + 1) slices earlier:
                # every frame used is still before the cutoff.
                x_ctx, latest = legal_clip(event, path, event.sta_cutoff - (k + 1) * slot,
                                           args.frames, args.fps, args.resolution)
                latest_minus_cutoff = max(latest_minus_cutoff, latest - event.sta_cutoff)
                enc_c, _, pred_c, _ = anticipate(model, x_ctx.unsqueeze(0).to(device),
                                                 torch.full((1,), float(h), device=device))
                predicted, persistence = pred_c[:, :grid ** 2], enc_c[:, -grid ** 2:]
                fidelity_rows[h].append({
                    "pred_token_cos": float(_cosine(predicted, target).mean()),
                    "persist_token_cos": float(_cosine(persistence, target).mean()),
                    "pred": predicted.mean(dim=1).float().cpu(), "persist": persistence.mean(dim=1).float().cpu(),
                    "context_mean": enc_c.mean(dim=1).float().cpu(), "target": target.mean(dim=1).float().cpu()})

    P, E = torch.cat(pooled_pred), torch.cat(pooled_enc)
    report = {
        "backend": args.backend, "config": str(args.config) if args.config else None,
        "checkpoint": Path(args.checkpoint).name if args.checkpoint else None,
        "vjepa2_commit": _vjepa_commit(args.vjepa_root) if args.vjepa_root else None,
        "frames": args.frames, "fps": args.fps, "resolution": args.resolution,
        "anticipation_s": args.anticipation, "horizons_s": args.horizons,
        "preprocessing": "full frame resized to RxR, ImageNet mean/std",
        "device": str(device), "torch": torch.__version__, "threads": torch.get_num_threads(),
        "events": [e.narration_id for e in events], "model": model_info,
        "checks": {"runs": True, "finite": finite,
                   "matches_official_wrapper": wrapper_diff is not None and wrapper_diff <= 1e-3,
                   "max_abs_diff_vs_official_wrapper": wrapper_diff,
                   "latest_frame_minus_cutoff_s": latest_minus_cutoff},
        "timing": {"seconds_per_clip_mean": sum(seconds) / len(seconds),
                   "peak_cuda_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None},
        "predicted_tokens": {"mean_cosine_between_events": _pairwise_offdiag_cosine(P),
                             "mean_std_across_events": float(P.std(dim=0).mean()),
                             "encoder_reference_cosine_between_events": _pairwise_offdiag_cosine(E)},
    }
    if model_info["same_feature_space"]:
        fid = {}
        for h, rows in fidelity_rows.items():
            stack = {k: torch.cat([r[k] for r in rows]) for k in ("pred", "persist", "context_mean", "target")}
            fid[str(h)] = {
                "predictor_token_cosine": sum(r["pred_token_cos"] for r in rows) / len(rows),
                "persistence_token_cosine": sum(r["persist_token_cos"] for r in rows) / len(rows),
                "predictor_pooled_cosine": float(_cosine(stack["pred"], stack["target"]).mean()),
                "persistence_pooled_cosine": float(_cosine(stack["persist"], stack["target"]).mean()),
                "context_mean_pooled_cosine": float(_cosine(stack["context_mean"], stack["target"]).mean()),
                "predictor_retrieval_top1": _retrieval_top1(stack["pred"], stack["target"]),
                "persistence_retrieval_top1": _retrieval_top1(stack["persist"], stack["target"]),
                "chance_top1": 1 / len(rows)}
        report["fidelity"] = fid
    else:
        report["fidelity"] = {"skipped": "predictor output is in a different feature space than the encoder "
                                         "(distilled V-JEPA 2.1 student); compare with a non-distilled checkpoint"}
    collapsed = report["predicted_tokens"]["mean_cosine_between_events"] > 0.99
    report["verdict"] = ("runs on our STA clips; " + ("outputs look COLLAPSED (near-identical across events)" if collapsed
                         else "outputs differ across events") + ("" if finite else "; NON-FINITE values found"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["vjepa", "dummy"], default="vjepa")
    p.add_argument("--vjepa-root", help="Clone of facebookresearch/vjepa2")
    p.add_argument("--config", help="vjepa2 eval config, e.g. configs/eval_2_1/vitl-384/ek100.yaml")
    p.add_argument("--checkpoint", help="Pretrained checkpoint (.pt) matching --config")
    p.add_argument("--csv", required=True, help="Course annotation CSV (labels are not used)")
    p.add_argument("--video-root", required=True)
    p.add_argument("--out", required=True, help="JSON report path (must not exist)")
    p.add_argument("--frames", type=int, help="Frames per clip (default: config, else 8)")
    p.add_argument("--fps", type=int, help="Frames per second (default: config, else 8)")
    p.add_argument("--resolution", type=int, help="Input size (default: config, else 32)")
    p.add_argument("--anticipation", type=float, help="Seconds ahead for the STA prediction (default: config, else 1.0)")
    p.add_argument("--horizons", type=float, nargs="+", default=[0.0, 1.0],
                   help="Fidelity check horizons in seconds (multiples of one slice)")
    p.add_argument("--num-events", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    data = {}
    args.config_dict = None
    if args.backend == "vjepa":
        if not (args.vjepa_root and args.config and args.checkpoint):
            p.error("--backend vjepa needs --vjepa-root, --config and --checkpoint")
        import yaml
        args.config_dict = yaml.safe_load(Path(args.config).read_text())
        data = args.config_dict.get("experiment", {}).get("data", {})
    args.frames = args.frames or data.get("frames_per_clip", 8)
    args.fps = args.fps or data.get("frames_per_second", 8)
    args.resolution = args.resolution or data.get("resolution", 32)
    if args.anticipation is None:
        args.anticipation = float((data.get("anticipation_time_sec") or [1.0])[0])
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    return args


if __name__ == "__main__":
    print(json.dumps(run_feasibility(parse_args()), indent=2))
