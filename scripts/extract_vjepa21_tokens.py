"""
Member 2: V-JEPA 2.1 Compact Token Extractor (Sharded per video) & Memmap Consolidator.

Subcommands:
  1. extract     - Run on Slurm GPU array (writes shards/<video_id>.npz atomically)
  2. consolidate - Merge all shards/<video_id>.npz into mmap-ready .npy arrays + CSV indices
  3. dummy       - Generate tiny dummy .npy arrays for immediate local/CPU pipeline testing
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

import av
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.data.events import Event, Window, load_events, observation
from epic_starter.data.manifests import video_index


def egocentric_bottom_crop_and_resize(frames_rgb: np.ndarray, target_size: int = 384) -> torch.Tensor:
    """
    Convert uint8 [T, H, W, 3] (640x360) -> normalized float32 tensor [3, T, target_size, target_size].
    Resizes to (target_size, round(target_size * 284 / 256)) and takes a bottom-centered square crop.
    """
    # [T, H, W, 3] -> [T, 3, H, W] in [0, 1]
    x = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float().div_(255.0)
    resize_w = int(round(target_size * (284.0 / 256.0)))  # 426 for 384, 284 for 256
    resize_h = target_size
    x = F.interpolate(x, size=(resize_h, resize_w), mode="bilinear", align_corners=False)

    # Bottom-centered square crop (hands enter from bottom of frame, centered horizontally)
    left = (resize_w - target_size) // 2
    x = x[:, :, resize_h - target_size : resize_h, left : left + target_size]

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.permute(1, 0, 2, 3).contiguous()  # [3, T, H, W]


def sample_window_pts(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    origin: Fraction,
    window: Window,
    num_frames: int = 16,
) -> tuple[np.ndarray, bool, float]:
    """
    Decode exactly `num_frames` RGB frames inside `window` using presentation timestamps.
    Enforces strict STA anti-leakage (when <= window.end when inclusive_end=True).
    Returns (frames_uint8 [T, H, W, 3], has_valid_frames, max_sampled_pts_seconds).
    """
    if window.empty or window.end < window.start or window.start < 0:
        return np.zeros((num_frames, 360, 640, 3), dtype=np.uint8), False, -1.0

    lo, hi = Fraction(str(window.start)), Fraction(str(window.end))
    targets = np.array([
        window.start + (i + 0.5) * (window.end - window.start) / num_frames
        for i in range(num_frames)
    ], dtype=np.float64)

    best = [None] * num_frames
    distances = np.full(num_frames, np.inf)

    offset = math.floor((origin + lo) / stream.time_base)
    container.seek(max(0, offset), backward=True, any_frame=False, stream=stream)

    for frame in container.decode(stream):
        if frame.pts is None or frame.time_base is None:
            continue
        when = frame.pts * frame.time_base - origin
        if when > hi or (when == hi and not window.inclusive_end):
            break
        if when < lo or frame.is_corrupt:
            continue
        when_f = float(when)
        dist = np.abs(targets - when_f)
        for i in np.flatnonzero(dist < distances):
            distances[i] = dist[i]
            best[i] = (frame.to_ndarray(format="rgb24"), when_f)

    valid_any = any(b is not None for b in best)
    if not valid_any:
        return np.zeros((num_frames, 360, 640, 3), dtype=np.uint8), False, -1.0

    # Fill any missing target slot with nearest decoded valid frame
    first_valid = next(b for b in best if b is not None)
    H, W, _ = first_valid[0].shape
    out = np.zeros((num_frames, H, W, 3), dtype=np.uint8)
    max_pts = -1.0
    last_arr, last_t = first_valid
    for i in range(num_frames):
        if best[i] is not None:
            last_arr, last_t = best[i]
        out[i] = last_arr
        if last_t > max_pts:
            max_pts = last_t

    if window.inclusive_end and max_pts > window.end + 1e-5:
        raise AssertionError(f"PTS leakage: decoded frame at {max_pts}s > window.end {window.end}s")
    return out, True, max_pts


@torch.no_grad()
def pool_24_tokens(patch_tokens: torch.Tensor, t_slices: int = 8, h_grid: int = 24, w_grid: int = 24) -> torch.Tensor:
    """
    Convert full V-JEPA 2.1 spatiotemporal tokens [B, T'*H'*W', D] -> compact [B, 24, D] float16:
      - 8 temporal tubelet-mean tokens: spatial average over (H', W')
      - 16 active spatial grid tokens: 4x4 adaptive spatial pool over the last 2 temporal slices (final 0.5s)
    """
    B, N, D = patch_tokens.shape
    x = patch_tokens.view(B, t_slices, h_grid, w_grid, D)

    # 1. Temporal tubelet tokens: [B, 8, D]
    e_temp = x.mean(dim=(2, 3))

    # 2. Active spatial 4x4 grid from last 2 temporal slices: [B, 16, D]
    last_two = x[:, -2:, :, :, :].mean(dim=1)  # [B, H', W', D]
    last_two_chw = last_two.permute(0, 3, 1, 2)  # [B, D, H', W']
    spat_4x4 = F.adaptive_avg_pool2d(last_two_chw, (4, 4))  # [B, D, 4, 4]
    e_spat = spat_4x4.flatten(2).transpose(1, 2)  # [B, 16, D]

    return torch.cat([e_temp, e_spat], dim=1).to(torch.float16)  # [B, 24, D]


class VJEPA21Extractor(nn.Module if hasattr(torch, "nn") else object):
    """Wrapper that loads frozen V-JEPA 2.1 encoder & predictor from facebookresearch/vjepa2."""

    def __init__(self, vjepa_repo: Path, ckpt_path: Path, img_size: int = 384, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.img_size = img_size
        self.patch_size = 16
        self.tubelet_size = 2
        self.num_frames = 16
        self.t_slices = self.num_frames // self.tubelet_size  # 8
        self.grid_size = self.img_size // self.patch_size     # 24 for 384, 16 for 256

        if str(vjepa_repo) not in sys.path:
            sys.path.insert(0, str(vjepa_repo))

        # Load checkpoint
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        import src.models.vision_transformer as vit
        import src.models.predictor as pred_mod

        self.encoder = vit.vit_large(
            img_size=(img_size, img_size),
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
        )
        enc_sd = state.get("encoder", state.get("target_encoder", state.get("model", state)))
        enc_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_sd.items()}
        self.encoder.load_state_dict(enc_sd, strict=False)
        self.encoder.eval().to(self.device)

        self.predictor = None
        if "predictor" in state:
            try:
                self.predictor = pred_mod.vit_predictor(
                    num_patches=(img_size // self.patch_size) ** 2,
                    embed_dim=1024,
                    predictor_embed_dim=384,
                    depth=12,
                    num_heads=12,
                    num_frames=self.num_frames,
                    tubelet_size=self.tubelet_size,
                )
                pred_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state["predictor"].items()}
                self.predictor.load_state_dict(pred_sd, strict=False)
                self.predictor.eval().to(self.device)
            except Exception as e:
                print(f"[WARN] Predictor init fallback ({e}); using temporal extrap fallback for sta_pred_tokens.")
                self.predictor = None

    @torch.no_grad()
    def extract_batch(self, clips: torch.Tensor, run_predictor: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        clips: [B, 3, 16, H, W] on CPU -> returns (obs_tokens [B, 24, 1024] fp16, pred_tokens [B, 24, 1024] fp16).
        """
        x = clips.to(self.device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feat = self.encoder(x)  # [B, T'*H'*W', 1024]
            obs_24 = pool_24_tokens(feat, self.t_slices, self.grid_size, self.grid_size)

            pred_24 = None
            if run_predictor:
                B, N, D = feat.shape
                # JFAA-style near-future latent prediction: feed observed tubelets and predict shifted future tubelets
                if self.predictor is not None:
                    try:
                        # Construct context indices (first 6 tubelets) & target mask indices (last 2 tubelets)
                        n_spat = self.grid_size * self.grid_size
                        ctx_idx = torch.arange(0, 6 * n_spat, device=self.device).unsqueeze(0).expand(B, -1)
                        tgt_idx = torch.arange(6 * n_spat, 8 * n_spat, device=self.device).unsqueeze(0).expand(B, -1)
                        pred_tail = self.predictor(feat, ctx_idx, tgt_idx)
                        if isinstance(pred_tail, (tuple, list)):
                            pred_tail = pred_tail[0]
                        full_pred = torch.cat([feat[:, 2 * n_spat :, :], pred_tail], dim=1)
                        pred_24 = pool_24_tokens(full_pred, self.t_slices, self.grid_size, self.grid_size)
                    except Exception:
                        pass
                if pred_24 is None:
                    # Linear latent velocity extrapolation fallback in token space
                    vel = obs_24[:, 4:8, :] - obs_24[:, 0:4, :]
                    pred_temp = torch.cat([obs_24[:, 4:8, :], obs_24[:, 4:8, :] + 0.5 * vel], dim=1)
                    pred_24 = torch.cat([pred_temp, obs_24[:, 8:, :]], dim=1)

        return obs_24.cpu(), (pred_24.cpu() if pred_24 is not None else None)


def run_extract(args):
    out_root = Path(args.out_dir)
    shards_dir = out_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(Path(args.train_csv), labeled=True)
    if args.test_csv:
        events.extend(load_events(Path(args.test_csv), labeled=False))

    v_index = video_index(Path(args.video_root))
    by_video: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_video[e.video_id].append(e)

    all_videos = sorted(by_video.keys())
    my_videos = all_videos[args.rank :: args.world_size]
    if args.max_videos > 0:
        my_videos = my_videos[: args.max_videos]

    print(f"[Rank {args.rank}/{args.world_size}] Assigned {len(my_videos)} / {len(all_videos)} videos.")
    extractor = VJEPA21Extractor(Path(args.vjepa_repo), Path(args.ckpt), img_size=args.img_size, device=args.device)

    for vid in tqdm(my_videos, desc=f"Rank {args.rank}"):
        shard_path = shards_dir / f"{vid}.npz"
        if shard_path.exists() and not args.overwrite:
            continue

        video_path = v_index.get(vid)
        if video_path is None:
            raise FileNotFoundError(f"Missing video file for {vid}")

        vid_events = sorted(by_video[vid], key=lambda e: e.start)
        n_ev = len(vid_events)

        ar_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
        sta_obs_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
        sta_pred_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
        narration_ids = np.array([e.narration_id for e in vid_events])
        sta_clip_end_s = np.full(n_ev, -1.0, dtype=np.float64)
        sta_use_prior = np.zeros(n_ev, dtype=bool)

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            origin = (stream.start_time or 0) * stream.time_base

            # 1. Decode AR and STA event-anchored clips in batched GPU passes
            ar_batch, ar_indices = [], []
            sta_batch, sta_indices = [], []

            for idx, ev in enumerate(vid_events):
                # AR Window: [start, stop)
                w_ar = observation(ev, "AR")
                frames_ar, valid_ar, _ = sample_window_pts(container, stream, origin, w_ar, num_frames=16)
                if not valid_ar:
                    raise RuntimeError(f"AR window empty for {ev.narration_id}")
                ar_batch.append(egocentric_bottom_crop_and_resize(frames_ar, args.img_size))
                ar_indices.append(idx)

                # STA Window: exact event-anchored 2.0s window [max(0, sta_cutoff - 2.0), sta_cutoff]
                w_sta = observation(ev, "STA", context=args.sta_clip_len)
                frames_sta, valid_sta, max_pts = sample_window_pts(container, stream, origin, w_sta, num_frames=16)
                if not valid_sta:
                    sta_use_prior[idx] = True
                    sta_clip_end_s[idx] = -1.0
                else:
                    sta_use_prior[idx] = False
                    sta_clip_end_s[idx] = max_pts
                    sta_batch.append(egocentric_bottom_crop_and_resize(frames_sta, args.img_size))
                    sta_indices.append(idx)

                if len(ar_batch) >= args.batch_size:
                    tok_ar, _ = extractor.extract_batch(torch.stack(ar_batch), run_predictor=False)
                    ar_arr[ar_indices] = tok_ar.numpy()
                    ar_batch, ar_indices = [], []

                if len(sta_batch) >= args.batch_size:
                    tok_obs, tok_pred = extractor.extract_batch(torch.stack(sta_batch), run_predictor=True)
                    sta_obs_arr[sta_indices] = tok_obs.numpy()
                    sta_pred_arr[sta_indices] = tok_pred.numpy()
                    sta_batch, sta_indices = [], []

            if ar_batch:
                tok_ar, _ = extractor.extract_batch(torch.stack(ar_batch), run_predictor=False)
                ar_arr[ar_indices] = tok_ar.numpy()
            if sta_batch:
                tok_obs, tok_pred = extractor.extract_batch(torch.stack(sta_batch), run_predictor=True)
                sta_obs_arr[sta_indices] = tok_obs.numpy()
                sta_pred_arr[sta_indices] = tok_pred.numpy()

            # 2. Dense Video Memory Bank for Member 3 (AGA) at stride args.bank_stride
            max_video_t = max(e.stop for e in vid_events)
            bank_ends = np.arange(args.bank_stride, max_video_t + args.bank_stride, args.bank_stride, dtype=np.float64)
            bank_arr = np.zeros((len(bank_ends), 24, 1024), dtype=np.float16)
            bank_actual_ends = np.full(len(bank_ends), -1.0, dtype=np.float64)
            b_batch, b_indices = [], []

            for b_idx, t_end in enumerate(bank_ends):
                w_bank = Window(max(0.0, float(t_end - args.sta_clip_len)), float(t_end), inclusive_end=True)
                frames_b, valid_b, max_pts_b = sample_window_pts(container, stream, origin, w_bank, num_frames=16)
                if valid_b:
                    bank_actual_ends[b_idx] = max_pts_b
                    b_batch.append(egocentric_bottom_crop_and_resize(frames_b, args.img_size))
                    b_indices.append(b_idx)
                if len(b_batch) >= args.batch_size:
                    tok_b, _ = extractor.extract_batch(torch.stack(b_batch), run_predictor=False)
                    bank_arr[b_indices] = tok_b.numpy()
                    b_batch, b_indices = [], []
            if b_batch:
                tok_b, _ = extractor.extract_batch(torch.stack(b_batch), run_predictor=False)
                bank_arr[b_indices] = tok_b.numpy()

        # Atomic write to Lustre shard via tmp file + rename
        tmp_shard = shards_dir / f".{vid}.{uuid.uuid4().hex}.tmp.npz"
        np.savez(
            tmp_shard,
            video_id=vid,
            narration_ids=narration_ids,
            ar_tokens=ar_arr,
            sta_obs_tokens=sta_obs_arr,
            sta_pred_tokens=sta_pred_arr,
            sta_clip_end_s=sta_clip_end_s,
            sta_use_prior=sta_use_prior,
            bank_tokens=bank_arr,
            bank_t_end_s=bank_actual_ends,
        )
        tmp_shard.replace(shard_path)


def run_consolidate(args):
    out_root = Path(args.out_dir)
    shards_dir = out_root / "shards"
    shard_files = sorted(shards_dir.glob("*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {shards_dir}")

    # Pass 1: count total events and bank rows
    total_events = 0
    total_bank = 0
    shard_meta = []
    for sf in tqdm(shard_files, desc="Scanning shards"):
        with np.load(sf) as data:
            n_ev = len(data["narration_ids"])
            n_bk = len(data["bank_t_end_s"])
            shard_meta.append((sf, n_ev, n_bk))
            total_events += n_ev
            total_bank += n_bk

    print(f"Consolidating {len(shard_files)} shards: {total_events} events, {total_bank} bank steps...")

    ar_mmap = np.lib.format.open_memmap(
        out_root / "ar_tokens.npy", mode="w+", dtype=np.float16, shape=(total_events, 24, 1024)
    )
    sta_obs_mmap = np.lib.format.open_memmap(
        out_root / "sta_obs_tokens.npy", mode="w+", dtype=np.float16, shape=(total_events, 24, 1024)
    )
    sta_pred_mmap = np.lib.format.open_memmap(
        out_root / "sta_pred_tokens.npy", mode="w+", dtype=np.float16, shape=(total_events, 24, 1024)
    )
    bank_mmap = np.lib.format.open_memmap(
        out_root / "bank_tokens.npy", mode="w+", dtype=np.float16, shape=(total_bank, 24, 1024)
    )

    event_rows = []
    bank_rows = []
    ev_cursor = 0
    bk_cursor = 0

    for sf, n_ev, n_bk in tqdm(shard_meta, desc="Writing memmaps"):
        with np.load(sf) as data:
            vid = str(data["video_id"])
            ar_mmap[ev_cursor : ev_cursor + n_ev] = data["ar_tokens"]
            sta_obs_mmap[ev_cursor : ev_cursor + n_ev] = data["sta_obs_tokens"]
            sta_pred_mmap[ev_cursor : ev_cursor + n_ev] = data["sta_pred_tokens"]

            nids = data["narration_ids"]
            ends = data["sta_clip_end_s"]
            priors = data["sta_use_prior"]
            for i in range(n_ev):
                event_rows.append({
                    "narration_id": str(nids[i]),
                    "row": ev_cursor + i,
                    "video_id": vid,
                    "sta_clip_end_s": float(ends[i]),
                    "sta_use_prior": bool(priors[i]),
                })
            ev_cursor += n_ev

            if n_bk > 0:
                bank_mmap[bk_cursor : bk_cursor + n_bk] = data["bank_tokens"]
                b_ends = data["bank_t_end_s"]
                for j in range(n_bk):
                    bank_rows.append({
                        "video_id": vid,
                        "t_end_s": float(b_ends[j]),
                        "row": bk_cursor + j,
                    })
                bk_cursor += n_bk

    ar_mmap.flush()
    sta_obs_mmap.flush()
    sta_pred_mmap.flush()
    bank_mmap.flush()

    pd.DataFrame(event_rows).to_csv(out_root / "events_index.csv", index=False)
    pd.DataFrame(bank_rows).to_csv(out_root / "bank_index.csv", index=False)

    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass

    meta = {
        "backbone": "vjepa2_1_vitl_384",
        "img_size": args.img_size,
        "frames_per_clip": 16,
        "tokens_per_clip": "24 (8 temporal tubelets + 16 spatial 4x4 grid from last 2 tubelets)",
        "embed_dim": 1024,
        "dtype": "float16",
        "crop": "bottom-centered square crop after aspect-preserving resize",
        "bank_stride_seconds": args.bank_stride,
        "total_events": total_events,
        "total_bank_steps": total_bank,
        "num_videos": len(shard_files),
        "git_commit": git_hash,
        "consolidated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Consolidation complete! Saved to {out_root}")


def run_dummy(args):
    """Create tiny dummy consolidated arrays for N events so all 4 team members can test pipelines today."""
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    all_events = load_events(Path(args.train_csv), labeled=True)
    n_ev = min(args.num_events, len(all_events))
    rng = np.random.default_rng(42)
    chosen = np.sort(rng.choice(len(all_events), size=n_ev, replace=False))
    events = [all_events[i] for i in chosen]
    for name in ("ar_tokens.npy", "sta_obs_tokens.npy", "sta_pred_tokens.npy"):
        arr = np.lib.format.open_memmap(out_root / name, mode="w+", dtype=np.float16, shape=(n_ev, 24, 1024))
        arr[:] = rng.standard_normal((n_ev, 24, 1024), dtype=np.float32).astype(np.float16)
        arr.flush()

    rows = []
    for i, e in enumerate(events):
        use_prior = e.sta_cutoff < 0
        rows.append({
            "narration_id": e.narration_id,
            "row": i,
            "video_id": e.video_id,
            "sta_clip_end_s": -1.0 if use_prior else e.sta_cutoff,
            "sta_use_prior": use_prior,
        })
    pd.DataFrame(rows).to_csv(out_root / "events_index.csv", index=False)
    print(f"Created dummy TokenBank with {n_ev} events at {out_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--train-csv", required=True)
    p_ext.add_argument("--test-csv", default=None)
    p_ext.add_argument("--video-root", required=True)
    p_ext.add_argument("--vjepa-repo", required=True)
    p_ext.add_argument("--ckpt", required=True)
    p_ext.add_argument("--out-dir", required=True)
    p_ext.add_argument("--img-size", type=int, default=384)
    p_ext.add_argument("--sta-clip-len", type=float, default=2.0)
    p_ext.add_argument("--bank-stride", type=float, default=1.0)
    p_ext.add_argument("--batch-size", type=int, default=16)
    p_ext.add_argument("--rank", type=int, default=0)
    p_ext.add_argument("--world-size", type=int, default=1)
    p_ext.add_argument("--max-videos", type=int, default=0)
    p_ext.add_argument("--device", default="cuda")
    p_ext.add_argument("--overwrite", action="store_true")

    p_con = sub.add_parser("consolidate")
    p_con.add_argument("--out-dir", required=True)
    p_con.add_argument("--img-size", type=int, default=384)
    p_con.add_argument("--bank-stride", type=float, default=1.0)

    p_dum = sub.add_parser("dummy")
    p_dum.add_argument("--train-csv", required=True)
    p_dum.add_argument("--out-dir", required=True)
    p_dum.add_argument("--num-events", type=int, default=256)

    args = parser.parse_args()
    if args.cmd == "extract":
        run_extract(args)
    elif args.cmd == "consolidate":
        run_consolidate(args)
    elif args.cmd == "dummy":
        run_dummy(args)


if __name__ == "__main__":
    main()
