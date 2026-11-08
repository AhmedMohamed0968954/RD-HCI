"""
Member 2: Fast Single-Pass V-JEPA 2.1 ViT-L/384 Feature & Predictor Token Extractor + Consolidator.

Strictly adheres to:
  - docs/MEMORY_INTERFACE.md (Member 3 FeatureBank & Anchors)
  - docs/MEMBER4_FORMAT_CONTRACT.md (Member 4 load_vjepa_model & anticipate)
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Disable tqdm background monitor thread to avoid SIGILL on GraniteRapids Python 3.13
tqdm.monitor_interval = 0

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.data.events import Event, Window, load_events, observation
from epic_starter.data.manifests import video_index
from epic_starter.models.future_latent import anticipate, load_vjepa_model


def write_json_safe(path: Path, data: dict) -> None:
    """Pure-Python JSON formatter that bypasses Python 3.13's C _json encoder on GraniteRapids."""
    lines = ["{"]
    items = list(data.items())
    for idx, (k, v) in enumerate(items):
        comma = "," if idx < len(items) - 1 else ""
        if isinstance(v, bool):
            val_str = "true" if v else "false"
        elif isinstance(v, (int, float)):
            val_str = str(v)
        else:
            escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
            val_str = f'"{escaped}"'
        lines.append(f'  "{k}": {val_str}{comma}')
    lines.append("}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def preprocess_clip(frames_rgb: np.ndarray, target_size: int = 384, crop_mode: str = "bottom_center") -> torch.Tensor:
    """
    Convert uint8 [T, H, W, 3] (640x360) -> ImageNet-normalized float32 tensor [3, T, target_size, target_size].
    """
    x = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float().div_(255.0)
    if crop_mode == "resize_square":
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    else:
        resize_w = int(round(target_size * (284.0 / 256.0)))  # 426 for 384
        x = F.interpolate(x, size=(target_size, resize_w), mode="bilinear", align_corners=False)
        left = (resize_w - target_size) // 2
        x = x[:, :, :, left : left + target_size]

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.permute(1, 0, 2, 3).contiguous()  # [3, T, H, W]


def extract_video_streaming(
    video_path: Path,
    vid_events: list[Event],
    extractor: "VJEPA21Extractor",
    img_size: int = 384,
    crop_mode: str = "bottom_center",
    bank_stride: float = 1.0,
    sta_clip_len: float = 2.0,
    batch_size: int = 32,
    num_frames: int = 16,
):
    """
    True single-pass streaming decoder + GPU extractor.
    Only active windows (spanning ~2-5 seconds) hold RGB frames in RAM.
    The moment `when > window.end`, the window is finalized, preprocessed, freed from RAM,
    and queued into the GPU batch buffer. Host memory stays < 1 GB on any video length.
    """
    import av
    import gc

    n_ev = len(vid_events)
    ar_windows = [observation(ev, "AR") for ev in vid_events]
    sta_windows = [observation(ev, "STA", context=sta_clip_len) for ev in vid_events]

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        origin = float((stream.start_time or 0) * stream.time_base)
        if stream.duration is not None and stream.time_base is not None:
            est_duration = float(stream.duration * stream.time_base)
        else:
            est_duration = 0.0

        max_event_t = max([w.end for w in ar_windows if not w.empty] + [est_duration, 1.0])
        upper_t = max(max_event_t, est_duration) + 10.0
        bank_ends = np.arange(bank_stride, upper_t + bank_stride, bank_stride, dtype=np.float64)
        bank_windows = [
            Window(max(0.0, float(t_end - sta_clip_len)), float(t_end), inclusive_end=True)
            for t_end in bank_ends
        ]

        all_windows = ar_windows + sta_windows + bank_windows
        n_ar = len(ar_windows)
        n_sta = len(sta_windows)
        n_bank_init = len(bank_windows)
        n_tot = len(all_windows)

        win_starts = np.array([w.start for w in all_windows], dtype=np.float64)
        win_ends = np.array([w.end for w in all_windows], dtype=np.float64)
        win_inc = np.array([w.inclusive_end for w in all_windows], dtype=bool)
        win_empty = np.array([w.empty or w.end < w.start or w.start < 0 for w in all_windows], dtype=bool)

        steps = (np.arange(num_frames, dtype=np.float64) + 0.5) / num_frames
        targets = win_starts[:, None] + (win_ends - win_starts)[:, None] * steps[None, :]
        best_dist = np.full((n_tot, num_frames), np.inf, dtype=np.float64)
        best_pts = np.full((n_tot, num_frames), -1.0, dtype=np.float64)
        best_rgb: dict[int, list[np.ndarray | None]] = {}

        ar_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
        sta_obs_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
        sta_pred_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
        ar_clip_end_s = np.full(n_ev, -1.0, dtype=np.float64)
        sta_clip_end_s = np.full(n_ev, -1.0, dtype=np.float64)
        sta_use_prior = np.ones(n_ev, dtype=bool)

        bank_arr = np.zeros((n_bank_init, 24, 1024), dtype=np.float16)
        bank_actual_ends = np.full(n_bank_init, -1.0, dtype=np.float64)

        enc_batch, enc_targets = [], []   # list of (kind, idx) where kind in {"ar", "bank"}
        sta_batch, sta_indices = [], []

        def flush_enc():
            nonlocal enc_batch, enc_targets
            if not enc_batch:
                return
            tok, _ = extractor.extract_batch(torch.stack(enc_batch), run_predictor=False)
            tok_np = tok.numpy()
            for k, (kind, idx) in enumerate(enc_targets):
                if kind == "ar":
                    ar_arr[idx] = tok_np[k]
                else:
                    bank_arr[idx] = tok_np[k]
            enc_batch, enc_targets = [], []

        def flush_sta():
            nonlocal sta_batch, sta_indices
            if not sta_batch:
                return
            tok_obs, tok_pred = extractor.extract_batch(torch.stack(sta_batch), run_predictor=True)
            sta_obs_arr[sta_indices] = tok_obs.numpy()
            sta_pred_arr[sta_indices] = tok_pred.numpy()
            sta_batch, sta_indices = [], []

        def complete_window(w_idx: int):
            slots = best_rgb.pop(w_idx, None)
            if slots is None or win_empty[w_idx] or not any(s is not None for s in slots):
                return
            pts_slots = best_pts[w_idx]
            first_valid_idx = next(i for i, s in enumerate(slots) if s is not None)
            last_arr = slots[first_valid_idx]
            last_t = float(pts_slots[first_valid_idx])
            H, W, _ = last_arr.shape
            out = np.empty((num_frames, H, W, 3), dtype=np.uint8)
            max_pts = -1.0
            for i in range(num_frames):
                if slots[i] is not None:
                    last_arr = slots[i]
                    last_t = float(pts_slots[i])
                out[i] = last_arr
                if last_t > max_pts:
                    max_pts = last_t
            if win_inc[w_idx] and max_pts > win_ends[w_idx] + 1e-5:
                raise AssertionError(f"PTS leakage: {max_pts} > {win_ends[w_idx]}")

            clip_t = preprocess_clip(out, img_size, crop_mode)
            del out, slots

            if w_idx < n_ar:
                ar_clip_end_s[w_idx] = max_pts
                enc_batch.append(clip_t)
                enc_targets.append(("ar", w_idx))
                if len(enc_batch) >= batch_size:
                    flush_enc()
            elif w_idx < n_ar + n_sta:
                ev_idx = w_idx - n_ar
                sta_use_prior[ev_idx] = False
                sta_clip_end_s[ev_idx] = max_pts
                sta_batch.append(clip_t)
                sta_indices.append(ev_idx)
                if len(sta_batch) >= batch_size:
                    flush_sta()
            else:
                b_idx = w_idx - n_ar - n_sta
                bank_actual_ends[b_idx] = max_pts
                enc_batch.append(clip_t)
                enc_targets.append(("bank", b_idx))
                if len(enc_batch) >= batch_size:
                    flush_enc()

        valid_w_indices = np.flatnonzero(~win_empty)
        order = valid_w_indices[np.argsort(win_starts[valid_w_indices], kind="stable")]
        n_order = len(order)
        ptr_activate = 0
        active: list[int] = []
        max_video_pts = -1.0

        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None or frame.is_corrupt:
                continue
            when = float(frame.pts * stream.time_base) - origin
            if when < 0.0:
                continue
            if when > max_video_pts:
                max_video_pts = when

            while ptr_activate < n_order and win_starts[order[ptr_activate]] <= when:
                w_new = int(order[ptr_activate])
                best_rgb[w_new] = [None] * num_frames
                active.append(w_new)
                ptr_activate += 1

            if not active:
                continue

            rgb_cache = None
            still_active = []
            for w_idx in active:
                hi = win_ends[w_idx]
                inc = win_inc[w_idx]
                if when > hi or (when == hi and not inc):
                    complete_window(w_idx)
                    continue
                still_active.append(w_idx)

                dist = np.abs(targets[w_idx] - when)
                improved = np.flatnonzero(dist < best_dist[w_idx])
                if len(improved) > 0:
                    if rgb_cache is None:
                        rgb_cache = frame.to_ndarray(format="rgb24")
                    slots = best_rgb[w_idx]
                    for slot in improved:
                        best_dist[w_idx, slot] = dist[slot]
                        best_pts[w_idx, slot] = when
                        slots[slot] = rgb_cache
            active = still_active

        # Finalize any windows still active at EOF
        for w_idx in active:
            complete_window(w_idx)
        flush_enc()
        flush_sta()

    best_rgb.clear()
    gc.collect()

    if n_ev > 0 and (ar_clip_end_s < 0).any():
        bad_idx = int(np.flatnonzero(ar_clip_end_s < 0)[0])
        raise RuntimeError(f"AR window empty for {vid_events[bad_idx].narration_id}")

    actual_limit = max(max_event_t, max_video_pts) + bank_stride
    keep_bank = int(np.sum(bank_ends <= actual_limit))
    return (
        ar_arr,
        sta_obs_arr,
        sta_pred_arr,
        ar_clip_end_s,
        sta_clip_end_s,
        sta_use_prior,
        bank_arr[:keep_bank],
        bank_actual_ends[:keep_bank],
    )


@torch.no_grad()
def pool_encoder_24_tokens(encoder_tokens: torch.Tensor, t_slices: int = 8, grid_size: int = 24) -> torch.Tensor:
    B, N, D = encoder_tokens.shape
    x = encoder_tokens.view(B, t_slices, grid_size, grid_size, D)
    e_temp = x.mean(dim=(2, 3))
    last_two = x[:, -2:, :, :, :].mean(dim=1).permute(0, 3, 1, 2)
    spat_4x4 = F.adaptive_avg_pool2d(last_two, (4, 4)).flatten(2).transpose(1, 2)
    return torch.cat([e_temp, spat_4x4], dim=1).to(torch.float16)


@torch.no_grad()
def pool_predicted_24_tokens(
    predicted_tokens: torch.Tensor,
    obs_24_tokens: torch.Tensor,
    grid_size: int = 24,
) -> torch.Tensor:
    B, N_pred, D = predicted_tokens.shape
    n_spat = grid_size * grid_size
    t_pred = max(1, N_pred // n_spat)
    x_pred = predicted_tokens.view(B, t_pred, grid_size, grid_size, D).float()

    pred_temp = x_pred.mean(dim=(2, 3))
    obs_temp = obs_24_tokens[:, :8, :].float()
    if t_pred < 8:
        merged_temp = torch.cat([obs_temp[:, t_pred:, :], pred_temp], dim=1)
    else:
        merged_temp = pred_temp[:, -8:, :]

    pred_spatial = x_pred.mean(dim=1).permute(0, 3, 1, 2)
    pred_4x4 = F.adaptive_avg_pool2d(pred_spatial, (4, 4)).flatten(2).transpose(1, 2)
    return torch.cat([merged_temp, pred_4x4], dim=1).to(torch.float16)


class VJEPA21Extractor(nn.Module):
    def __init__(self, vjepa_repo: Path, ckpt_path: Path, config_rel: str, img_size: int = 384, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.img_size = img_size
        self.frames_per_clip = 16
        self.fps = 8
        self.grid_size = img_size // 16
        self.t_slices = self.frames_per_clip // 2

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        cfg_path = Path(vjepa_repo) / config_rel
        with cfg_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.model = load_vjepa_model(
            vjepa_root=vjepa_repo,
            config=config,
            checkpoint=ckpt_path,
            frames_per_clip=self.frames_per_clip,
            frames_per_second=self.fps,
            resolution=img_size,
        ).to(self.device)

        for m in self.model.modules():
            if hasattr(m, "use_sdpa"):
                m.use_sdpa = False

    @torch.no_grad()
    def extract_batch(self, clips: torch.Tensor, run_predictor: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = clips.to(self.device, non_blocking=True)
        B = x.size(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
            if run_predictor:
                ant_times = torch.ones(B, dtype=torch.float32, device=self.device)
                enc_tokens, _, pred_tokens, _ = anticipate(self.model, x, ant_times)
                obs_24 = pool_encoder_24_tokens(enc_tokens.float(), self.t_slices, self.grid_size)
                pred_24 = pool_predicted_24_tokens(pred_tokens, obs_24, self.grid_size)
                return obs_24.cpu(), pred_24.cpu()
            else:
                enc_full = self.model.encoder(x)
                embed_dim = self.model.encoder.embed_dim
                enc_tokens = enc_full[:, :, -embed_dim:] if enc_full.size(-1) > embed_dim else enc_full
                obs_24 = pool_encoder_24_tokens(enc_tokens.float(), self.t_slices, self.grid_size)
                return obs_24.cpu(), None


def run_extract(args):
    out_root = Path(args.out_dir)
    shards_dir = out_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(Path(args.train_csv), labeled=True)
    if args.test_csv and Path(args.test_csv).is_file():
        events.extend(load_events(Path(args.test_csv), labeled=False))

    v_index = video_index(Path(args.video_root))
    by_video: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_video[e.video_id].append(e)

    all_videos = sorted(v_index.keys()) if args.include_all_disk_videos else sorted(by_video.keys())
    my_videos = all_videos[args.rank :: args.world_size]
    if args.max_videos > 0:
        my_videos = my_videos[: args.max_videos]

    print(f"[Rank {args.rank}/{args.world_size}] Assigned {len(my_videos)} / {len(all_videos)} videos.")
    extractor = VJEPA21Extractor(
        Path(args.vjepa_repo),
        Path(args.ckpt),
        config_rel=args.config_rel,
        img_size=args.img_size,
        device=args.device,
    )

    for vid in tqdm(my_videos, desc=f"Rank {args.rank}", disable=False):
        shard_path = shards_dir / f"{vid}.npz"
        if shard_path.exists() and not args.overwrite:
            continue

        video_path = v_index[vid]
        vid_events = sorted(by_video.get(vid, []), key=lambda e: e.start)
        narration_ids = np.array([e.narration_id for e in vid_events], dtype=str)
        action_start_s = np.array([e.start for e in vid_events], dtype=np.float64)
        action_stop_s = np.array([e.stop for e in vid_events], dtype=np.float64)

        (
            ar_arr,
            sta_obs_arr,
            sta_pred_arr,
            ar_clip_end_s,
            sta_clip_end_s,
            sta_use_prior,
            bank_arr,
            bank_actual_ends,
        ) = extract_video_streaming(
            video_path=video_path,
            vid_events=vid_events,
            extractor=extractor,
            img_size=args.img_size,
            crop_mode=args.crop_mode,
            bank_stride=args.bank_stride,
            sta_clip_len=args.sta_clip_len,
            batch_size=args.batch_size,
            num_frames=16,
        )

        tmp_shard = shards_dir / f".{vid}.{uuid.uuid4().hex}.tmp.npz"
        np.savez(
            tmp_shard,
            video_id=vid,
            narration_ids=narration_ids,
            action_start_s=action_start_s,
            action_stop_s=action_stop_s,
            ar_tokens=ar_arr,
            sta_obs_tokens=sta_obs_arr,
            sta_pred_tokens=sta_pred_arr,
            ar_clip_end_s=ar_clip_end_s,
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

    total_events = 0
    total_bank = 0
    shard_meta = []
    for sf in shard_files:
        with np.load(sf) as data:
            n_ev = int(len(data["narration_ids"]))
            n_bk = int(len(data["bank_t_end_s"]))
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

    for sf, n_ev, n_bk in shard_meta:
        with np.load(sf) as data:
            vid = str(data["video_id"])
            if n_ev > 0:
                ar_mmap[ev_cursor : ev_cursor + n_ev] = data["ar_tokens"]
                sta_obs_mmap[ev_cursor : ev_cursor + n_ev] = data["sta_obs_tokens"]
                sta_pred_mmap[ev_cursor : ev_cursor + n_ev] = data["sta_pred_tokens"]

                nids = data["narration_ids"]
                a_starts = data["action_start_s"] if "action_start_s" in data else np.zeros(n_ev)
                a_stops = data["action_stop_s"] if "action_stop_s" in data else np.zeros(n_ev)
                ar_ends = data["ar_clip_end_s"] if "ar_clip_end_s" in data else a_stops
                sta_ends = data["sta_clip_end_s"]
                priors = data["sta_use_prior"]
                for i in range(n_ev):
                    event_rows.append({
                        "narration_id": str(nids[i]),
                        "row": int(ev_cursor + i),
                        "video_id": vid,
                        "action_start_s": float(a_starts[i]),
                        "action_stop_s": float(a_stops[i]),
                        "ar_clip_end_s": float(ar_ends[i]),
                        "sta_clip_end_s": float(sta_ends[i]),
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
                        "row": int(bk_cursor + j),
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
        "schema_version": "1.0",
        "model_family": "V-JEPA 2.1",
        "backbone": "vjepa2_1_vitl_dist_vitG_384",
        "checkpoint": "vjepa2_1_vitl_dist_vitG_384.pt",
        "config": "configs/eval_2_1/vitl-384/ek100.yaml",
        "resolution": int(args.img_size),
        "img_size": int(args.img_size),
        "frames_per_clip": 16,
        "fps": 8,
        "clip_duration_s": 2.0,
        "bank_stride_s": float(args.bank_stride),
        "bank_stride_seconds": float(args.bank_stride),
        "bank_time_origin": "stream.start_time * stream.time_base",
        "frame_timestamp_policy": "nearest legal PTS in uniform bin, strictly <= cutoff for STA",
        "crop": str(args.crop_mode),
        "normalization": "ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])",
        "pooling_rule": "8 temporal tubelet spatial-means + 16 spatial 4x4 adaptive pool over last 2 tubelets",
        "tokens": 24,
        "dim": 1024,
        "embed_dim": 1024,
        "dtype": "float16",
        "predictor_output_space": "distilled teacher space (1664 raw -> 1024 last-channel slice via AnticipativeWrapper)",
        "total_events": int(total_events),
        "total_bank_steps": int(total_bank),
        "num_videos": int(len(shard_files)),
        "git_commit": str(git_hash),
        "consolidated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_safe(out_root / "meta.json", meta)
    print(f"Consolidation complete! Saved to {out_root}")


def run_dummy(args):
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    all_events = load_events(Path(args.train_csv), labeled=True)
    rng_py = np.random.default_rng(42)
    indices = sorted(rng_py.choice(len(all_events), size=min(args.num_events, len(all_events)), replace=False))
    events = [all_events[i] for i in indices]
    n_ev = len(events)

    for name in ("ar_tokens.npy", "sta_obs_tokens.npy", "sta_pred_tokens.npy"):
        arr = np.lib.format.open_memmap(out_root / name, mode="w+", dtype=np.float16, shape=(n_ev, 24, 1024))
        arr[:] = rng_py.standard_normal((n_ev, 24, 1024), dtype=np.float32).astype(np.float16)
        arr.flush()

    rows = []
    for i, e in enumerate(events):
        use_prior = e.sta_cutoff < 0
        rows.append({
            "narration_id": e.narration_id,
            "row": i,
            "video_id": e.video_id,
            "action_start_s": e.start,
            "action_stop_s": e.stop,
            "ar_clip_end_s": e.stop,
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
    p_ext.add_argument("--config-rel", default="configs/eval_2_1/vitl-384/ek100.yaml")
    p_ext.add_argument("--out-dir", required=True)
    p_ext.add_argument("--img-size", type=int, default=384)
    p_ext.add_argument("--crop-mode", choices=["bottom_center", "resize_square"], default="bottom_center")
    p_ext.add_argument("--sta-clip-len", type=float, default=2.0)
    p_ext.add_argument("--bank-stride", type=float, default=1.0)
    p_ext.add_argument("--batch-size", type=int, default=32)
    p_ext.add_argument("--rank", type=int, default=0)
    p_ext.add_argument("--world-size", type=int, default=1)
    p_ext.add_argument("--max-videos", type=int, default=0)
    p_ext.add_argument("--include-all-disk-videos", action="store_true")
    p_ext.add_argument("--device", default="cuda")
    p_ext.add_argument("--overwrite", action="store_true")

    p_con = sub.add_parser("consolidate")
    p_con.add_argument("--out-dir", required=True)
    p_con.add_argument("--img-size", type=int, default=384)
    p_con.add_argument("--crop-mode", choices=["bottom_center", "resize_square"], default="bottom_center")
    p_con.add_argument("--bank-stride", type=float, default=1.0)

    p_dum = sub.add_parser("dummy")
    p_dum.add_argument("--train-csv", required=True)
    p_dum.add_argument("--out-dir", required=True)
    p_dum.add_argument("--num-events", type=int, default=500)

    args = parser.parse_args()
    if args.cmd == "extract":
        run_extract(args)
    elif args.cmd == "consolidate":
        run_consolidate(args)
    elif args.cmd == "dummy":
        run_dummy(args)


if __name__ == "__main__":
    main()
