"""
Non-destructive Phase 2 extraction:
1. Exports sta_pred17_1664.npy ([67217, 17, 1664] float16) aligned 1-to-1 with existing events_index.csv.
2. Extracts hidden-test anchors from manifests/hybrid-v3/test_sta.jsonl into separate *_test*.npy files
   so existing train/val feature files and Member 3's bank hashes remain untouched.
"""
from __future__ import annotations
import argparse
import gc
import json
import json.encoder
json.encoder.c_make_encoder = None
from pathlib import Path
import sys
import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.data.manifests import read_manifest
from epic_starter.data.events import Event, Window, load_events, observation
from epic_starter.models.future_latent import anticipate
from scripts.extract_vjepa21_tokens import (
    VJEPA21Extractor,
    preprocess_clip,
    pool_encoder_24_tokens,
    pool_predicted_24_tokens,
)


@torch.no_grad()
def pool_predicted_17_tokens(predicted_full: torch.Tensor, grid_size: int = 24) -> torch.Tensor:
    """Pool raw ViT-G teacher predictor output [B, N_pred, 1664] -> [B, 17, 1664] float16."""
    B, N_pred, D = predicted_full.shape
    n_spat = grid_size * grid_size
    t_pred = max(1, N_pred // n_spat)
    x_pred = predicted_full.view(B, t_pred, grid_size, grid_size, D).float()
    pred_temp = x_pred.mean(dim=(1, 2, 3)).unsqueeze(1)  # [B, 1, 1664]
    pred_spatial = x_pred.mean(dim=1).permute(0, 3, 1, 2)  # [B, 1664, H, W]
    pred_4x4 = F.adaptive_avg_pool2d(pred_spatial, (4, 4)).flatten(2).transpose(1, 2)  # [B, 16, 1664]
    return torch.cat([pred_temp, pred_4x4], dim=1).to(torch.float16)


@torch.no_grad()
def extract_batch_with_1664(extractor: VJEPA21Extractor, clips: torch.Tensor, run_predictor: bool):
    x = clips.to(extractor.device, non_blocking=True)
    B = x.size(0)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(extractor.device.type == "cuda")):
        if run_predictor:
            ant_times = torch.ones(B, dtype=torch.float32, device=extractor.device)
            enc_tokens, _, pred_tokens, pred_full = anticipate(extractor.model, x, ant_times)
            obs_24 = pool_encoder_24_tokens(enc_tokens.float(), extractor.t_slices, extractor.grid_size)
            pred_24 = pool_predicted_24_tokens(pred_tokens, obs_24, extractor.grid_size)
            pred_17 = pool_predicted_17_tokens(pred_full, extractor.grid_size)
            return obs_24.cpu(), pred_24.cpu(), pred_17.cpu()
        else:
            enc_full = extractor.model.encoder(x)
            embed_dim = extractor.model.encoder.embed_dim
            enc_tokens = enc_full[:, :, -embed_dim:] if enc_full.size(-1) > embed_dim else enc_full
            obs_24 = pool_encoder_24_tokens(enc_tokens.float(), extractor.t_slices, extractor.grid_size)
            return obs_24.cpu(), None, None


def extract_video_events(
    video_path: Path,
    ev_list: list[tuple[Event, bool]],  # (Event, is_test)
    extractor: VJEPA21Extractor,
    img_size: int = 384,
    sta_clip_len: float = 2.0,
    batch_size: int = 32,
    num_frames: int = 16,
):
    """Single-pass streaming decode. Train/val events run STA only; test events run AR + STA."""
    n_ev = len(ev_list)
    windows = []
    win_meta = []  # (ev_idx, kind) where kind in {"ar", "sta"}

    for idx, (ev, is_test) in enumerate(ev_list):
        if is_test:
            windows.append(observation(ev, "AR"))
            win_meta.append((idx, "ar"))
        windows.append(observation(ev, "STA", context=sta_clip_len))
        win_meta.append((idx, "sta"))

    n_tot = len(windows)
    win_starts = np.array([w.start for w in windows], dtype=np.float64)
    win_ends = np.array([w.end for w in windows], dtype=np.float64)
    win_inc = np.array([w.inclusive_end for w in windows], dtype=bool)
    win_empty = np.array([w.empty or w.end < w.start or w.start < 0 for w in windows], dtype=bool)

    steps = (np.arange(num_frames, dtype=np.float64) + 0.5) / num_frames
    targets = win_starts[:, None] + (win_ends - win_starts)[:, None] * steps[None, :]
    best_dist = np.full((n_tot, num_frames), np.inf, dtype=np.float64)
    best_pts = np.full((n_tot, num_frames), -1.0, dtype=np.float64)
    best_rgb: dict[int, list[np.ndarray | None]] = {}

    ar_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
    sta_obs_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
    sta_pred_arr = np.zeros((n_ev, 24, 1024), dtype=np.float16)
    sta_pred17_arr = np.zeros((n_ev, 17, 1664), dtype=np.float16)
    ar_clip_end_s = np.full(n_ev, -1.0, dtype=np.float64)
    sta_clip_end_s = np.full(n_ev, -1.0, dtype=np.float64)
    sta_use_prior = np.ones(n_ev, dtype=bool)

    ar_batch, ar_indices = [], []
    sta_batch, sta_indices = [], []

    def flush_ar():
        nonlocal ar_batch, ar_indices
        if not ar_batch:
            return
        tok, _, _ = extract_batch_with_1664(extractor, torch.stack(ar_batch), run_predictor=False)
        ar_arr[ar_indices] = tok.numpy()
        ar_batch, ar_indices = [], []

    def flush_sta():
        nonlocal sta_batch, sta_indices
        if not sta_batch:
            return
        tok_obs, tok_pred, tok_p17 = extract_batch_with_1664(extractor, torch.stack(sta_batch), run_predictor=True)
        sta_obs_arr[sta_indices] = tok_obs.numpy()
        sta_pred_arr[sta_indices] = tok_pred.numpy()
        sta_pred17_arr[sta_indices] = tok_p17.numpy()
        sta_batch, sta_indices = [], []

    def finalize_window(wi: int):
        frames = best_rgb.pop(wi, None)
        if frames is None:
            return
        valid = [f for f in frames if f is not None]
        if not valid:
            return
        filled = []
        last = valid[0]
        for f in frames:
            if f is not None:
                last = f
            filled.append(last)
        clip = preprocess_clip(np.stack(filled, axis=0), target_size=img_size, crop_mode="bottom_center")
        ev_idx, kind = win_meta[wi]
        max_pts = float(best_pts[wi].max())
        if kind == "ar":
            ar_clip_end_s[ev_idx] = max_pts
            ar_batch.append(clip)
            ar_indices.append(ev_idx)
            if len(ar_batch) >= batch_size:
                flush_ar()
        else:
            sta_clip_end_s[ev_idx] = max_pts
            sta_use_prior[ev_idx] = False
            sta_batch.append(clip)
            sta_indices.append(ev_idx)
            if len(sta_batch) >= batch_size:
                flush_sta()

    order = np.argsort(win_starts)
    ptr = 0
    active: list[int] = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 4
        origin = float((stream.start_time or 0) * stream.time_base)

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            when = float(frame.pts * frame.time_base) - origin

            while ptr < n_tot and win_starts[order[ptr]] <= when + 1e-6:
                wi = int(order[ptr])
                ptr += 1
                if not win_empty[wi]:
                    active.append(wi)
                    best_rgb[wi] = [None] * num_frames

            rgb = None
            still_active = []
            for wi in active:
                end_t = win_ends[wi]
                in_win = (when <= end_t + 1e-6) if win_inc[wi] else (when < end_t - 1e-6)
                if in_win:
                    dists = np.abs(targets[wi] - when)
                    improved = dists < best_dist[wi]
                    if np.any(improved):
                        if rgb is None:
                            rgb = frame.to_ndarray(format="rgb24")
                        for fi in np.flatnonzero(improved):
                            best_dist[wi, fi] = dists[fi]
                            best_pts[wi, fi] = when
                            best_rgb[wi][fi] = rgb
                    still_active.append(wi)
                else:
                    finalize_window(wi)
            active = still_active

        for wi in active:
            finalize_window(wi)

    flush_ar()
    flush_sta()
    gc.collect()
    return {
        "narration_id": np.array([ev.narration_id for ev, _ in ev_list]),
        "is_test": np.array([is_test for _, is_test in ev_list], dtype=bool),
        "ar_tokens": ar_arr,
        "sta_obs_tokens": sta_obs_arr,
        "sta_pred_tokens": sta_pred_arr,
        "sta_pred17_1664": sta_pred17_arr,
        "ar_clip_end_s": ar_clip_end_s,
        "sta_clip_end_s": sta_clip_end_s,
        "sta_use_prior": sta_use_prior,
    }


def run_extract(args):
    out_root = Path(args.out_dir)
    shard_dir = out_root / "shards_pred1664_test"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Load train/val events from existing events_index.csv + manifest JSONLs
    m_dir = Path(args.manifest_dir)
    by_video: dict[str, list[tuple[Event, bool]]] = {}

    for fname, is_test in [("train_sta.jsonl", False), ("val_sta.jsonl", False), ("test_sta.jsonl", True)]:
        for r in read_manifest(m_dir / fname):
            ev = Event(
                narration_id=r["narration_id"],
                participant_id=r["participant_id"],
                video_id=r["video_id"],
                start=float(r["start"]),
                stop=float(r["stop"]),
                sta_cutoff=float(r["sta_cutoff"]),
                verb=r.get("verb"),
                noun=r.get("noun"),
            )
            by_video.setdefault(ev.video_id, []).append((ev, is_test))

    video_ids = sorted(by_video.keys())
    my_videos = [v for i, v in enumerate(video_ids) if i % args.world_size == args.rank]
    print(f"[Rank {args.rank}/{args.world_size}] Processing {len(my_videos)}/{len(video_ids)} videos")

    extractor = VJEPA21Extractor(
        vjepa_repo=Path(args.vjepa_repo),
        ckpt_path=Path(args.ckpt),
        config_rel="configs/eval_2_1/vitl-384/ek100.yaml",
        img_size=args.img_size,
        device=args.device,
    )

    video_root = Path(args.video_root)
    for idx, vid in enumerate(my_videos, 1):
        shard_path = shard_dir / f"{vid}.npz"
        if shard_path.is_file():
            continue
        p_id = vid.split("_")[0]
        vpath = video_root / p_id / f"{vid}.MP4"
        if not vpath.is_file():
            vpath = video_root / p_id / f"{vid}.mp4"
        if not vpath.is_file():
            raise FileNotFoundError(f"Missing video: {vpath}")
        res = extract_video_events(vpath, by_video[vid], extractor, img_size=args.img_size, batch_size=args.batch_size)
        np.savez(shard_path, **res)
        if idx % 10 == 0 or idx == len(my_videos):
            print(f"[Rank {args.rank}] {idx}/{len(my_videos)} done (last={vid}, events={len(by_video[vid])})", flush=True)


def run_consolidate(args):
    out_root = Path(args.out_dir)
    shard_dir = out_root / "shards_pred1664_test"
    m_dir = Path(args.manifest_dir)

    ev_idx = pd.read_csv(out_root / "events_index.csv")
    row_map = dict(zip(ev_idx["narration_id"], ev_idx["row"]))
    n_trainval = len(ev_idx)
    p17_tv = np.zeros((n_trainval, 17, 1664), dtype=np.float16)
    filled_tv = np.zeros(n_trainval, dtype=bool)

    test_rows = read_manifest(m_dir / "test_sta.jsonl")
    test_map = {r["narration_id"]: i for i, r in enumerate(test_rows)}
    n_test = len(test_rows)
    ar_test = np.zeros((n_test, 24, 1024), dtype=np.float16)
    obs_test = np.zeros((n_test, 24, 1024), dtype=np.float16)
    pred_test = np.zeros((n_test, 24, 1024), dtype=np.float16)
    p17_test = np.zeros((n_test, 17, 1664), dtype=np.float16)
    test_ar_end = np.full(n_test, -1.0, dtype=np.float64)
    test_sta_end = np.full(n_test, -1.0, dtype=np.float64)
    test_use_prior = np.ones(n_test, dtype=bool)
    filled_test = np.zeros(n_test, dtype=bool)

    shards = sorted(shard_dir.glob("*.npz"))
    print(f"Consolidating {len(shards)} shards in RAM...", flush=True)
    for s_idx, sp in enumerate(shards, 1):
        with np.load(sp, allow_pickle=False) as d:
            nids = d["narration_id"]
            is_test = d["is_test"]
            p17 = d["sta_pred17_1664"]
            has_test = bool(np.any(is_test))
            if has_test:
                ar_a = d["ar_tokens"]
                obs_a = d["sta_obs_tokens"]
                pred_a = d["sta_pred_tokens"]
                ar_e = d["ar_clip_end_s"]
                sta_e = d["sta_clip_end_s"]
                use_p = d["sta_use_prior"]

        tv_mask = ~is_test
        if np.any(tv_mask):
            tv_local = np.flatnonzero(tv_mask)
            tv_global = np.array([row_map[str(nids[i])] for i in tv_local], dtype=np.int64)
            p17_tv[tv_global] = p17[tv_local]
            filled_tv[tv_global] = True

        if has_test:
            te_local = np.flatnonzero(is_test)
            te_global = np.array([test_map[str(nids[i])] for i in te_local], dtype=np.int64)
            ar_test[te_global] = ar_a[te_local]
            obs_test[te_global] = obs_a[te_local]
            pred_test[te_global] = pred_a[te_local]
            p17_test[te_global] = p17[te_local]
            test_ar_end[te_global] = ar_e[te_local]
            test_sta_end[te_global] = sta_e[te_local]
            test_use_prior[te_global] = use_p[te_local]
            filled_test[te_global] = True

        if s_idx % 100 == 0 or s_idx == len(shards):
            print(f"  [{s_idx}/{len(shards)}] tv={filled_tv.sum()}/{n_trainval} | test={filled_test.sum()}/{n_test}", flush=True)

    assert filled_tv.all(), f"Missing {n_trainval - filled_tv.sum()} train/val events"
    assert filled_test.all(), f"Missing {n_test - filled_test.sum()} test events"

    print("Writing consolidated .npy files...", flush=True)
    np.save(out_root / "sta_pred17_1664.npy", p17_tv)
    np.save(out_root / "ar_test_tokens.npy", ar_test)
    np.save(out_root / "sta_obs_test_tokens.npy", obs_test)
    np.save(out_root / "sta_pred_test_tokens.npy", pred_test)
    np.save(out_root / "sta_pred17_1664_test.npy", p17_test)

    pd.DataFrame({
        "row": np.arange(n_test),
        "narration_id": [r["narration_id"] for r in test_rows],
        "video_id": [r["video_id"] for r in test_rows],
        "ar_clip_end_s": test_ar_end,
        "sta_clip_end_s": test_sta_end,
        "sta_use_prior": test_use_prior,
    }).to_csv(out_root / "test_events_index.csv", index=False)
    print(f"Done! Train/val sta_pred17_1664 filled: {filled_tv.sum()}/{n_trainval} | Test filled: {filled_test.sum()}/{n_test}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--manifest-dir", required=True)
    p_ext.add_argument("--video-root", required=True)
    p_ext.add_argument("--vjepa-repo", required=True)
    p_ext.add_argument("--ckpt", required=True)
    p_ext.add_argument("--out-dir", required=True)
    p_ext.add_argument("--img-size", type=int, default=384)
    p_ext.add_argument("--batch-size", type=int, default=32)
    p_ext.add_argument("--rank", type=int, default=0)
    p_ext.add_argument("--world-size", type=int, default=8)
    p_ext.add_argument("--device", default="cuda")

    p_con = sub.add_parser("consolidate")
    p_con.add_argument("--manifest-dir", required=True)
    p_con.add_argument("--out-dir", required=True)

    args = p.parse_args()
    if args.cmd == "extract":
        run_extract(args)
    else:
        run_consolidate(args)


if __name__ == "__main__":
    main()
