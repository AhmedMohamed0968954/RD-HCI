"""Feature bank: the handoff between feature extraction (Member 2) and STA memory (Member 3).

Training reads the team's consolidated layout (one folder per feature version):

    team/features/<version>/
      meta.json            backbone, resolution, frames/fps, crop, stride, git commit
      bank_tokens.npy      [M, T, D] float16  dense memory bank (all videos)
      bank_index.csv       video_id, t_end_s, row      (t_end_s = clip end time, seconds)
      sta_obs_tokens.npy   [N_events, T, D]   exact clip ending at each event's STA cutoff
      events_index.csv     narration_id, row, video_id, sta_clip_end_s (+ ar_clip_end_s)

Arrays are opened with mmap_mode="r" lazily, after DataLoader workers start, so each sample
reads only its own rows and workers share the OS page cache. A clip is legal for an STA event
with cutoff c only if its end time <= c; both readers check this.
`make_mock_bank` builds an in-memory bank, `write_mock_consolidated` writes the same layout
with random features, so everything can be tested before real extraction exists.
"""
from __future__ import annotations
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

EPS = 1e-6


def cache_path(cache_dir, root: Path, array_name: str) -> Path | None:
    """Cache file for the token mean of root/array_name, keyed by the source file's size and
    modification time, so re-extracted or re-consolidated features never reuse a stale cache."""
    if not cache_dir:
        return None
    st = (root / array_name).stat()
    return Path(cache_dir) / root.name / f"{Path(array_name).stem}_mean_{st.st_size}_{st.st_mtime_ns}.npy"


def mean_tokens(array, cache: Path | None = None, chunk: int = 2048) -> np.ndarray:
    """[M,T,D] -> [M,1,D] float16 by averaging tokens, streamed in chunks (never all in RAM).
    With `cache`, the result is saved there once (atomic rename) and reused by later runs.
    """
    M, _, D = array.shape
    if cache is not None and cache.is_file():
        cached = np.load(cache)
        if cached.shape == (M, 1, D):
            return cached
    out = np.empty((M, 1, D), dtype=np.float16)
    for i in range(0, M, chunk):
        out[i:i + chunk, 0] = np.asarray(array[i:i + chunk], dtype=np.float32).mean(axis=1)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, out)
        os.replace(tmp, cache)
    return out


class RowView:
    """Rows of one video inside a [M,T,D] array; behaves like a [N,T,D] tensor for indexing."""

    def __init__(self, array, rows):
        self.array, self.rows = array, rows
        self.shape = (len(rows),) + tuple(array.shape[1:])

    def __getitem__(self, i):
        return torch.from_numpy(np.array(self.array[self.rows[i]]))


def _check_times(video_id, times):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(f"{video_id}: clip end times must be a finite 1-D array")
    if len(times) > 1 and not (np.diff(times) > 0).all():
        raise ValueError(f"{video_id}: clip end times must be strictly increasing (duplicates?)")
    return times


class FeatureBank:
    """video_id -> {"times": [N], "feats": [N,T,D]}.

    FeatureBank(entries={...})   in-memory (mock data, unit tests)
    FeatureBank(root=folder)     consolidated layout above, memory-mapped
    pooled=True averages the T tokens of every clip once ([M,T,D] -> [M,1,D]) and keeps the
    result in RAM (~1 GB for the full bank). The current model averages tokens anyway, so this
    gives identical outputs while avoiding re-reading ~25 GB from Lustre every epoch.
    Use pooled=False when a token-level probe replaces the mean pooling.
    """

    def __init__(self, entries: dict | None = None, root: str | Path | None = None,
                 pooled: bool = False, cache_dir: str | Path | None = None):
        if (entries is None) == (root is None):
            raise ValueError("Give either in-memory entries or a consolidated features folder")
        self.root = Path(root) if root is not None else None
        self.pooled = pooled
        self._array = None
        self._full = None            # memmap of all T tokens (consolidated) when pooled
        self._full_entries = {}      # all T tokens (in-memory) when pooled
        self.entries, self.index, self.dropped = {}, {}, {}
        if entries is not None:
            for video_id, e in entries.items():
                feats = e["feats"]
                if not isinstance(feats, torch.Tensor) or feats.ndim != 3:
                    raise ValueError(f"{video_id}: feats must be a tensor [N,T,D]")
                times = _check_times(video_id, e["times"])
                if feats.shape[0] != len(times):
                    raise ValueError(f"{video_id}: feats and times differ in length")
                if pooled:
                    self._full_entries[video_id] = feats
                    feats = feats.float().mean(dim=1, keepdim=True).half()
                self.entries[video_id] = {"times": times, "feats": feats}
            shapes = {tuple(e["feats"].shape[1:]) for e in self.entries.values()}
        else:
            rows_by_video = defaultdict(list)
            self.dropped = {"undecodable": 0, "duplicate_time": 0}
            with (self.root / "bank_index.csv").open(newline="") as f:
                for r in csv.DictReader(f):
                    t = float(r["t_end_s"])
                    if not math.isfinite(t) or t < 0:        # extractor marks undecodable clips with -1
                        self.dropped["undecodable"] += 1
                        continue
                    rows_by_video[r["video_id"]].append((t, int(r["row"])))
            for video_id, pairs in rows_by_video.items():
                pairs.sort()
                kept = [pairs[0]]
                for t, row in pairs[1:]:                     # same last frame twice (video end): keep one
                    if t - kept[-1][0] <= EPS:
                        self.dropped["duplicate_time"] += 1
                        continue
                    kept.append((t, row))
                times = _check_times(video_id, [t for t, _ in kept])
                self.index[video_id] = (times, np.array([r for _, r in kept], dtype=np.int64))
            header = np.load(self.root / "bank_tokens.npy", mmap_mode="r")
            n_rows = header.shape[0]
            if pooled:
                self._array = mean_tokens(header, cache_path(cache_dir, self.root, "bank_tokens.npy"))
            shapes = {tuple(self._array.shape[1:]) if pooled else tuple(header.shape[1:])}
            self.token_shape = tuple(header.shape[1:])
            del header
            used = np.concatenate([rows for _, rows in self.index.values()]) if self.index else np.array([])
            if len(used) and (used.min() < 0 or used.max() >= n_rows):
                raise ValueError("bank_index.csv points outside bank_tokens.npy")
        if len(shapes) > 1:
            raise ValueError(f"All videos must share [T,D]; found {shapes}")
        self.shape = next(iter(shapes)) if shapes else None
        if entries is not None:
            if self._full_entries:
                self.token_shape = tuple(next(iter(self._full_entries.values())).shape[1:])
            else:
                self.token_shape = self.shape
        meta = self.root / "meta.json" if self.root else None
        self.meta = json.loads(meta.read_text()) if meta and meta.is_file() else {}

    def __getstate__(self):                       # never send an open memmap to workers
        state = dict(self.__dict__)
        if not self.pooled:
            state["_array"] = None
        state["_full"] = None
        return state

    def full_tokens(self, video_id: str, position: int) -> torch.Tensor:
        """All T tokens [T,D] of the clip at `position` in this video's (time-sorted) bank,
        even when the bank itself is pooled. Used as the present clip when the exact STA clip
        is missing (e.g. hidden-test events that were not extracted)."""
        if video_id in self._full_entries:
            return self._full_entries[video_id][position]
        if video_id in self.entries:
            return self.entries[video_id]["feats"][position]
        _, rows = self.index[video_id]
        if self._full is None:
            self._full = np.load(self.root / "bank_tokens.npy", mmap_mode="r")
        return torch.from_numpy(np.array(self._full[rows[position]]))

    def __contains__(self, video_id):
        return video_id in self.entries or video_id in self.index

    def __getitem__(self, video_id):
        if video_id in self.entries:
            return self.entries[video_id]
        if video_id not in self.index:
            raise KeyError(f"No features for {video_id}")
        if self._array is None:
            self._array = np.load(self.root / "bank_tokens.npy", mmap_mode="r")
        times, rows = self.index[video_id]
        return {"times": times, "feats": RowView(self._array, rows)}


class Anchors:
    """Exact clip ending at each event's STA cutoff (sta_obs_tokens.npy + events_index.csv)."""

    def __init__(self, root: str | Path, pooled: bool = False, cache_dir: str | Path | None = None):
        self.root, self._array, self.rows, self.pooled = Path(root), None, {}, pooled
        with (self.root / "events_index.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            if "sta_clip_end_s" not in (reader.fieldnames or []):
                # Team format contract: an ambiguous legacy clip_end_s (AR stop or STA cutoff?)
                # is never reinterpreted silently.
                raise ValueError("events_index.csv needs an sta_clip_end_s column "
                                 "(see docs/MEMBER4_FORMAT_CONTRACT.md)")
            has_flag = "sta_use_prior" in reader.fieldnames
            for r in reader:
                end = float(r["sta_clip_end_s"])
                no_clip = end < 0 or (has_flag and r["sta_use_prior"].strip().lower() in {"true", "1"})
                # Events without any observable frame keep a zero block in the array: never use it.
                self.rows[r["narration_id"]] = None if no_clip else (int(r["row"]), end)
        if pooled:
            self._array = mean_tokens(np.load(self.root / "sta_obs_tokens.npy", mmap_mode="r"),
                                      cache_path(cache_dir, self.root, "sta_obs_tokens.npy"))

    def __getstate__(self):
        state = dict(self.__dict__)
        if not self.pooled:
            state["_array"] = None
        return state

    def for_event(self, event):
        """Tokens [T,D] for this event, or None. Raises if the clip ends after the cutoff."""
        if event.sta_cutoff < 0 or self.rows.get(event.narration_id) is None:
            return None
        row, end = self.rows[event.narration_id]
        if not math.isfinite(end) or end > event.sta_cutoff + EPS:
            raise AssertionError(f"Leakage: STA clip for {event.narration_id} ends at {end} "
                                 f"> cutoff {event.sta_cutoff}")
        if self._array is None:
            self._array = np.load(self.root / "sta_obs_tokens.npy", mmap_mode="r")
        return torch.from_numpy(np.array(self._array[row]))


def make_mock_bank(events, stride=1.0, tokens=24, dim=32, seed=0, encode_time=True) -> FeatureBank:
    """In-memory random bank on a regular grid. With encode_time, feats[i,:,0] = clip end time."""
    rng = torch.Generator().manual_seed(seed)
    end_by_video = {}
    for e in events:
        end_by_video[e.video_id] = max(end_by_video.get(e.video_id, 0.0), e.stop)
    entries = {}
    for video_id, end in sorted(end_by_video.items()):
        times = np.arange(stride, end + stride, stride, dtype=np.float64)
        feats = torch.randn(len(times), tokens, dim, generator=rng)
        if encode_time:
            feats[:, :, 0] = torch.from_numpy(times).float()[:, None]
        entries[video_id] = {"times": times, "feats": feats.half()}
    return FeatureBank(entries)


def write_mock_consolidated(events, root, stride=1.0, tokens=24, dim=32, seed=0):
    """Write the consolidated layout with random features (encode_time as in make_mock_bank)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bank = make_mock_bank(events, stride, tokens, dim, seed)
    total = sum(len(e["times"]) for e in bank.entries.values())
    arr = np.lib.format.open_memmap(root / "bank_tokens.npy", mode="w+", dtype=np.float16,
                                    shape=(total, tokens, dim))
    row = 0
    with (root / "bank_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "t_end_s", "row"])
        for video_id, e in bank.entries.items():
            n = len(e["times"])
            arr[row:row + n] = e["feats"].numpy()
            for i, t in enumerate(e["times"]):
                w.writerow([video_id, f"{t:.6f}", row + i])
            row += n
    arr.flush()
    del arr
    obs = np.lib.format.open_memmap(root / "sta_obs_tokens.npy", mode="w+", dtype=np.float16,
                                    shape=(len(events), tokens, dim))
    with (root / "events_index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["narration_id", "row", "video_id", "ar_clip_end_s", "sta_clip_end_s", "sta_use_prior"])
        for i, e in enumerate(events):
            obs[i] = np.random.default_rng(seed + i).standard_normal((tokens, dim)).astype(np.float16)
            obs[i, :, 0] = e.sta_cutoff
            cold = e.sta_cutoff < 0
            w.writerow([e.narration_id, i, e.video_id, f"{e.stop:.6f}",
                        "-1.0" if cold else f"{e.sta_cutoff:.6f}", cold])
    obs.flush()
    del obs
    (root / "meta.json").write_text(json.dumps({"backbone": "MOCK", "bank_stride_s": stride,
                                                "tokens": tokens, "dim": dim}, indent=2))
    return root
