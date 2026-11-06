"""Member 1: checks that reject STA observations past the cutoff.

The video sampler already refuses frames after ``sta_observation_end_seconds``.
These checks extend that guarantee to everything built on top of it:

* manifests (a STA window must end exactly at start - 1 s),
* the consolidated feature index (``events_index.csv``) written by
  ``scripts/extract_vjepa21_tokens.py consolidate``,
* history rows a loader picks from the memory bank (``bank_index.csv``).

Every function raises ``LeakageError`` with the offending IDs instead of
silently dropping rows, so a bad feature bank can never train quietly.
"""
from __future__ import annotations
import math
import numpy as np

EPS = 1e-6


class LeakageError(AssertionError):
    pass


def _fail(message, offenders):
    offenders = list(offenders)
    if offenders:
        raise LeakageError(f"{message}: {len(offenders)} rows, first {offenders[:5]}")


def assert_times_before_cutoff(times, cutoff, what="frames"):
    """All observed timestamps must be <= cutoff. A negative cutoff allows nothing."""
    t = np.asarray(times, dtype=np.float64)
    t = t[np.isfinite(t)]
    if cutoff < 0 and t.size:
        raise LeakageError(f"{what}: cutoff {cutoff} < 0 means no history, got {t.size} timestamps")
    if t.size and t.max() > cutoff + EPS:
        raise LeakageError(f"{what}: timestamp {t.max():.6f}s is after the STA cutoff {cutoff:.6f}s")


def check_manifest_rows(rows):
    """STA rows must observe [.., start-1]; AR rows are never used for STA."""
    bad = []
    for r in rows:
        if r.get("task") != "STA":
            continue
        cutoff, w = r["sta_cutoff"], r["observation"]
        if abs(cutoff - (r["start"] - 1.0)) > EPS or w["end"] > cutoff + EPS:
            bad.append(r["narration_id"])
        elif cutoff < 0 and not w["empty"]:
            bad.append(r["narration_id"])
    _fail("STA manifest windows end after the cutoff", bad)


def check_event_features(events_index, events, require_all=True):
    """Validate events_index.csv against the annotations.

    events_index: DataFrame with narration_id, row, sta_clip_end_s, sta_use_prior.
    events: iterable of Event (train and/or test) giving the official cutoffs.
    """
    cutoffs = {e.narration_id: e.sta_cutoff for e in events}
    idx = events_index
    dupes = idx["narration_id"][idx["narration_id"].duplicated()].tolist()
    _fail("Duplicate narration_id in feature index", dupes)
    _fail("Duplicate feature row", idx["row"][idx["row"].duplicated()].tolist())
    unknown = [n for n in idx["narration_id"] if n not in cutoffs]
    _fail("Feature rows for unknown events", unknown)
    if require_all:
        _fail("Events without features", sorted(set(cutoffs) - set(idx["narration_id"])))
    late, prior_wrong, missing_history = [], [], []
    for nid, end, use_prior in zip(idx["narration_id"], idx["sta_clip_end_s"], idx["sta_use_prior"]):
        cutoff = cutoffs[nid]
        use_prior = str(use_prior).lower() in {"true", "1"}
        if cutoff < 0:
            if not use_prior or float(end) >= 0:
                prior_wrong.append(nid)
        elif use_prior:
            missing_history.append(nid)  # history exists but the extractor found no frame
        elif not math.isfinite(float(end)) or float(end) > cutoff + EPS or float(end) < 0:
            late.append(nid)
    _fail("STA feature clip ends after the cutoff", late)
    _fail("Negative-cutoff events must use the prior and have clip end -1", prior_wrong)
    _fail("Events with legal history were marked use_prior (missing frames?)", missing_history)


def history_rows(bank_video, cutoff, steps=16, spacing=1.0):
    """Pick memory-bank rows for one STA event without reading the future.

    bank_video: DataFrame for ONE video with columns t_end_s and row.
    For k = 1..steps the anchor is tau_k = cutoff - (steps - k) * spacing and we
    take the last bank clip whose end is <= tau_k. Anchors before the first bank
    clip return row -1 (pad). Returns (rows, clip_end_times, anchors).
    """
    anchors = np.array([cutoff - (steps - k) * spacing for k in range(1, steps + 1)])
    b = bank_video[bank_video["t_end_s"] >= 0].sort_values("t_end_s")
    ends, rows = b["t_end_s"].to_numpy(np.float64), b["row"].to_numpy(np.int64)
    pos = np.searchsorted(ends, anchors + EPS, side="right") - 1
    picked_rows = np.where(pos >= 0, rows[np.clip(pos, 0, None)], -1)
    picked_ends = np.where(pos >= 0, ends[np.clip(pos, 0, None)], np.nan)
    if cutoff < 0:
        picked_rows[:] = -1
        picked_ends[:] = np.nan
    assert_times_before_cutoff(picked_ends, cutoff, "memory bank history")
    if np.any(picked_ends > anchors + EPS):
        raise LeakageError("A history step uses a clip ending after its own anchor")
    return picked_rows, picked_ends, anchors


def check_batch(clip_end_times, cutoffs):
    """Cheap guard for a training loop: every [B, S] history time <= its event cutoff."""
    t = np.asarray(clip_end_times, dtype=np.float64)
    c = np.asarray(cutoffs, dtype=np.float64).reshape(-1, *([1] * (t.ndim - 1)))
    late = np.isfinite(t) & (t > c + EPS)
    if late.any():
        raise LeakageError(f"{int(late.sum())} history entries after their STA cutoff")
