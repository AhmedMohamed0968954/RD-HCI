"""STA history loader (Member 3): S memory steps ending exactly at the STA cutoff.

For an event with cutoff c = start - 1 s, step k (k = 1..S) is placed at
    tau_k = c - (S - k) * stride
so the last step lands on c. Each step takes the latest bank clip whose end time is
<= tau_k ("snap down"). Snapping down means no step ever reads video after its own
tau, and therefore never after c. Steps are invalid (masked) when tau_k < 0, when no
clip ends at or before tau_k, or when the nearest one is older than `tolerance`.
An optional exact `anchor` clip (ending at c) replaces the last step.
"""
from __future__ import annotations
from bisect import bisect_left
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

from epic_starter.data.events import Event
from .bank import FeatureBank

EPS = 1e-6


def step_times(cutoff: float, steps: int = 16, stride: float = 1.0) -> np.ndarray:
    if steps < 1 or not stride > 0:
        raise ValueError("Need steps >= 1 and stride > 0")
    return cutoff - (steps - 1 - np.arange(steps)) * stride


def build_history(event: Event, entry: dict | None, steps=16, stride=1.0, tolerance=None,
                  anchor: torch.Tensor | None = None, shape=None) -> dict:
    """Return feats [S,T,D] (zeros where invalid), valid [S], step_times [S], used_times [S]."""
    taus = step_times(event.sta_cutoff, steps, stride)
    tolerance = stride if tolerance is None else tolerance
    if entry is None and shape is None:
        raise ValueError("Need a bank entry or an explicit [T,D] shape")
    T, D = shape if entry is None else entry["feats"].shape[1:]
    feats = torch.zeros(steps, T, D, dtype=torch.float32)
    valid = np.zeros(steps, dtype=bool)
    used = np.full(steps, np.nan)
    if entry is not None:
        times = entry["times"]
        for k, tau in enumerate(taus):
            if tau < 0:
                continue
            i = int(np.searchsorted(times, tau + EPS, side="right")) - 1
            if i < 0 or tau - times[i] > tolerance + EPS:
                continue
            feats[k] = entry["feats"][i].float()
            valid[k], used[k] = True, times[i]
    if anchor is not None and event.sta_cutoff >= 0:
        if tuple(anchor.shape) != (T, D):
            raise ValueError("Anchor must have the same [T,D] as the bank")
        feats[-1] = anchor.float()
        valid[-1], used[-1] = True, event.sta_cutoff
    # Hard leakage guarantee: nothing used ends after its step time or after the cutoff.
    ok = valid
    if (used[ok] > taus[ok] + EPS).any() or (used[ok] > event.sta_cutoff + EPS).any():
        raise AssertionError(f"Leakage: feature after cutoff for {event.narration_id}")
    return {"feats": feats, "valid": valid, "step_times": taus, "used_times": used}


class AuxLabeler:
    """Auxiliary target per step: the first action starting in [tau + lo, tau + hi].
    Built from labelled events of the SAME split only (training targets, never inputs).
    """

    def __init__(self, events, lo=0.5, hi=2.5):
        if not lo < hi:
            raise ValueError("Need lo < hi")
        self.lo, self.hi = lo, hi
        by_video = defaultdict(list)
        for e in events:
            if e.verb is not None:
                by_video[e.video_id].append(e)
        self.by_video = {}
        for v, es in by_video.items():
            es.sort(key=lambda e: (e.start, e.narration_id))
            self.by_video[v] = ([e.start for e in es], es)

    def __call__(self, video_id, taus):
        verbs = np.full(len(taus), -1, dtype=np.int64)
        nouns = np.full(len(taus), -1, dtype=np.int64)
        starts, es = self.by_video.get(video_id, ([], []))
        for k, tau in enumerate(taus):
            j = bisect_left(starts, tau + self.lo - EPS)
            if j < len(starts) and starts[j] <= tau + self.hi + EPS:
                verbs[k], nouns[k] = es[j].verb, es[j].noun
        return verbs, nouns


class HistoryDataset(Dataset):
    """One item per STA event. Works with any FeatureBank (mock now, real later)."""

    def __init__(self, events, bank: FeatureBank, steps=16, stride=1.0, tolerance=None,
                 aux: AuxLabeler | None = None, anchors=None):
        # anchors: an Anchors object (sta_obs_tokens.npy) or a dict {narration_id: [T,D]}
        self.events, self.bank, self.aux, self.anchors = list(events), bank, aux, anchors or {}
        self.steps, self.stride, self.tolerance = steps, stride, tolerance
        missing = sorted({e.video_id for e in self.events if e.sta_cutoff >= 0 and e.video_id not in bank})
        if missing:
            raise FileNotFoundError(f"No features for {len(missing)} videos: {missing[:5]}")

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        e = self.events[idx]
        entry = self.bank[e.video_id] if e.sta_cutoff >= 0 else None
        anchor = (self.anchors.for_event(e) if hasattr(self.anchors, "for_event")
                  else self.anchors.get(e.narration_id))
        h = build_history(e, entry, self.steps, self.stride, self.tolerance,
                          anchor, shape=self.bank.shape)
        if self.aux is not None:
            av, an = self.aux(e.video_id, h["step_times"])
        else:
            av = an = np.full(self.steps, -1, dtype=np.int64)
        av[~h["valid"]] = -1
        an[~h["valid"]] = -1
        return {"feats": h["feats"], "valid": torch.from_numpy(h["valid"]),
                "step_times": torch.from_numpy(h["step_times"]),
                "used_times": torch.from_numpy(h["used_times"]),
                "aux_verb": torch.from_numpy(av), "aux_noun": torch.from_numpy(an),
                "verb_label": torch.tensor(-1 if e.verb is None else e.verb),
                "noun_label": torch.tensor(-1 if e.noun is None else e.noun),
                "use_prior": not h["valid"].any(), "narration_id": e.narration_id}
