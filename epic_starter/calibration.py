"""Member 1: post-hoc logit adjustment and the cold-start prior (blueprint 6.2, 6.3).

    z_cal = z / T - tau * log(pi)

pi is the training-split class frequency (Laplace-smoothed). A positive tau
pushes frequent classes down and lifts rare ones. AR (Action Top-1) usually
wants tau near 0; STA (class-mean Top-5 recall) usually wants a larger tau, so
the two tasks are tuned separately on the validation split.

Only training labels feed pi and the cold-start prior. Validation labels are
used solely to pick tau; report that tuning honestly (see ``split_halves``).
"""
from __future__ import annotations
import math
import numpy as np

NUM_VERBS, NUM_NOUNS = 97, 300


def class_log_prior(labels, num_classes, delta=1e-4):
    """log pi from training labels with additive smoothing delta (per class share)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    if counts.shape[0] != num_classes:
        raise ValueError("Label outside class range")
    probs = counts / counts.sum()
    probs = (probs + delta) / (1.0 + delta * num_classes)
    return np.log(probs)


def adjust(logits, log_prior, tau=0.0, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64)
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("Temperature must be positive")
    return z / temperature - tau * np.asarray(log_prior, dtype=np.float64)[None, :]


def grid_search(verb_logits, noun_logits, verb_labels, noun_labels, log_pi_v, log_pi_n,
                task, taus_v, taus_n, temperature=1.0, mask=None):
    """Return the (tau_v, tau_n) maximising the task's primary metric, plus the full table.

    task="AR" maximises Action Top-1; task="STA" maximises Action class-mean Top-5 recall.
    ``mask`` optionally restricts tuning to some events (e.g. events with history).
    """
    from .epic_metrics import action_top5, labels as check_labels, probabilities
    if task not in {"AR", "STA"}:
        raise ValueError("task must be AR or STA")
    zv, zn = np.asarray(verb_logits, np.float64), np.asarray(noun_logits, np.float64)
    if mask is not None:
        zv, zn = zv[mask], zn[mask]
        verb_labels, noun_labels = np.asarray(verb_labels)[mask], np.asarray(noun_labels)[mask]
    n = len(zv)
    yv, yn = check_labels(verb_labels, n, NUM_VERBS), check_labels(noun_labels, n, NUM_NOUNS)
    ya = yv * NUM_NOUNS + yn
    table = np.zeros((len(taus_v), len(taus_n)))
    if task == "AR":
        # Action Top-1 only needs each marginal's argmax: evaluate the two axes separately.
        top_v = [np.argmax(adjust(zv, log_pi_v, t, temperature), axis=1) for t in taus_v]
        top_n = [np.argmax(adjust(zn, log_pi_n, t, temperature), axis=1) for t in taus_n]
        for i, tv in enumerate(top_v):
            ok_v = tv == yv
            for j, tn in enumerate(top_n):
                table[i, j] = 100 * np.mean(ok_v & (tn == yn))
    else:
        pv = [probabilities(adjust(zv, log_pi_v, t, temperature), NUM_VERBS, "logits") for t in taus_v]
        pn = [probabilities(adjust(zn, log_pi_n, t, temperature), NUM_NOUNS, "logits") for t in taus_n]
        classes = np.unique(ya)
        for i, a in enumerate(pv):
            for j, b in enumerate(pn):
                hit = (action_top5(a, b) == ya[:, None]).any(axis=1)
                table[i, j] = 100 * np.mean([hit[ya == c].mean() for c in classes])
    i, j = np.unravel_index(np.argmax(table), table.shape)
    return {"task": task, "metric": "action_top1" if task == "AR" else "action_mt5r",
            "tau_v": float(taus_v[i]), "tau_n": float(taus_n[j]), "best": float(table[i, j]),
            "baseline_tau0": _value_at_zero(table, taus_v, taus_n),
            "taus_v": [float(t) for t in taus_v], "taus_n": [float(t) for t in taus_n],
            "table": table.round(4).tolist(), "temperature": temperature, "events": n}


def _value_at_zero(table, taus_v, taus_n):
    zi = [k for k, t in enumerate(taus_v) if abs(t) < 1e-12]
    zj = [k for k, t in enumerate(taus_n) if abs(t) < 1e-12]
    return float(table[zi[0], zj[0]]) if zi and zj else None


def split_halves(video_ids, seed=0):
    """Split validation events into two halves by video, so tau can be tuned on
    one half and reported on the other (avoids an optimistic tuned score)."""
    vids = sorted(set(video_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(vids)
    first = set(vids[: len(vids) // 2])
    return np.array([v in first for v in video_ids])


def cold_start_prior(train_events, window_s=5.0, alpha=0.1):
    """Distribution of video-initial actions (start < window_s) in TRAINING events.

    Used for STA events whose cutoff is negative (no frames to look at). alpha is
    additive smoothing so unseen opening classes keep a small probability.
    """
    verb = np.full(NUM_VERBS, alpha, dtype=np.float64)
    noun = np.full(NUM_NOUNS, alpha, dtype=np.float64)
    used = 0
    for e in train_events:
        if e.verb is None or e.noun is None:
            raise ValueError("Cold-start prior needs labeled training events")
        if e.start < window_s:
            verb[e.verb] += 1
            noun[e.noun] += 1
            used += 1
    if used == 0:
        raise ValueError("No training actions start inside the cold-start window")
    return {"verb_output": (verb / verb.sum()).tolist(), "noun_output": (noun / noun.sum()).tolist(),
            "window_seconds": window_s, "alpha": alpha, "events_used": used,
            "source": "internal training split only"}


def cold_start_logits(prior, log_pi_v, log_pi_n, tau_v=0.0, tau_n=0.0):
    """Scores for a no-history STA event: log pi_init - tau * log pi (same tau as STA)."""
    v = np.log(np.asarray(prior["verb_output"])) - tau_v * np.asarray(log_pi_v)
    n = np.log(np.asarray(prior["noun_output"])) - tau_n * np.asarray(log_pi_n)
    return v.astype(np.float32), n.astype(np.float32)
