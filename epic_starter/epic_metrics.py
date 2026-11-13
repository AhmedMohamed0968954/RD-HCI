"""Local AR/STA metrics and submission checks. Scores below are percentages."""
from __future__ import annotations
from pathlib import Path
import zipfile
import numpy as np

def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def probabilities(scores, classes, score_type):
    # Logits are unnormalized model outputs; softmax converts them into probabilities.
    values = array(scores).astype(np.float64)
    if values.ndim != 2 or values.shape[1] != classes or not np.isfinite(values).all():
        raise ValueError(f"Expected finite [N,{classes}] scores")
    if score_type == "logits":
        values = np.exp(values - values.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)
    if score_type != "probabilities":
        raise ValueError("Specify logits or probabilities explicitly")
    if (values < 0).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    return values

def labels(value, n, classes):
    a = array(value)
    if a.shape != (n,) or not np.isfinite(a).all() or not np.equal(a, np.floor(a)).all():
        raise ValueError("Expected one integer label per event")
    a = a.astype(np.int64)
    if (a < 0).any() or (a >= classes).any():
        raise ValueError("Label outside course class range")
    return a

def rank5(values):
    # Stable ties: lower class ID wins. Joint IDs are verb * 300 + noun.
    return np.argsort(-values, axis=1, kind="stable")[:, :5]

def action_top5(vp, np_, k=10):
    """Top-5 (verb, noun) pairs of the joint score p_v * p_n, as IDs verb*300+noun.

    Any pair in the joint top 5 has its verb in the verb top 5 and its noun in the
    noun top 5 (otherwise five better pairs exist), so searching the top-k x top-k
    grid with k >= 5 is exact. k=10 also absorbs near-ties. Ties break toward the
    lower joint ID, matching a stable sort over all 29,100 pairs, but ~300x faster.
    """
    n = len(vp)
    cv = np.argsort(-vp, axis=1, kind="stable")[:, :k]
    cn = np.argsort(-np_, axis=1, kind="stable")[:, :k]
    rows = np.arange(n)[:, None, None]
    score = (vp[rows[:, :, 0], cv][:, :, None] * np_[rows[:, :, 0], cn][:, None, :]).reshape(n, -1)
    ids = (cv[:, :, None] * 300 + cn[:, None, :]).reshape(n, -1)
    order = np.lexsort((ids, -score), axis=1)[:, :5]
    return np.take_along_axis(ids, order, axis=1)

def scores_for(y, top):
    # Top-5 accuracy averages samples; MT5R averages per-class recalls with equal class weights.
    hit1 = top[:, 0] == y
    hit5 = (top == y[:, None]).any(axis=1)
    recalls = [hit5[y == c].mean() for c in np.unique(y)]
    return {"top1": float(100 * hit1.mean()),
            "top5": float(100 * hit5.mean()),
            "mt5r": float(100 * np.mean(recalls))}

def evaluate(verb_scores, noun_scores, verb_labels, noun_labels,
             task="AR", score_type="logits", subsets=None):
    if task not in {"AR", "STA"}:
        raise ValueError("task must be AR or STA")
    vp = probabilities(verb_scores, 97, score_type)
    np_ = probabilities(noun_scores, 300, score_type)
    n = len(vp)
    if n == 0 or len(np_) != n:
        raise ValueError("Nonempty aligned prediction batches required")
    yv, yn = labels(verb_labels, n, 97), labels(noun_labels, n, 300)
    av = yv * 300 + yn
    # An action is a (verb, noun) pair, not the average of two independent accuracies.
    topv, topn = rank5(vp), rank5(np_)
    topa = action_top5(vp, np_)
    masks = {"all": np.ones(n, dtype=bool)}
    for name, mask in (subsets or {}).items():
        mask = array(mask)
        if name == "all" or mask.dtype != np.bool_ or mask.shape != (n,):
            raise ValueError("Subset masks must be named boolean arrays of length N")
        masks[name] = mask
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            result[name] = {"events": 0, "primary_score": None}
            continue
        part = {"events": int(mask.sum())}
        for prefix, y, top in (("verb", yv, topv), ("noun", yn, topn), ("action", av, topa)):
            part.update({prefix + "_" + k: v for k, v in scores_for(y[mask], top[mask]).items()})
        part["primary_metric"] = "action_top1" if task == "AR" else "action_mt5r"
        part["primary_score"] = part[part["primary_metric"]]
        result[name] = part
    return result

def validate_predictions(predictions, expected_ids):
    """Check exact event coverage, class dimensions, and finite scores before packaging."""
    expected_ids = list(expected_ids)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected event IDs are not unique")
    found = {}
    for item in predictions:
        event_id = item.get("narration_id")
        if not isinstance(event_id, str) or event_id in found:
            raise ValueError("Invalid or duplicate prediction ID")
        row = {"narration_id": event_id}
        for key, count in (("verb_output", 97), ("noun_output", 300)):
            values = array(item[key])
            if values.shape != (count,) or values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError(f"{event_id}: {key} must be a finite floating vector of length {count}")
            converted = values.astype(np.float32)
            if not np.isfinite(converted).all():
                raise ValueError("Values overflow float32")
            row[key] = converted
        found[event_id] = row
    if set(found) != set(expected_ids):
        raise ValueError(f"Prediction ID mismatch: missing={len(set(expected_ids)-set(found))}, "
                         f"unexpected={len(set(found)-set(expected_ids))}")
    return [found[key] for key in expected_ids]

def save_submission(ar_predictions, sta_predictions, expected_ids, out_dir):
    """Local packaging only. Does not submit to Codabench."""
    import torch
    expected_ids = list(expected_ids)
    checked = [validate_predictions(p, expected_ids) for p in (ar_predictions, sta_predictions)]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    filenames = ["submission.pt", "submission_sta.pt"]
    if any((out / name).exists() for name in filenames + ["submission.zip"]):
        raise FileExistsError("Submission output exists; choose a fresh directory")
    for filename, rows in zip(filenames, checked):
        saved = [{"narration_id": row["narration_id"],
                  "verb_output": torch.from_numpy(row["verb_output"].copy()),
                  "noun_output": torch.from_numpy(row["noun_output"].copy())} for row in rows]
        torch.save(saved, out / filename)
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in filenames:
            archive.write(out / filename, arcname=filename)
    return out / "submission.zip"


def evaluate_breakdown(verb_scores, noun_scores, verb_labels, noun_labels, task="AR",
                       score_type="logits", unseen_mask=None, tail_verbs=(), tail_nouns=()):
    """Overall / seen / unseen / tail table for the validation split.

    Tail follows the EPIC-KITCHENS-100 style: verb metrics on events with a tail
    verb, noun metrics on events with a tail noun, action metrics on events whose
    verb OR noun is tail. Tail class lists must come from training counts
    (``tail_classes.json``), never from validation labels.
    """
    vp = probabilities(verb_scores, 97, score_type)
    np_ = probabilities(noun_scores, 300, score_type)
    n = len(vp)
    yv, yn = labels(verb_labels, n, 97), labels(noun_labels, n, 300)
    ya = yv * 300 + yn
    tops = {"verb": (yv, rank5(vp)), "noun": (yn, rank5(np_)), "action": (ya, action_top5(vp, np_))}
    tv, tn = np.isin(yv, list(tail_verbs)), np.isin(yn, list(tail_nouns))
    every = np.ones(n, dtype=bool)
    groups = {"overall": {"verb": every, "noun": every, "action": every}}
    if unseen_mask is not None:
        u = array(unseen_mask).astype(bool)
        if u.shape != (n,):
            raise ValueError("unseen_mask must have one entry per event")
        groups["unseen"] = {"verb": u, "noun": u, "action": u}
        groups["seen"] = {"verb": ~u, "noun": ~u, "action": ~u}
    if len(tail_verbs) or len(tail_nouns):
        groups["tail"] = {"verb": tv, "noun": tn, "action": tv | tn}
    primary = "action_top1" if task == "AR" else "action_mt5r"
    result = {}
    for name, masks in groups.items():
        part = {}
        for comp, (y, top) in tops.items():
            m = masks[comp]
            part[comp + "_events"] = int(m.sum())
            if m.any():
                part.update({comp + "_" + k: v for k, v in scores_for(y[m], top[m]).items()})
        part["primary_metric"] = primary
        part["primary_score"] = part.get(primary)
        result[name] = part
    return result


def format_breakdown(result, keys=("verb_top1", "noun_top1", "action_top1", "verb_top5",
                                   "noun_top5", "action_top5", "action_mt5r")):
    """Plain-text table, ready for logs or slides."""
    lines = ["subset    " + " ".join(f"{k:>12}" for k in keys)]
    for name, part in result.items():
        cells = [f"{part[k]:12.2f}" if part.get(k) is not None else f"{'-':>12}" for k in keys]
        lines.append(f"{name:<9} " + " ".join(cells))
    return "\n".join(lines)
