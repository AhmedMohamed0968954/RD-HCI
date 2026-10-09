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
    topa = np.empty((n, 5), dtype=np.int64)
    for start in range(0, n, 64):
        joint = (vp[start:start+64, :, None] * np_[start:start+64, None, :]).reshape(-1, 97 * 300)
        topa[start:start+64] = rank5(joint)
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
