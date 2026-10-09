"""Step 2: create reproducible splits, class statistics and training-only priors."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from .events import Event, load_events, observation

def grouped_split(events: list[Event], val_fraction: float = .2, seed: int = 42):
    # Keep all events from one video in the same split to avoid adjacent-clip leakage.
    # A fixed seed lets every team member reproduce the same split.
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    videos = sorted({e.video_id for e in events})
    if len(videos) < 2:
        raise ValueError("At least two videos are required for a grouped split")
    random.Random(seed).shuffle(videos)
    count = max(1, min(len(videos) - 1, round(len(videos) * val_fraction)))
    held_out = set(videos[:count])
    return ([e for e in events if e.video_id not in held_out],
            [e for e in events if e.video_id in held_out])

def training_priors(events: list[Event], alpha: float = 1.0):
    # STA events without history still need predictions: training class frequencies are a fallback.
    # alpha smooths the counts. Never use validation or test labels to compute priors.
    if not events or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Need training events and positive smoothing")
    verb, noun = [alpha] * 97, [alpha] * 300
    for e in events:
        if e.verb is None or e.noun is None:
            raise ValueError("Priors require labeled training events")
        verb[e.verb] += 1
        noun[e.noun] += 1
    return {"verb_output": [v / sum(verb) for v in verb],
            "noun_output": [v / sum(noun) for v in noun],
            "source": "internal training split only", "alpha": alpha}

def video_index(root: Path):
    if not Path(root).is_dir():
        raise FileNotFoundError("Video root does not exist: " + str(root))
    index = {}
    for path in Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() == ".mp4":
            if path.stem in index:
                raise ValueError("Duplicate video_id on disk: " + path.stem)
            index[path.stem] = path
    return index

def manifest_record(event: Event, task: str, context: float):
    return dict(asdict(event), task=task, observation=asdict(observation(event, task, context)))

def prepare(train_csv, videos, out_dir, test_csv=None, seed=42, val_fraction=.2,
            context=5.0, allow_missing_videos=False):
    """Write lightweight manifests without copying videos. Each JSONL line describes one event."""
    if not math.isfinite(context) or context <= 0:
        raise ValueError("Positive finite STA context required")
    events = load_events(Path(train_csv))
    train, val = grouped_split(events, val_fraction, seed)
    tests = load_events(Path(test_csv), False) if test_csv else []
    if {e.narration_id for e in tests} & {e.narration_id for e in events}:
        raise ValueError("Train and hidden-test event IDs overlap")
    index = video_index(Path(videos))
    missing = sorted({e.video_id for e in events + tests} - set(index))
    if missing and not allow_missing_videos:
        raise FileNotFoundError(f"{len(missing)} referenced videos missing; first: {missing[:5]}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    planned = ["summary.json", "priors.json", "class_counts.json"]
    splits = {"train": train, "val": val}
    if tests:
        splits["test"] = tests
    planned += [f"{split}_{task.lower()}.jsonl" for split in splits for task in ("AR", "STA")]
    if any((out_dir / name).exists() for name in planned):
        raise FileExistsError("Output files already exist; choose a new output directory")
    for split, subset in splits.items():
        for task in ("AR", "STA"):
            with (out_dir / f"{split}_{task.lower()}.jsonl").open("w", encoding="utf-8") as f:
                for e in subset:
                    f.write(json.dumps(manifest_record(e, task, context)) + "\n")
    summary = {
        "annotation_sha256": hashlib.sha256(Path(train_csv).read_bytes()).hexdigest(),
        "seed": seed, "split_unit": "video_id", "validation_video_fraction": val_fraction,
        "sta_context_seconds": context, "num_videos_on_disk": len(index),
        "missing_video_ids": missing, "hidden_test_provided": bool(tests),
        "splits": {name: {"events": len(es), "videos": len({e.video_id for e in es}),
                         "sta_no_history_events": sum(e.sta_cutoff < 0 for e in es)}
                   for name, es in splits.items()},
        "validation_video_ids": sorted({e.video_id for e in val}),
        "train_video_ids": sorted({e.video_id for e in train}),
        "video_overlap": sorted({e.video_id for e in train} & {e.video_id for e in val}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "priors.json").write_text(json.dumps(training_priors(train), indent=2), encoding="utf-8")
    counts = {name: class_counts(es) for name, es in splits.items()}
    (out_dir / "class_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return summary


def class_counts(events):
    """Retain zero-count classes so class IDs never change between splits."""
    verbs = Counter(e.verb for e in events if e.verb is not None)
    nouns = Counter(e.noun for e in events if e.noun is not None)
    actions = Counter(f"{e.verb}:{e.noun}" for e in events if e.verb is not None)
    return {"verb": [verbs[i] for i in range(97)],
            "noun": [nouns[i] for i in range(300)], "action": dict(sorted(actions.items())),
            "participants": dict(sorted(Counter(e.participant_id for e in events).items()))}


def read_manifest(path):
    """Validate generated JSONL before it becomes training input."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    for row in rows:
        e = Event(**{key: row[key] for key in Event.__dataclass_fields__})
        if not e.narration_id or e.narration_id in seen:
            raise ValueError("Empty or duplicate event ID")
        seen.add(e.narration_id)
        if not all(math.isfinite(x) for x in [e.start,e.stop,e.sta_cutoff]) or e.start < 0 or e.stop <= e.start:
            raise ValueError("Invalid action interval")
        if abs(e.sta_cutoff - (e.start-1)) > 1e-6 or e.sta_cutoff > e.start-1+1e-9:
            raise ValueError("Invalid STA cutoff")
        for label, count in [(e.verb,97),(e.noun,300)]:
            if label is not None and (not isinstance(label,int) or not 0 <= label < count):
                raise ValueError("Invalid class ID")
        w = row["observation"]
        task = row["task"]
        if task not in {"AR","STA"}:
            raise ValueError("Invalid task")
        if task == "AR":
            expected = asdict(observation(e,task))
            if w != expected:
                raise ValueError("AR manifest window differs from annotation")
        else:
            if not w["inclusive_end"] or w["end"] != e.sta_cutoff:
                raise ValueError("STA window must end at the cutoff")
            if e.sta_cutoff < 0:
                if w != asdict(observation(e,task)):
                    raise ValueError("Invalid empty STA window")
            elif w["empty"] or not math.isfinite(w["start"]) or not 0 <= w["start"] <= w["end"]:
                raise ValueError("Invalid STA history window")
    return rows
