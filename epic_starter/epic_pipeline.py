"""Course EPIC-KITCHENS: grouped splits and timestamp-safe AR/STA sampling."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re

@dataclass(frozen=True)
class Event:
    """One annotated action; start/stop are seconds, verb/noun keep the original class IDs."""
    narration_id: str
    participant_id: str
    video_id: str
    start: float
    stop: float
    sta_cutoff: float
    verb: int | None = None
    noun: int | None = None

@dataclass(frozen=True)
class Window:
    start: float
    end: float
    inclusive_end: bool
    empty: bool = False

def seconds(value: str) -> float:
    parts = str(value).strip().split(":")
    if len(parts) == 3:
        h, m, s = map(Decimal, parts)
        if h < 0 or not (0 <= m < 60) or not (0 <= s < 60):
            raise ValueError("Invalid timestamp: " + str(value))
        answer = float(h * 3600 + m * 60 + s)
    elif len(parts) == 1:
        answer = float(value)
    else:
        raise ValueError("Invalid timestamp: " + str(value))
    if not math.isfinite(answer):
        raise ValueError("Non-finite timestamp")
    return answer

def load_events(path: Path, labeled: bool = True) -> list[Event]:
    """Read and validate annotations. Use labeled=False for the hidden test table."""
    events, seen = [], set()
    required = {"narration_id", "participant_id", "video_id", "start_timestamp", "stop_timestamp"}
    if labeled:
        required |= {"verb_class", "noun_class"}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        for row in reader:
            event_id = row["narration_id"].strip()
            if not event_id or event_id in seen:
                raise ValueError("Empty/duplicate narration_id: " + event_id)
            seen.add(event_id)
            participant, video = row["participant_id"].strip(), row["video_id"].strip()
            if not participant or not video:
                raise ValueError("Missing participant/video ID for " + event_id)
            start, stop = seconds(row["start_timestamp"]), seconds(row["stop_timestamp"])
            if start < 0 or stop <= start:
                raise ValueError("Invalid action interval for " + event_id)
            cutoff = float(Decimal(str(start)) - Decimal("1.0"))
            given = row.get("sta_observation_end_seconds", "").strip()
            if given:
                supplied = seconds(given)
                if not math.isclose(supplied, cutoff, rel_tol=0, abs_tol=1e-6):
                    raise ValueError("STA cutoff does not equal start minus one: " + event_id)
                cutoff = min(cutoff, supplied)
            verb = int(row["verb_class"]) if labeled else None
            noun = int(row["noun_class"]) if labeled else None
            if labeled and not (0 <= verb < 97 and 0 <= noun < 300):
                raise ValueError("Label outside fixed course class IDs: " + event_id)
            events.append(Event(event_id, participant, video, start, stop, cutoff, verb, noun))
    if not events:
        raise ValueError("Empty event table")
    return events

def observation(event: Event, task: str, context: float = 5.0) -> Window:
    # AR observes the action itself; STA observes history and leaves a one-second gap.
    # For start=20 and context=5, STA may only use frames in [14, 19] seconds.
    if task == "AR":
        return Window(event.start, event.stop, False)
    if task != "STA" or context <= 0 or not math.isfinite(context):
        raise ValueError("Use AR or STA with positive finite context")
    if event.sta_cutoff < 0:
        return Window(0.0, event.sta_cutoff, True, True)
    return Window(max(0.0, event.sta_cutoff - context), event.sta_cutoff, True)

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
    planned = ["summary.json", "priors.json"]
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
        "seed": seed, "split_unit": "video_id", "validation_video_fraction": val_fraction,
        "sta_context_seconds": context, "num_videos_on_disk": len(index),
        "missing_video_ids": missing, "hidden_test_provided": bool(tests),
        "splits": {name: {"events": len(es), "videos": len({e.video_id for e in es}),
                         "sta_no_history_events": sum(e.sta_cutoff < 0 for e in es)}
                   for name, es in splits.items()},
        "validation_video_ids": sorted({e.video_id for e in val}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "priors.json").write_text(json.dumps(training_priors(train), indent=2), encoding="utf-8")
    return summary

def decode_window(path: Path | None, window: Window, num_frames=8, image_size=224):
    """Return RGB uint8 T,H,W,C, valid mask and relative presentation timestamps.
    PTS filters are applied BEFORE RGB conversion or feature extraction.
    Empty windows do not open the video. Repeated valid frames are permitted.
    """
    import numpy as np
    if num_frames < 1 or image_size < 1:
        raise ValueError("Positive num_frames/image_size required")
    output = np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)
    # Frames have shape [time, height, width, RGB]; zeros are placeholders, checked via mask.
    mask = np.zeros(num_frames, dtype=bool)
    times = np.full(num_frames, np.nan, dtype=np.float64)
    if window.empty:
        return output, mask, times
    if window.start < 0 or window.end < window.start:
        raise ValueError("Invalid observation window")
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("Video is missing; an unavailable file is not an empty-history event")
    import av
    lo, hi = Fraction(str(window.start)), Fraction(str(window.end))
    targets = np.array([window.start + (i + .5) * (window.end - window.start) / num_frames
                        for i in range(num_frames)])
    best = [None] * num_frames
    distances = np.full(num_frames, np.inf)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream: " + str(path))
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        origin = (stream.start_time or 0) * stream.time_base
        # PTS * time_base gives presentation time; subtract origin for video-relative seconds.
        # Do not divide frame numbers by average FPS: videos may have variable frame rates.
        offset = math.floor((origin + lo) / stream.time_base)
        container.seek(offset, backward=True, any_frame=False, stream=stream)
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Missing presentation timestamp; refusing frame-number fallback")
            when = frame.pts * frame.time_base - origin
            if previous is not None and when < previous:
                raise ValueError("Non-monotonic presentation timestamps")
            previous = when
            if when > hi or (when == hi and not window.inclusive_end):
                break
            if when < lo:
                continue
            if frame.is_corrupt:
                raise ValueError("Corrupt video frame")
            distance = np.abs(targets - float(when))
            # Choose the nearest legal frame for each target; short windows may repeat frames.
            for i in np.flatnonzero(distance < distances):
                distances[i] = distance[i]
                best[i] = (frame, float(when))
        for i, chosen in enumerate(best):
            if chosen is not None:
                frame, when = chosen
                output[i] = frame.reformat(width=image_size, height=image_size).to_ndarray(format="rgb24")
                mask[i], times[i] = True, when
    return output, mask, times

def sample_event(event: Event, index, task: str, context=5., num_frames=8, image_size=224):
    """Future Dataset entry point: compute a window, sample frames, return validity information."""
    window = observation(event, task, context)
    frames, valid, times = decode_window(index.get(event.video_id), window, num_frames, image_size)
    if task == "AR" and not valid.any():
        raise ValueError("AR interval has no decodable frames: " + event.narration_id)
    return {"frames": frames, "valid_mask": valid, "frame_times_seconds": times,
            "use_prior": bool(task == "STA" and not valid.any())}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--train-csv", required=True)
    prep.add_argument("--test-csv")
    prep.add_argument("--video-root", required=True)
    prep.add_argument("--out-dir", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--val-fraction", type=float, default=.2)
    prep.add_argument("--context", type=float, default=5.)
    prep.add_argument("--allow-missing-videos", action="store_true")
    sample = sub.add_parser("sample")
    sample.add_argument("--csv", required=True)
    sample.add_argument("--unlabeled", action="store_true")
    sample.add_argument("--video-root", required=True)
    sample.add_argument("--narration-id", required=True)
    sample.add_argument("--task", choices=["AR", "STA", "both"], default="both")
    sample.add_argument("--context", type=float, default=5.)
    sample.add_argument("--num-frames", type=int, default=8)
    sample.add_argument("--image-size", type=int, default=224)
    sample.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.train_csv, args.video_root, args.out_dir, args.test_csv,
                         args.seed, args.val_fraction, args.context, args.allow_missing_videos)
        print(json.dumps(result, indent=2))
    else:
        import numpy as np
        matches = [e for e in load_events(Path(args.csv), not args.unlabeled)
                   if e.narration_id == args.narration_id]
        if len(matches) != 1:
            raise ValueError("Event ID not found exactly once")
        event = matches[0]
        index = video_index(Path(args.video_root))
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tasks = ["AR", "STA"] if args.task == "both" else [args.task]
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", event.narration_id)
        for task in tasks:
            target = out / f"{prefix}_{task.lower()}.npz"
            metadata = target.with_suffix(".json")
            if target.exists() or metadata.exists():
                raise FileExistsError("Sample output exists: " + str(target))
            result = sample_event(event, index, task, args.context, args.num_frames, args.image_size)
            np.savez_compressed(target, **result)
            details = manifest_record(event, task, args.context)
            details["sampled_frame_times_seconds"] = [float(t) if math.isfinite(t) else None
                                                       for t in result["frame_times_seconds"]]
            details["use_prior"] = result["use_prior"]
            metadata.write_text(json.dumps(details, indent=2), encoding="utf-8")
            print(json.dumps({"task": task, "output": str(target), "use_prior": result["use_prior"],
                              "times": details["sampled_frame_times_seconds"]}))
if __name__ == "__main__":
    main()
