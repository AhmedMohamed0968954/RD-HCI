"""Command-line entry point. Implementation lives in the small data/ modules.

Existing imports such as `from epic_pipeline import sample_event` still work.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re

if __package__:
    from .data.events import Event, Window, seconds, load_events, observation
    from .data.manifests import grouped_split, training_priors, video_index, manifest_record, prepare
    from .data.sampling import decode_window, sample_event
else:
    from data.events import Event, Window, seconds, load_events, observation
    from data.manifests import grouped_split, training_priors, video_index, manifest_record, prepare
    from data.sampling import decode_window, sample_event

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
