"""Step 5: audit a prepared split, validate batches and measure loading speed.

Run from the repository root: python scripts/check_data.py --help
No model is trained. All compared worker settings use the same selected events.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Subset
from dataset import EpicKitchensDataset, make_dataloader
from epic_starter.data.events import load_events, seconds
from epic_starter.data.manifests import read_manifest


def audit_legacy_manifest(path, annotation_csv):
    """Report coverage without mistaking a partial clip export for a full split."""
    events = {e.narration_id:e for e in load_events(Path(annotation_csv))}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    seen, duplicates, unknown, mismatched = set(), [], [], []
    missing_files = 0
    for row in rows:
        eid = row["narration_id"]
        if eid in seen:
            duplicates.append(eid)
        seen.add(eid)
        if eid not in events:
            unknown.append(eid)
        else:
            event = events[eid]
            if (row["video_id"] != event.video_id or row["participant_id"] != event.participant_id or
                    int(row["verb_class"]) != event.verb or int(row["noun_class"]) != event.noun or
                    seconds(row["start_timestamp"]) != event.start or seconds(row["stop_timestamp"]) != event.stop):
                mismatched.append(eid)
        clip = Path(row.get("clip_path", ""))
        if not clip.is_absolute():
            clip = Path(path).resolve().parent / clip
        missing_files += not clip.is_file()
    return dict(rows=len(rows), unique_events=len(seen), annotation_events=len(events),
                covered_events=len(seen & events.keys()), missing_annotation_events=len(events.keys()-seen),
                duplicate_ids=duplicates, unknown_ids=unknown, mismatched_annotation_rows=mismatched,
                unavailable_clip_files=missing_files,
                note="Clip availability is machine-local; missing files do not prove extraction failed elsewhere.")


def verify_splits(directory):
    rows = {f"{split}_{task}": read_manifest(Path(directory)/f"{split}_{task}.jsonl")
            for split in ("train", "val") for task in ("ar", "sta")}
    for name, values in rows.items():
        expected_task = name.split("_")[-1].upper()
        if any(row["task"] != expected_task for row in values):
            raise ValueError("Task does not match manifest filename: " + name)
    for split in ("train", "val"):
        ar, sta = rows[split+"_ar"], rows[split+"_sta"]
        keys = ("narration_id", "video_id", "participant_id", "start", "stop", "sta_cutoff", "verb", "noun")
        if [tuple(r[k] for k in keys) for r in ar] != [tuple(r[k] for k in keys) for r in sta]:
            raise ValueError("AR/STA splits differ in events, labels or order")
    for key in ("video_id", "narration_id"):
        if {r[key] for r in rows["train_ar"]} & {r[key] for r in rows["val_ar"]}:
            raise ValueError("Train/validation leakage: " + key)
    return {name:len(values) for name,values in rows.items()}


def select_indices(events, count):
    # Include an empty-history event when available, plus normal history and broad coverage.
    if count < 1:
        raise ValueError("Sample count must be positive")
    selected = []
    for condition in (lambda e:e.sta_cutoff < 0, lambda e:e.sta_cutoff >= 5):
        index = next((i for i,e in enumerate(events) if condition(e)), None)
        if index is not None and index not in selected:
            selected.append(index)
    for i in np.linspace(0, len(events)-1, min(count,len(events)), dtype=int):
        if int(i) not in selected:
            selected.append(int(i))
    return selected[:min(count,len(events))]


def validate_batch(batch):
    frames, mask, times = batch["frames"], batch["valid_mask"], batch["frame_times_seconds"]
    if frames.dtype != torch.float32 or frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("Expected float32 [B,T,3,H,W]")
    if not torch.isfinite(frames).all() or (frames < 0).any() or (frames > 1).any():
        raise ValueError("Inspection batches must be finite, unnormalized RGB in [0,1]")
    if mask.shape != frames.shape[:2] or times.shape != mask.shape:
        raise ValueError("Frame/mask/timestamp dimensions disagree")
    for i, task in enumerate(batch["task"]):
        if not batch["timing_verified"][i]:
            raise ValueError("Cannot certify timing from legacy pre-trimmed clips")
        valid = times[i][mask[i]]
        if not torch.isfinite(valid).all() or (valid < batch["observation_start"][i]).any():
            raise ValueError("Invalid frame timestamps")
        end = batch["observation_end"][i]
        if task == "AR":
            if not len(valid) or (valid >= end).any() or batch["use_prior"][i]:
                raise ValueError("Invalid AR frames")
        elif (valid > end).any() or (valid > batch["action_start"][i]-1+1e-9).any():
            raise ValueError("STA frame leaks into gap/target")
        if bool(batch["use_prior"][i]) != (task == "STA" and len(valid) == 0):
            raise ValueError("Incorrect empty-history fallback flag")
        if batch["has_label"][i]:
            if not 0 <= batch["verb_label"][i] < 97 or not 0 <= batch["noun_label"][i] < 300:
                raise ValueError("Labels outside class range")


def contact_sheet(sample, destination):
    images = (sample["frames"].permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
    width = images.shape[2]
    canvas = Image.new("RGB", (width*len(images), images.shape[1]+50), "white")
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(images):
        canvas.paste(Image.fromarray(frame), (i*width,50))
        stamp = float(sample["frame_times_seconds"][i])
        draw.text((i*width+3,30), f"{stamp:.3f}s" if math.isfinite(stamp) else "NO HISTORY", fill="black")
    draw.text((3,5), f'{sample["task"]} {sample["narration_id"]} window={sample["observation_start"]:.3f}..{sample["observation_end"]:.3f}', fill="black")
    canvas.save(destination)


def run_checks(args):
    output = Path(args.out_dir)
    if output.exists():
        raise FileExistsError("Use a new output directory for each inspection")
    output.mkdir(parents=True)
    report = {"split_counts": verify_splits(args.manifest_dir), "config": vars(args),
              "benchmark_note": "Includes worker startup and decode. Sequential runs may benefit from filesystem caching; not a controlled cold-cache comparison.",
              "checks": []}
    # A health report may restrict this diagnostic sample, never the training split.
    # Affected events stay in manifests and Dataset continues to fail on unreadable input.
    bad = set()
    if args.health_report:
        health = json.loads(Path(args.health_report).read_text())
        bad = set(health["failed_video_ids"]) | set(health["missing_video_ids"])
        report["unreadable_videos_excluded_from_inspection_only"] = sorted(bad)
        report["known_unreadable_videos_remaining"] = len(bad)
    if args.legacy_manifest:
        report["legacy_manifest_audit"] = audit_legacy_manifest(args.legacy_manifest,args.annotation_csv)
    torch.set_num_threads(1)
    for split in ("train","val"):
        for task in ("ar","sta"):
            data = EpicKitchensDataset(str(Path(args.manifest_dir)/f"{split}_{task}.jsonl"),
                                       video_root=args.video_root,num_frames=args.num_frames,image_size=args.image_size)
            eligible = [i for i,e in enumerate(data.events) if e.video_id not in bad]
            if not eligible:
                raise ValueError("No readable events remain for inspection")
            indices = [eligible[i] for i in select_indices([data.events[j] for j in eligible],args.samples)]
            subset = Subset(data,indices)
            for workers in args.workers:
                start = time.perf_counter()
                ids, empty = [], 0
                loader = make_dataloader(subset,args.batch_size,workers,shuffle=False)
                for batch in loader:
                    validate_batch(batch)
                    ids.extend(batch["narration_id"])
                    empty += int(batch["use_prior"].sum())
                elapsed = time.perf_counter()-start
                if ids != [data.events[i].narration_id for i in indices]:
                    raise ValueError("DataLoader dropped, duplicated or reordered events")
                record = dict(split=split,task=task,workers=workers,events=len(ids),
                              no_history=empty,seconds=elapsed,events_per_second=len(ids)/elapsed)
                report["checks"].append(record)
                (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
                print(json.dumps(record),flush=True)
            for i in indices[:2]:
                sample = data[i]
                name = f"{split}_{task}_{i}"
                contact_sheet(sample,output/f"{name}.png")
                details = {k:sample[k] for k in ["narration_id","task","use_prior","observation_start","observation_end"]}
                details["frame_times_seconds"] = [float(t) if math.isfinite(t) else None for t in sample["frame_times_seconds"]]
                (output/f"{name}.json").write_text(json.dumps(details,indent=2))
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir",required=True)
    parser.add_argument("--video-root",required=True)
    parser.add_argument("--out-dir",required=True)
    parser.add_argument("--annotation-csv")
    parser.add_argument("--legacy-manifest")
    parser.add_argument("--health-report",help="Exclude listed unreadable videos from inspection only, not manifests")
    parser.add_argument("--samples",type=int,default=16)
    parser.add_argument("--num-frames",type=int,default=8)
    parser.add_argument("--image-size",type=int,default=224)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--workers",type=int,nargs="+",default=[0,2,4])
    args = parser.parse_args()
    if args.samples < 1 or any(w < 0 for w in args.workers):
        parser.error("Invalid sample/worker counts")
    if args.legacy_manifest and not args.annotation_csv:
        parser.error("Legacy audit needs --annotation-csv")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if allocated and max(args.workers)+1 > allocated:
        parser.error("Request at least max(workers)+1 CPUs for this benchmark")
    run_checks(args)


if __name__ == "__main__":
    main()
