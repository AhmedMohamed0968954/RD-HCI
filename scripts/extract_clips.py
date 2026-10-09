"""Optional AR clip export. Original CLI/function names are kept for continuity.

The shared training pipeline reads original videos directly. This exporter is for
inspection or an explicitly chosen AR cache, never for STA model input.
"""
import argparse
import json
import re
import uuid
import sys
from pathlib import Path

# Support direct execution from scripts/ as well as python -m scripts.extract_clips.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.data.events import Event, load_events, observation, seconds
from epic_starter.data.manifests import read_manifest
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import subprocess
import pandas as pd
from tqdm import tqdm


def find_source_video(
    video_dir: Path, participant_id: str, video_id: str
) -> Path | None:
  candidates = [
      video_dir / f"{video_id}.mp4",
      video_dir / f"{video_id}.MP4",
      video_dir / participant_id / f"{video_id}.mp4",
      video_dir / participant_id / f"{video_id}.MP4",
      video_dir / f"{video_id}.mkv",
      video_dir / f"{video_id}.MKV",
  ]
  for path in candidates:
    if path.is_file():
      return path
  return None


def trim_segment(
    row_dict: dict, video_dir: str, output_dir: str, overwrite: bool
):
  narration_id = str(row_dict["narration_id"])
  video_id = str(row_dict["video_id"])
  participant_id = str(row_dict["participant_id"])
  start = seconds(str(row_dict["start_timestamp"]))
  stop = seconds(str(row_dict["stop_timestamp"]))
  if start < 0 or stop <= start or not re.fullmatch(r"[A-Za-z0-9_.-]+", narration_id) or narration_id in {".",".."}:
    return narration_id, None, "Invalid interval or unsafe event ID"
  event = Event(narration_id, participant_id, video_id, start, stop, start-1)
  window = observation(event, "AR")
  start_ts = str(window.start)
  duration = str(window.end-window.start)

  src_path = find_source_video(Path(video_dir), participant_id, video_id)
  if src_path is None:
    return narration_id, None, f"Video {video_id} not found."

  dst_clip = Path(output_dir) / f"{narration_id}.mp4"

  dst_clip.parent.mkdir(parents=True, exist_ok=True)
  sidecar = dst_clip.with_suffix(".json")
  signature = dict(video=str(src_path.resolve()), source_size=src_path.stat().st_size,
                   source_mtime_ns=src_path.stat().st_mtime_ns, start=start, stop=stop,
                   task="AR", codec="libx264", crf=22)
  if not overwrite and dst_clip.exists():
    try:
      saved = json.loads(sidecar.read_text())
      if saved["signature"] == signature and saved["bytes"] == dst_clip.stat().st_size:
        validate_clip(dst_clip)
        return narration_id, str(dst_clip.resolve()), None
    except (OSError, ValueError, KeyError):
      pass
    return narration_id, None, "Existing clip cannot be verified; use --overwrite to rebuild"
  temporary = dst_clip.with_name(dst_clip.stem + "." + uuid.uuid4().hex + ".part.mp4")

  cmd = [
      "ffmpeg",
      "-y",
      "-ss",
      start_ts,
      "-t",
      duration,
      "-i",
      str(src_path),
      "-c:v",
      "libx264",
      "-threads",
      "1",
      "-preset",
      "veryfast",
      "-crf",
      "22",
      "-an",  # Audio stripped: evaluation uses video modality only
      "-avoid_negative_ts",
      "make_zero",
      str(temporary),
  ]

  res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
  if res.returncode != 0:
    temporary.unlink(missing_ok=True)
    return narration_id, None, res.stderr.decode("utf-8", errors="ignore")
  try:
    validate_clip(temporary)
    temporary.replace(dst_clip)
    sidecar.write_text(json.dumps(dict(signature=signature,bytes=dst_clip.stat().st_size),indent=2))
  except Exception as exc:
    temporary.unlink(missing_ok=True)
    return narration_id, None, str(exc)
  return narration_id, str(dst_clip.resolve()), None


def validate_clip(path):
  import av
  with av.open(str(path)) as video:
    frame = next(video.decode(video=0), None)
    if frame is None or frame.is_corrupt:
      raise ValueError("Empty or corrupt clip")


def main():
  parser = argparse.ArgumentParser(
      description="Extract trimmed video clips for EPIC-KITCHENS-100."
  )
  parser.add_argument(
      "--csv_path",
      type=str,
      default="EPIC_100_train.csv",
      help="Path to training CSV",
  )
  parser.add_argument(
      "--video_dir",
      type=str,
      required=True,
      help="Path to downloaded untrimmed videos",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="clips_train",
      help="Directory to store cut clips",
  )
  parser.add_argument(
      "--output_csv",
      type=str,
      default="EPIC_100_train_manifest.csv",
      help="Output CSV with clip paths",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=1,
      help="Worker processes",
  )
  parser.add_argument(
      "--overwrite", action="store_true", help="Re-extract existing clips"
  )
  parser.add_argument("--limit", type=int, help="Export only the first N selected events")
  parser.add_argument("--manifest_jsonl", help="Restrict export to a prepared AR split")
  args = parser.parse_args()
  if args.num_workers < 1 or (args.limit is not None and args.limit < 1):
    parser.error("Worker count and limit must be positive")
  if Path(args.output_csv).resolve() == Path(args.csv_path).resolve():
    parser.error("Output manifest must not overwrite input annotations")
  if Path(args.output_csv).exists() and not args.overwrite:
    parser.error("Output manifest exists; choose a new path or --overwrite")
  load_events(Path(args.csv_path))  # Validate event IDs, timestamps and class ranges.

  df = pd.read_csv(args.csv_path)
  if args.manifest_jsonl:
    selected = read_manifest(args.manifest_jsonl)
    if any(r["task"] != "AR" for r in selected):
      parser.error("This exporter accepts AR manifests only; STA reads original videos")
    ids = {r["narration_id"] for r in selected}
    if not ids <= set(df["narration_id"]):
      parser.error("Selected manifest contains events absent from CSV")
    df = df[df["narration_id"].isin(ids)].copy()
  if args.limit:
    df = df.head(args.limit).copy()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  rows = df.to_dict("records")
  clip_paths, errors = {}, {}

  print(
      f"Extracting {len(rows)} clips to {output_dir} using {args.num_workers}"
      " workers..."
  )

  with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    futures = {
        executor.submit(
            trim_segment, r, args.video_dir, args.output_dir, args.overwrite
        ): r[
            "narration_id"
        ]
        for r in rows
    }

    for future in tqdm(as_completed(futures), total=len(futures)):
      try:
        narration_id, clip_path, error = future.result()
      except Exception as exc:
        narration_id, clip_path, error = futures[future], None, str(exc)
      if clip_path is not None:
        clip_paths[narration_id] = clip_path
      elif error is not None:
        errors[narration_id] = error
        print(f"Error [{narration_id}]: {error[:100]}")

  df["clip_path"] = df["narration_id"].map(clip_paths)
  # Preserve every selected event. Failed exports remain visible in the manifest.
  df["extraction_error"] = df["narration_id"].map(errors).fillna("")
  df["extraction_status"] = df["clip_path"].notna().map({True:"ok",False:"failed"})
  saved_df = df

  keep_cols = [
      "narration_id",
      "participant_id",
      "video_id",
      "clip_path",
      "verb",
      "verb_class",
      "noun",
      "noun_class",
      "start_timestamp",
      "stop_timestamp",
      "extraction_status",
      "extraction_error",
  ]
  saved_df = saved_df[
      [c for c in keep_cols if c in saved_df.columns]
  ].reset_index(drop=True)
  destination = Path(args.output_csv)
  destination.parent.mkdir(parents=True,exist_ok=True)
  temporary_csv = destination.with_name(destination.name + ".part")
  saved_df.to_csv(temporary_csv, index=False)
  temporary_csv.replace(destination)
  print(f"Extraction complete: {len(clip_paths)} OK, {len(errors)} failed. Manifest: {args.output_csv}")
  if errors:
    raise SystemExit(1)


if __name__ == "__main__":
  main()