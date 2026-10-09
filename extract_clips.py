import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
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
  start_ts = str(row_dict["start_timestamp"])
  stop_ts = str(row_dict["stop_timestamp"])

  src_path = find_source_video(Path(video_dir), participant_id, video_id)
  if src_path is None:
    return narration_id, None, f"Video {video_id} not found."

  dst_clip = Path(output_dir) / f"{narration_id}.mp4"

  if not overwrite and dst_clip.is_file() and dst_clip.stat().st_size > 0:
    return narration_id, str(dst_clip), None

  cmd = [
      "ffmpeg",
      "-y",
      "-ss",
      start_ts,
      "-to",
      stop_ts,
      "-i",
      str(src_path),
      "-c:v",
      "libx264",
      "-preset",
      "veryfast",
      "-crf",
      "22",
      "-an",  # Audio stripped: evaluation uses video modality only
      "-avoid_negative_ts",
      "make_zero",
      str(dst_clip),
  ]

  res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
  if res.returncode != 0:
    return narration_id, None, res.stderr.decode("utf-8", errors="ignore")

  return narration_id, str(dst_clip), None


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
      default=os.cpu_count() or 4,
      help="Worker processes",
  )
  parser.add_argument(
      "--overwrite", action="store_true", help="Re-extract existing clips"
  )
  args = parser.parse_args()

  df = pd.read_csv(args.csv_path)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  rows = df.to_dict("records")
  clip_paths = {}

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
      narration_id, clip_path, error = future.result()
      if clip_path is not None:
        clip_paths[narration_id] = clip_path
      elif error is not None:
        print(f"Error [{narration_id}]: {error[:100]}")

  df["clip_path"] = df["narration_id"].map(clip_paths)
  saved_df = df.dropna(subset=["clip_path"])

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
  ]
  saved_df = saved_df[
      [c for c in keep_cols if c in saved_df.columns]
  ].reset_index(drop=True)
  saved_df.to_csv(args.output_csv, index=False)
  print(f"Extraction complete. Manifest saved to {args.output_csv}")


if __name__ == "__main__":
  main()