import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.io as io


class EpicKitchensDataset(Dataset):

  def __init__(self, manifest_csv: str, num_frames: int = 16, transform=None):
    self.df = pd.read_csv(manifest_csv)
    self.num_frames = num_frames
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def _sample_indices(self, total_frames: int) -> list[int]:
    if total_frames <= self.num_frames:
      return (
          list(range(total_frames))
          + [total_frames - 1] * (self.num_frames - total_frames)
          if total_frames > 0
          else [0] * self.num_frames
      )
    step = total_frames / float(self.num_frames)
    return [int(step * i) for i in range(self.num_frames)]

  def __getitem__(self, idx: int):
    row = self.df.iloc[idx]
    clip_path = row["clip_path"]

    # Read video tensor (T, H, W, C)
    video_frames, _, _ = io.read_video(clip_path, pts_unit="sec")

    indices = self._sample_indices(len(video_frames))
    frames = video_frames[indices].permute(
        0, 3, 1, 2
    )  # -> (T, C, H, W), uint8

    if self.transform:
      frames = self.transform(frames)

    verb_label = torch.tensor(int(row["verb_class"]), dtype=torch.long)
    noun_label = torch.tensor(int(row["noun_class"]), dtype=torch.long)

    return {
        "video": frames,
        "verb_label": verb_label,
        "noun_label": noun_label,
        "narration_id": row["narration_id"],
    }