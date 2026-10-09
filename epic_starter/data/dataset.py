"""Step 4: PyTorch Dataset and DataLoader for the shared AR/STA pipeline.

Retains the original EpicKitchensDataset name, positional arguments and video key.
Use original videos for AR/STA; legacy pre-trimmed clips remain available for AR.
"""
from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from epic_starter.data.events import Event, Window, load_events, observation
from epic_starter.data.manifests import read_manifest, video_index
from epic_starter.data.sampling import decode_window, sample_event


class EpicKitchensDataset(Dataset):
    def __init__(self, manifest_csv: str, num_frames: int = 16, transform=None,
                 *, video_root=None, task=None, context=5., image_size=224,
                 source="auto", clip_root=None, normalize=False,
                 mean=(.485,.456,.406), std=(.229,.224,.225)):
        # The first three arguments match the original teammate implementation.
        # transform receives uint8 [T,C,H,W]. Return uint8 or float32 in [0,1].
        # Normalize here, once, after the transform; validation has no random transform.
        self.path = Path(manifest_csv).resolve()
        self.num_frames, self.image_size = int(num_frames), int(image_size)
        if self.num_frames < 1 or self.image_size < 1 or not np.isfinite(context) or context <= 0:
            raise ValueError("Frame count, size and context must be positive")
        self.transform, self.context, self.normalize = transform, float(context), normalize
        self.mean = torch.tensor(mean,dtype=torch.float32).reshape(1,3,1,1)
        self.std = torch.tensor(std,dtype=torch.float32).reshape(1,3,1,1)
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid normalization parameters")
        self.records = None
        if self.path.suffix.lower() == ".jsonl":
            self.records = read_manifest(self.path)
            tasks = {r["task"] for r in self.records}
            if len(tasks) != 1 or (task is not None and task not in tasks):
                raise ValueError("Task must match the prepared manifest")
            self.task = next(iter(tasks))
            self.events = [Event(**{k:r[k] for k in Event.__dataclass_fields__}) for r in self.records]
            self.df = pd.DataFrame(self.records)
        else:
            self.df = pd.read_csv(self.path)
            if ("verb_class" in self.df) != ("noun_class" in self.df):
                raise ValueError("Provide both label columns or neither")
            self.events = load_events(self.path, labeled="verb_class" in self.df)
            self.task = task or "AR"
        if self.task not in {"AR","STA"} or source not in {"auto","original","clips"}:
            raise ValueError("Invalid task/source")
        self.source = ("original" if video_root else "clips") if source == "auto" else source
        self.clip_root = Path(clip_root).resolve() if clip_root else self.path.parent
        if self.source == "original":
            if video_root is None:
                raise ValueError("Original mode requires video_root")
            self.index = video_index(Path(video_root))
            missing = sorted({e.video_id for e in self.events} - self.index.keys())
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} videos: {missing[:5]}")
        else:
            if self.task != "AR":
                raise ValueError("Target-action clips cannot be used for STA; use original videos")
            if "clip_path" not in self.df or self.df["clip_path"].isna().any():
                raise ValueError("Clip manifest has missing paths; inspect extraction errors")
            warnings.warn("Legacy AR clips have unverified source timestamps; prefer original mode.",
                          UserWarning, stacklevel=2)

    def __len__(self):
        return len(self.events)

    def _sample_indices(self, total_frames: int) -> list[int]:
        # Keep the original index sampler for pre-trimmed AR clips only.
        if total_frames <= 0:
            raise ValueError("Decoded clip contains no frames")
        if total_frames <= self.num_frames:
            return list(range(total_frames)) + [total_frames-1]*(self.num_frames-total_frames)
        step = total_frames / float(self.num_frames)
        return [int(step*i) for i in range(self.num_frames)]

    def _read_clip(self, idx):
        import av
        path = Path(str(self.df.iloc[idx]["clip_path"]))
        if not path.is_absolute():
            path = self.clip_root/path
        # Two passes avoid storing every decoded RGB frame in memory.
        with av.open(str(path)) as container:
            count = sum(1 for _ in container.decode(video=0))
        wanted = self._sample_indices(count)
        images = {}
        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    if frame.is_corrupt:
                        raise ValueError("Corrupt clip frame")
                    images[i] = frame.reformat(width=self.image_size,height=self.image_size).to_ndarray(format="rgb24")
        return dict(frames=np.stack([images[i] for i in wanted]),
                    valid_mask=np.ones(self.num_frames,dtype=bool),
                    frame_times_seconds=np.full(self.num_frames,np.nan),use_prior=False)

    def __getitem__(self, idx: int):
        event = self.events[idx]
        window = (Window(**self.records[idx]["observation"]) if self.records
                  else observation(event,self.task,self.context))
        if self.source == "clips":
            clip = self._read_clip(idx)
        elif self.records:
            frames,valid,times = decode_window(self.index[event.video_id],window,self.num_frames,self.image_size)
            if self.task == "AR" and not valid.any():
                raise ValueError("AR interval contains no frames: " + event.narration_id)
            clip = dict(frames=frames,valid_mask=valid,frame_times_seconds=times,
                        use_prior=bool(self.task == "STA" and not valid.any()))
        else:
            clip = sample_event(event,self.index,self.task,self.context,self.num_frames,self.image_size)
        frames = torch.from_numpy(clip["frames"]).permute(0,3,1,2).contiguous()
        if self.transform is not None and not clip["use_prior"]:
            frames = self.transform(frames)
        if not isinstance(frames,torch.Tensor) or tuple(frames.shape) != (self.num_frames,3,self.image_size,self.image_size):
            raise ValueError("Transform must return [T,3,H,W] at configured size")
        if frames.dtype == torch.uint8:
            frames = frames.float().div(255)
        else:
            frames = frames.float()
            if not torch.isfinite(frames).all() or (frames < 0).any() or (frames > 1).any():
                raise ValueError("Float transforms must return unnormalized values in [0,1]")
        if self.normalize and not clip["use_prior"]:
            frames = (frames-self.mean)/self.std
        return {"frames":frames,"video":frames,
                "verb_label":torch.tensor(event.verb if event.verb is not None else -1,dtype=torch.long),
                "noun_label":torch.tensor(event.noun if event.noun is not None else -1,dtype=torch.long),
                "has_label":event.verb is not None,"narration_id":event.narration_id,
                "video_id":event.video_id,"task":self.task,
                "valid_mask":torch.from_numpy(clip["valid_mask"]),
                "frame_times_seconds":torch.from_numpy(clip["frame_times_seconds"]),
                "use_prior":clip["use_prior"],"timing_verified":self.source == "original",
                "observation_start":window.start,"observation_end":window.end,
                "action_start":event.start,"action_stop":event.stop}


def collate_samples(samples):
    # Keep the video alias without batching a second copy of the same tensor.
    batch = default_collate([{k:v for k,v in s.items() if k != "video"} for s in samples])
    batch["video"] = batch["frames"]
    return batch


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_dataloader(dataset,batch_size=4,num_workers=0,shuffle=False,seed=42):
    """Keep every event, including empty-history STA and the final short batch."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid batch size or worker count")
    kwargs = dict(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,
                  drop_last=False,collate_fn=collate_samples,worker_init_fn=seed_worker,
                  generator=torch.Generator().manual_seed(seed))
    if num_workers:
        kwargs.update(multiprocessing_context="spawn",prefetch_factor=1)
    return DataLoader(dataset,**kwargs)
