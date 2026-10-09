"""Step 3: sample legal RGB frames from original videos using actual timestamps."""
from __future__ import annotations
from fractions import Fraction
import math
from pathlib import Path
from .events import Event, Window, observation

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
