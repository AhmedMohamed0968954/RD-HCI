"""Step 1: read action annotations and describe legal observation windows."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path

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
