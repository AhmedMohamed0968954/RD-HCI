"""Member 1: hybrid Seen + Unseen validation split (blueprint section 3.2).

Two validation subsets are carved out of the official training annotations:

* unseen: every event of a few whole, medium-sized participants. Mirrors the
  leaderboard's "Unseen Participants" metric (P18 and P32 appear only in test).
* seen: for each remaining participant, their chronologically last videos, until
  roughly ``seen_fraction`` of that participant's events. Whole videos only, so
  adjacent clips of one recording never sit on both sides of the split.

Everything is deterministic: the same CSV and arguments give the same split.
Tail classes are computed from the training side only and saved next to the
split so evaluation never peeks at validation labels to define them.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math

from .events import Event

NUM_VERBS, NUM_NOUNS = 97, 300


def video_number(video_id: str) -> int:
    """P01_09 -> 9, P01_101 -> 101. EPIC-100 extension videos (1xx) were recorded later."""
    try:
        return int(video_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("Unexpected video_id format: " + video_id) from exc


@dataclass(frozen=True)
class HybridSplit:
    train: list[Event]
    val: list[Event]
    unseen_participants: tuple[str, ...]
    seen_val_videos: tuple[str, ...]

    def subset_of(self, event: Event) -> str:
        """'train', 'val_unseen' or 'val_seen' for one event."""
        if event.participant_id in self.unseen_participants:
            return "val_unseen"
        if event.video_id in self.seen_val_videos:
            return "val_seen"
        return "train"


def _choose_unseen(events, target_fraction, k_options, pool_quantiles):
    """Pick whole participants whose combined share is near the target.

    Candidates are restricted to medium-sized participants (between the given
    quantiles of per-participant event counts) so one giant kitchen such as P22
    cannot dominate the unseen metric, and tiny ones cannot make it noisy.
    Among combinations within +-1 percentage point of the target, the one
    covering the most distinct verb+noun classes wins; ties go to the closest
    fraction, then alphabetical order, so the result is deterministic. If no
    combination lands in that band, the closest one is used.
    """
    total = len(events)
    by_p = defaultdict(list)
    for e in events:
        by_p[e.participant_id].append(e)
    sizes = sorted(len(v) for v in by_p.values())
    lo = sizes[int(pool_quantiles[0] * (len(sizes) - 1))]
    hi = sizes[int(pool_quantiles[1] * (len(sizes) - 1))]
    pool = sorted(p for p, v in by_p.items() if lo <= len(v) <= hi)
    best_key, best = None, None
    for k in k_options:
        for combo in combinations(pool, k):
            n = sum(len(by_p[p]) for p in combo)
            frac = n / total
            outside = abs(frac - target_fraction) > 0.01
            verbs = {e.verb for p in combo for e in by_p[p]}
            nouns = {e.noun for p in combo for e in by_p[p]}
            # Outside the band only wins if nothing lands inside it; then closest wins.
            coverage = 0 if outside else -(len(verbs) + len(nouns))
            key = (outside, coverage, abs(frac - target_fraction), combo)
            if best_key is None or key < best_key:
                best_key, best = key, combo
    if best is None:
        raise ValueError("No participant combination reaches the unseen target; pass them explicitly")
    return tuple(best)


def _choose_seen_videos(events, unseen, seen_fraction, max_fraction):
    by_p = defaultdict(lambda: defaultdict(int))
    for e in events:
        if e.participant_id not in unseen:
            by_p[e.participant_id][e.video_id] += 1
    chosen = []
    for p in sorted(by_p):
        videos = sorted(by_p[p], key=video_number)
        total = sum(by_p[p].values())
        taken = 0
        # Walk backwards in time; keep at least one training video per participant.
        for vid in reversed(videos[1:]):
            n = by_p[p][vid]
            if taken >= seen_fraction * total:
                break
            if (taken + n) > max_fraction * total:
                break  # stay contiguous in time: never skip a long video to grab an older one
            chosen.append(vid)
            taken += n
    return tuple(sorted(chosen))


def hybrid_split(events: list[Event], unseen_participants=None, unseen_fraction=0.06,
                 seen_fraction=0.08, seen_max_fraction=0.15, k_options=(2, 3),
                 pool_quantiles=(0.25, 0.75)) -> HybridSplit:
    if any(e.verb is None or e.noun is None for e in events):
        raise ValueError("The hybrid split needs labeled training events")
    participants = {e.participant_id for e in events}
    if unseen_participants:
        unseen = tuple(sorted(unseen_participants))
        unknown = set(unseen) - participants
        if unknown:
            raise ValueError("Unknown participants: " + ", ".join(sorted(unknown)))
    else:
        unseen = _choose_unseen(events, unseen_fraction, k_options, pool_quantiles)
    seen_videos = _choose_seen_videos(events, set(unseen), seen_fraction, seen_max_fraction)
    seen_set = set(seen_videos)
    train = [e for e in events if e.participant_id not in unseen and e.video_id not in seen_set]
    val = [e for e in events if e.participant_id in unseen or e.video_id in seen_set]
    split = HybridSplit(train, val, unseen, seen_videos)
    _assert_isolated(split)
    return split


def _assert_isolated(split: HybridSplit):
    train_videos = {e.video_id for e in split.train}
    val_videos = {e.video_id for e in split.val}
    if train_videos & val_videos:
        raise AssertionError("Video appears in both train and validation")
    if {e.participant_id for e in split.train} & set(split.unseen_participants):
        raise AssertionError("An unseen participant leaked into training")
    ids = [e.narration_id for e in split.train + split.val]
    if len(ids) != len(set(ids)):
        raise AssertionError("Event assigned twice")


def tail_classes(counts: list[int], head_share: float = 0.8) -> list[int]:
    """Classes outside the head. Head = most frequent classes covering ``head_share``
    of instances (the EPIC-KITCHENS-100 head/tail convention). Zero-count classes
    are excluded: they cannot be evaluated."""
    order = sorted(range(len(counts)), key=lambda c: (-counts[c], c))
    total, running, tail = sum(counts), 0, []
    for c in order:
        if counts[c] == 0:
            continue
        if running >= head_share * total:
            tail.append(c)
        running += counts[c]
    return sorted(tail)


def coverage_report(split: HybridSplit, head_share: float = 0.8) -> dict:
    """Class coverage and tail definition, computed from training labels only."""
    tv = Counter(e.verb for e in split.train)
    tn = Counter(e.noun for e in split.train)
    vv = Counter(e.verb for e in split.val)
    vn = Counter(e.noun for e in split.val)
    verb_counts = [tv[i] for i in range(NUM_VERBS)]
    noun_counts = [tn[i] for i in range(NUM_NOUNS)]
    n_train, n_val = len(split.train), len(split.val)
    train_verbs = {c for c in range(NUM_VERBS) if tv[c]}
    train_nouns = {c for c in range(NUM_NOUNS) if tn[c]}
    tail_v, tail_n = tail_classes(verb_counts, head_share), tail_classes(noun_counts, head_share)
    sv, sn = set(tail_v), set(tail_n)
    unseen_n = sum(e.participant_id in split.unseen_participants for e in split.val)
    return {
        "events": {"train": n_train, "val": n_val, "val_unseen": unseen_n,
                   "val_seen": n_val - unseen_n},
        "fractions": {"val": n_val / (n_train + n_val), "val_unseen": unseen_n / (n_train + n_val),
                      "val_seen": (n_val - unseen_n) / (n_train + n_val)},
        "unseen_participants": list(split.unseen_participants),
        "num_seen_val_videos": len(split.seen_val_videos),
        "coverage": {
            "verb": len(train_verbs & set(vv)) / max(1, len(train_verbs)),
            "noun": len(train_nouns & set(vn)) / max(1, len(train_nouns)),
            # Share of training instances whose class is also evaluated in val:
            # rare classes missing from val are expected and matter little here.
            "verb_instance_weighted": sum(tv[c] for c in train_verbs & set(vv)) / max(1, n_train),
            "noun_instance_weighted": sum(tn[c] for c in train_nouns & set(vn)) / max(1, n_train),
            "verb_classes_in_train": len(train_verbs), "noun_classes_in_train": len(train_nouns),
            "val_classes_missing_from_train": {
                "verb": sorted(set(vv) - train_verbs), "noun": sorted(set(vn) - train_nouns)},
        },
        "tail": {
            "definition": f"classes outside the head covering {head_share:.0%} of TRAIN instances; "
                          "a tail action has a tail verb or a tail noun",
            "verbs": tail_v, "nouns": tail_n,
            "val_events_tail_verb": sum(e.verb in sv for e in split.val),
            "val_events_tail_noun": sum(e.noun in sn for e in split.val),
            "val_events_tail_action": sum(e.verb in sv or e.noun in sn for e in split.val),
        },
        "train_counts": {"verb": verb_counts, "noun": noun_counts},
    }


def prepare_hybrid(train_csv, out_dir, test_csv=None, unseen_participants=None,
                   unseen_fraction=0.06, seen_fraction=0.08, seen_max_fraction=0.15,
                   context=5.0, head_share=0.8):
    """Write the hybrid split in the same JSONL format as ``manifests.prepare``.

    Outputs (never overwritten): train/val[_test] x ar/sta .jsonl, split.csv (one
    row per event: narration_id, participant_id, video_id, subset), summary.json,
    priors.json (train only), class_counts.json, tail_classes.json and
    cold_start_prior.json. Videos are not needed: this only reads annotations.
    """
    import csv
    import hashlib
    import json
    from pathlib import Path
    from .events import load_events
    from .manifests import manifest_record, training_priors, class_counts
    from ..calibration import cold_start_prior

    events = load_events(Path(train_csv))
    split = hybrid_split(events, unseen_participants, unseen_fraction, seen_fraction,
                         seen_max_fraction)
    tests = load_events(Path(test_csv), labeled=False) if test_csv else []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": split.train, "val": split.val}
    if tests:
        splits["test"] = tests
    planned = ["summary.json", "priors.json", "class_counts.json", "split.csv",
               "tail_classes.json", "cold_start_prior.json"]
    planned += [f"{n}_{t}.jsonl" for n in splits for t in ("ar", "sta")]
    if any((out / name).exists() for name in planned):
        raise FileExistsError("Output files already exist; choose a new output directory")
    for name, subset in splits.items():
        for task in ("AR", "STA"):
            with (out / f"{name}_{task.lower()}.jsonl").open("w", encoding="utf-8") as f:
                for e in subset:
                    row = manifest_record(e, task, context)
                    row["subset"] = split.subset_of(e) if name != "test" else "test"
                    f.write(json.dumps(row) + "\n")
    with (out / "split.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["narration_id", "participant_id", "video_id", "subset"])
        for e in events:
            w.writerow([e.narration_id, e.participant_id, e.video_id, split.subset_of(e)])
    report = coverage_report(split, head_share)
    (out / "tail_classes.json").write_text(json.dumps(
        {"definition": report["tail"]["definition"], "verbs": report["tail"]["verbs"],
         "nouns": report["tail"]["nouns"], "source": "internal training split only"},
        indent=2), encoding="utf-8")
    (out / "priors.json").write_text(json.dumps(training_priors(split.train), indent=2),
                                     encoding="utf-8")
    (out / "cold_start_prior.json").write_text(json.dumps(cold_start_prior(split.train), indent=2),
                                               encoding="utf-8")
    (out / "class_counts.json").write_text(json.dumps(
        {n: class_counts(es) for n, es in splits.items()}, indent=2), encoding="utf-8")
    summary = {
        "split_type": "hybrid_seen_unseen",
        "annotation_sha256": hashlib.sha256(Path(train_csv).read_bytes()).hexdigest(),
        "parameters": {"unseen_participants_given": list(unseen_participants or []),
                       "unseen_fraction": unseen_fraction, "seen_fraction": seen_fraction,
                       "seen_max_fraction": seen_max_fraction, "sta_context_seconds": context,
                       "head_share": head_share},
        "unseen_participants": list(split.unseen_participants),
        "seen_val_video_ids": list(split.seen_val_videos),
        "events": report["events"], "fractions": report["fractions"],
        "coverage": report["coverage"],
        "tail_val_events": {k: v for k, v in report["tail"].items() if k.startswith("val_")},
        "hidden_test_provided": bool(tests),
        "sta_no_history_events": {n: sum(e.sta_cutoff < 0 for e in es) for n, es in splits.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
