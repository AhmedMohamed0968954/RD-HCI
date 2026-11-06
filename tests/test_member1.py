"""Member 1: hybrid split, evaluator, calibration and leakage checks (numpy/pandas only)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from epic_starter.data.events import Event
from epic_starter.data.hybrid_split import hybrid_split, tail_classes, video_number, coverage_report
from epic_starter.epic_metrics import action_top5, rank5, evaluate, evaluate_breakdown
from epic_starter.calibration import adjust, class_log_prior, grid_search, cold_start_prior
from epic_starter.leakage import (LeakageError, assert_times_before_cutoff, check_event_features,
                                  check_manifest_rows, history_rows, check_batch)


def toy_events():
    rng = np.random.default_rng(1)
    events = []
    for p in range(1, 13):
        n_videos = 3 + p % 4
        for v in list(range(1, n_videos)) + [101]:
            for i in range(10 + 5 * p):
                start = 2.0 + 3 * i
                events.append(Event(f"P{p:02d}_{v:02d}_{i}", f"P{p:02d}", f"P{p:02d}_{v:02d}",
                                    start, start + 1.5, start - 1, int(rng.integers(0, 20)),
                                    int(rng.integers(0, 50))))
    return events


def test_video_number_orders_extension_after_original():
    assert sorted(["P01_101", "P01_09", "P01_10"], key=video_number) == ["P01_09", "P01_10", "P01_101"]


def test_hybrid_split_is_isolated_and_deterministic():
    events = toy_events()
    a, b = hybrid_split(events), hybrid_split(events)
    assert a.unseen_participants == b.unseen_participants and a.seen_val_videos == b.seen_val_videos
    assert not {e.video_id for e in a.train} & {e.video_id for e in a.val}
    assert not {e.participant_id for e in a.train} & set(a.unseen_participants)
    assert len(a.train) + len(a.val) == len(events)
    # Seen validation takes the LATEST videos: 101 before any 0x video of a participant.
    for p in {e.participant_id for e in a.train}:
        held = [video_number(v) for v in a.seen_val_videos if v.startswith(p)]
        kept = [video_number(e.video_id) for e in a.train if e.participant_id == p]
        if held:
            assert min(held) > max(kept)


def test_explicit_unseen_participants():
    s = hybrid_split(toy_events(), unseen_participants=["P05"])
    assert s.unseen_participants == ("P05",)
    with pytest.raises(ValueError):
        hybrid_split(toy_events(), unseen_participants=["P99"])


def test_tail_classes_cover_last_twenty_percent():
    counts = [50, 30, 10, 5, 5, 0]
    assert tail_classes(counts) == [2, 3, 4]   # head 50+30 = 80%; zero-count class excluded
    report = coverage_report(hybrid_split(toy_events()))
    assert 0 < report["coverage"]["verb"] <= 1


def test_fast_action_top5_matches_brute_force():
    rng = np.random.default_rng(0)
    vp = rng.dirichlet(np.ones(97), 200)
    np_ = rng.dirichlet(np.ones(300), 200)
    brute = rank5((vp[:, :, None] * np_[:, None, :]).reshape(200, -1))
    assert (action_top5(vp, np_) == brute).all()


def test_breakdown_perfect_scores_and_tail_masks():
    yv, yn = np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3])
    zv, zn = np.eye(97)[yv] * 20, np.eye(300)[yn] * 20
    r = evaluate_breakdown(zv, zn, yv, yn, "STA", unseen_mask=np.array([1, 0, 0, 1], bool),
                           tail_verbs=[3], tail_nouns=[2])
    assert r["overall"]["action_top1"] == 100 and r["overall"]["action_mt5r"] == 100
    assert r["unseen"]["verb_events"] == 2 and r["seen"]["verb_events"] == 2
    assert r["tail"]["verb_events"] == 1 and r["tail"]["noun_events"] == 1
    assert r["tail"]["action_events"] == 2   # tail verb OR tail noun
    old = evaluate(zv, zn, yv, yn, "STA")["all"]
    assert old["action_mt5r"] == r["overall"]["action_mt5r"]


def test_logit_adjustment_lifts_rare_classes():
    log_pi = class_log_prior([0] * 90 + [1] * 10, 2)
    z = np.array([[1.0, 0.5]])
    assert np.argmax(adjust(z, log_pi, 0.0)) == 0
    assert np.argmax(adjust(z, log_pi, 1.0)) == 1


def test_grid_search_finds_tau_that_undoes_head_bias():
    rng = np.random.default_rng(0)
    n = 3000
    yv = rng.choice(97, n, p=np.r_[0.5, np.full(96, 0.5 / 96)])
    yn = rng.integers(0, 300, n)
    log_pi_v = class_log_prior(yv, 97)
    log_pi_n = class_log_prior(yn, 300)
    zv = rng.standard_normal((n, 97)) + 3 * log_pi_v + 4 * np.eye(97)[yv]
    zn = rng.standard_normal((n, 300)) + 6 * np.eye(300)[yn]
    g = grid_search(zv, zn, yv, yn, log_pi_v, log_pi_n, "STA", [0.0, 1.0, 2.0, 3.0], [0.0])
    assert g["tau_v"] > 0 and g["best"] >= g["baseline_tau0"]


def test_cold_start_prior_uses_only_early_actions():
    events = [Event("a", "P01", "P01_01", 0.5, 1.0, -0.5, 4, 7),
              Event("b", "P01", "P01_01", 30.0, 31.0, 29.0, 9, 9)]
    p = cold_start_prior(events, alpha=1e-6)
    assert np.argmax(p["verb_output"]) == 4 and p["events_used"] == 1
    assert abs(sum(p["noun_output"]) - 1) < 1e-9


def test_times_before_cutoff():
    assert_times_before_cutoff([1.0, 2.0, np.nan], 2.0)
    with pytest.raises(LeakageError):
        assert_times_before_cutoff([2.01], 2.0)
    with pytest.raises(LeakageError):
        assert_times_before_cutoff([0.0], -1.0)   # negative cutoff: nothing may be observed


def test_feature_index_rejects_late_clip_and_missing_prior():
    events = [Event("a", "P01", "P01_01", 5.0, 6.0, 4.0, 0, 0),
              Event("b", "P01", "P01_01", 0.2, 1.0, -0.8, 0, 0)]
    good = pd.DataFrame({"narration_id": ["a", "b"], "row": [0, 1],
                         "sta_clip_end_s": [3.98, -1.0], "sta_use_prior": [False, True]})
    check_event_features(good, events)
    late = good.copy(); late.loc[0, "sta_clip_end_s"] = 4.01
    with pytest.raises(LeakageError, match="after the cutoff"):
        check_event_features(late, events)
    noprior = good.copy(); noprior.loc[1, "sta_use_prior"] = False
    with pytest.raises(LeakageError):
        check_event_features(noprior, events)
    with pytest.raises(LeakageError, match="without features"):
        check_event_features(good.iloc[:1], events)


def test_manifest_rows_check():
    row = {"narration_id": "a", "task": "STA", "start": 5.0, "sta_cutoff": 4.0,
           "observation": {"start": 0.0, "end": 4.0, "inclusive_end": True, "empty": False}}
    check_manifest_rows([row])
    bad = dict(row, observation=dict(row["observation"], end=4.5))
    with pytest.raises(LeakageError):
        check_manifest_rows([bad])


def test_history_rows_never_pass_anchor_or_cutoff():
    bank = pd.DataFrame({"t_end_s": np.arange(0.5, 40.0, 0.5), "row": np.arange(79)})
    rows, ends, anchors = history_rows(bank, cutoff=13.73, steps=16, spacing=1.0)
    assert ends[-1] == 13.5 and (ends[np.isfinite(ends)] <= anchors[np.isfinite(ends)]).all()
    assert (rows[:2] == -1).all()            # anchors before the first bank clip are padding
    rows, ends, _ = history_rows(bank, cutoff=-0.5)
    assert (rows == -1).all()
    with pytest.raises(LeakageError):
        check_batch(np.array([[1.0, 5.0]]), np.array([4.0]))
