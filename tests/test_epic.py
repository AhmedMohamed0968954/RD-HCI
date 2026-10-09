import csv
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from epic_starter.epic_pipeline import Event, Window, observation, decode_window, sample_event, grouped_split, training_priors, prepare, load_events
from epic_starter.epic_metrics import evaluate, validate_predictions, save_submission

@pytest.fixture
def video(tmp_path):
    import av
    path = tmp_path / "P01_01.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(31):
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:, :, 1 if i <= 10 else 0] = 240
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path

def event(start=2., stop=3., video_id="P01_01"):
    return Event("evt", "P01", video_id, start, stop, start-1, 0, 0)

def test_sta_never_samples_gap_or_target(video):
    result = sample_event(event(), {"P01_01": video}, "STA", num_frames=8, image_size=32)
    assert result["valid_mask"].all()
    assert (result["frame_times_seconds"] <= 1.).all()
    assert result["frames"][..., 0].max() < 25
    assert result["frames"][..., 1].mean() > 200

def test_ar_excludes_stop_boundary(video):
    _, mask, timestamps = decode_window(video, Window(.2, .7, False), 8, 32)
    assert mask.all()
    assert (timestamps >= .2).all() and (timestamps < .7).all()

def test_no_history_never_opens_video():
    result = sample_event(event(.4, 1.2), {}, "STA")
    assert result["use_prior"] and not result["valid_mask"].any()
    assert not result["frames"].any()
    assert np.isnan(result["frame_times_seconds"]).all()

def test_exact_one_second_can_use_first_frame(video):
    result = sample_event(event(1., 2.), {"P01_01": video}, "STA")
    assert result["valid_mask"].all() and not result["use_prior"]
    assert (result["frame_times_seconds"] == 0).all()

def test_missing_ar_video_is_error():
    with pytest.raises(FileNotFoundError):
        sample_event(event(), {}, "AR")

def test_variable_rate_and_nonzero_stream_origin(tmp_path):
    import av
    path = tmp_path / "vfr.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for pts in [20, 22, 25, 27, 31, 34]:
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[:, :, 1 if pts <= 25 else 0] = 240
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    frames, valid, timestamps = decode_window(path, Window(0, .65, True), 8, 32)
    assert valid.all()
    assert set(np.round(timestamps, 6)) <= {0., .2, .5}
    assert frames[..., 0].max() < 25

def test_groups_and_priors():
    events = [Event(str(i), "P01", f"V{i//2}", 2., 3., 1., i % 3, i % 4) for i in range(20)]
    train, val = grouped_split(events, .2, 42)
    assert not ({e.video_id for e in train} & {e.video_id for e in val})
    assert set(train + val) == set(events)
    assert grouped_split(events, .2, 42) == (train, val)
    priors = training_priors(train)
    assert len(priors["verb_output"]) == 97 and len(priors["noun_output"]) == 300
    assert np.isclose(sum(priors["verb_output"]), 1)
    assert np.isclose(priors["verb_output"][0], (1+sum(e.verb == 0 for e in train))/(97+len(train)))

def write_csv(path, rows):
    fields = ["narration_id", "participant_id", "video_id", "start_timestamp",
              "stop_timestamp", "verb_class", "noun_class", "sta_observation_end_seconds"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)

def row(eid="e1", video="V1", cutoff="19"):
    return dict(narration_id=eid, participant_id="P01", video_id=video,
                start_timestamp="00:00:20", stop_timestamp="00:00:21",
                verb_class=0, noun_class=0, sta_observation_end_seconds=cutoff)

def test_prepare_preserves_all_ids_and_fixed_split(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path, [row(), row("e2", "V2")])
    videos = tmp_path / "videos"; videos.mkdir()
    (videos / "V1.MP4").touch(); (videos / "V2.mp4").touch()
    summary = prepare(csv_path, videos, tmp_path / "prepared")
    assert summary["splits"]["train"]["events"] + summary["splits"]["val"]["events"] == 2
    assert not summary["hidden_test_provided"] and summary["missing_video_ids"] == []
    with pytest.raises(FileExistsError):
        prepare(csv_path, videos, tmp_path / "prepared")

def test_wrong_cutoff_and_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "x.csv"
    write_csv(path, [row(cutoff="20")])
    with pytest.raises(ValueError, match="cutoff"):
        load_events(path)
    write_csv(path, [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_events(path)

def test_class_mean_is_not_micro_accuracy():
    vs = np.full((10,97), -100.)
    vs[:,0] = 20.; vs[:,2:6] = 10.
    ns = np.full((10,300), -100.); ns[:,0] = 20.
    yv = np.array([0]*9 + [1]); yn = np.zeros(10, dtype=int)
    score = evaluate(vs, ns, yv, yn, task="STA",
                     subsets={"last": np.array([False]*9 + [True])})
    assert score["all"]["action_top5"] == 90.
    assert score["all"]["action_mt5r"] == 50.
    assert score["all"]["primary_score"] == 50.
    assert score["last"]["action_mt5r"] == 0.
    assert evaluate(vs, ns, yv, yn, task="AR")["all"]["primary_score"] == 90.

def test_joint_top5_is_not_two_independent_top5s():
    v, n = np.zeros((1,97)), np.zeros((1,300))
    v[0,:5] = n[0,:5] = [.4,.3,.2,.07,.03]
    score = evaluate(v,n,[4],[4],score_type="probabilities")["all"]
    assert score["verb_top5"] == score["noun_top5"] == 100.
    assert score["action_top5"] == 0.

def pred(eid):
    return {"narration_id":eid,"verb_output":np.ones(97,dtype=np.float32)/97,
            "noun_output":np.ones(300,dtype=np.float32)/300}

def test_submission_ids_shapes_and_nonfinite():
    assert validate_predictions([pred("b"),pred("a")],["a","b"])[0]["narration_id"] == "a"
    with pytest.raises(ValueError):
        validate_predictions([pred("a"),pred("a")],["a","b"])
    with pytest.raises(ValueError):
        validate_predictions([pred("a")],["a","b"])
    p=pred("a"); p["verb_output"][0]=np.nan
    with pytest.raises(ValueError):
        validate_predictions([p],["a"])
    p=pred("a"); p["noun_output"]=np.array(3)
    with pytest.raises(ValueError):
        validate_predictions([p],["a"])

def test_submission_zip_root(tmp_path):
    torch=pytest.importorskip("torch")
    import zipfile
    output=save_submission([pred("a")],[pred("a")],["a"],tmp_path)
    with zipfile.ZipFile(output) as f:
        assert set(f.namelist()) == {"submission.pt","submission_sta.pt"}
    loaded=torch.load(tmp_path/"submission_sta.pt",map_location="cpu",weights_only=True)
    assert loaded[0]["verb_output"].shape == (97,)
    assert loaded[0]["noun_output"].shape == (300,)
