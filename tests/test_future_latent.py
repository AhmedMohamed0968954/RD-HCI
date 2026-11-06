"""Future-latent helpers and the predictor feasibility script, using the dummy backend (no checkpoint)."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import numpy as np
import pytest
import torch
from epic_starter.data.events import load_events
from epic_starter.models.future_latent import DummyAnticipativeModel, anticipate, same_feature_space
from scripts.predictor_feasibility import legal_clip, parse_args, run_feasibility


@pytest.fixture
def project(tmp_path):
    """A 12 s, 10 fps, 32x32 video whose colour changes over time, and three STA events."""
    videos = tmp_path / "videos"
    videos.mkdir()
    with av.open(str(videos / "P01_01.mp4"), "w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(120):
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            pixels[:, :, i % 3] = 20 + 2 * i
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    rows = [dict(narration_id=f"e{i}", participant_id="P01", video_id="P01_01", start_timestamp=start,
                 stop_timestamp=start + 1, verb_class=0, noun_class=0)
            for i, start in enumerate([9.0, 10.0, 2.0])]  # e2 has only 1 s of history
    csv_path = tmp_path / "annotations.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, videos


def test_anticipate_returns_one_slice_after_the_requested_gap():
    torch.manual_seed(0)
    model = DummyAnticipativeModel(resolution=32)
    x = torch.randn(2, 3, 8, 32, 32)
    enc, enc_full, pred, pred_full = anticipate(model, x, torch.tensor([0.0, 1.0]))
    assert enc.shape == (2, 4 * 2 * 2, 32)  # 8 frames / tubelet 2 = 4 slices of 2x2 patches
    assert pred.shape == (2, 2 * 2, 32)  # one predicted slice
    assert not torch.allclose(pred[0], pred[1])  # different horizons -> different target positions
    assert torch.equal(model(x, torch.tensor([0.0, 1.0])), torch.cat([enc, pred], dim=1))
    assert same_feature_space(enc_full, pred_full, 32)
    assert not same_feature_space(enc_full, torch.zeros(2, 4, 48), 32)  # e.g. a distilled student


def test_legal_clip_refuses_to_read_past_the_cutoff(project):
    csv_path, videos = project
    event = load_events(csv_path)[0]
    with pytest.raises(ValueError, match="after the STA cutoff"):
        legal_clip(event, videos / "P01_01.mp4", event.sta_cutoff + 0.1, 8, 8, 32)
    x, latest = legal_clip(event, videos / "P01_01.mp4", event.sta_cutoff, 8, 8, 32)
    assert x.shape == (3, 8, 32, 32) and latest <= event.sta_cutoff


def test_feasibility_dummy_end_to_end(project, tmp_path):
    csv_path, videos = project
    out = tmp_path / "report.json"
    args = parse_args(["--backend", "dummy", "--csv", str(csv_path), "--video-root", str(videos),
                       "--out", str(out), "--num-events", "5"])
    report = run_feasibility(args)
    assert json.loads(out.read_text())["events"] == report["events"]
    assert sorted(report["events"]) == ["e0", "e1"]  # e2 lacks the 2.25 s of history the check needs
    checks = report["checks"]
    assert checks["finite"] and checks["matches_official_wrapper"]
    assert checks["latest_frame_minus_cutoff_s"] <= 0
    assert set(report["fidelity"]) == {"0.0", "1.0"}
    assert all(np.isfinite(v) for row in report["fidelity"].values() for v in row.values())
    with pytest.raises(FileExistsError):
        run_feasibility(args)
