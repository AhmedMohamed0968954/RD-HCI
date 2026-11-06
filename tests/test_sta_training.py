import csv
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pytest
import torch

from epic_starter.data.events import Event, load_events
from epic_starter.memory import Anchors, FeatureBank, build_history, write_mock_consolidated


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stamp(x):
    return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"


def write_csv(path, rows, labeled=True):
    cols = ["narration_id", "participant_id", "video_id", "start_timestamp", "stop_timestamp"]
    cols += ["verb_class", "noun_class"] if labeled else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(0)
    train, test = [], []
    for v in range(8):
        vid, t = f"P0{v % 3 + 1}_{v:02d}", 0.5
        for i in range(25):
            t += rng.uniform(1, 4)
            train.append(dict(narration_id=f"{vid}_{i}", participant_id=vid[:3], video_id=vid,
                              start_timestamp=stamp(t), stop_timestamp=stamp(t + 1.5),
                              verb_class=int(rng.integers(5)), noun_class=int(rng.integers(8))))
    for i, start in enumerate([0.4, 6.0, 20.0]):          # 0.4 s: cold start, must get the prior
        test.append(dict(narration_id=f"T_{i}", participant_id="P09", video_id="P09_01",
                         start_timestamp=stamp(start), stop_timestamp=stamp(start + 2)))
    write_csv(tmp_path / "train.csv", train)
    write_csv(tmp_path / "test.csv", test, labeled=False)
    events = load_events(tmp_path / "train.csv") + load_events(tmp_path / "test.csv", labeled=False)
    write_mock_consolidated(events, tmp_path / "features" / "mock_v1", tokens=6, dim=16)
    return tmp_path


def test_pooled_bank_equals_token_mean(toy):
    root = toy / "features" / "mock_v1"
    full, pooled = FeatureBank(root=root), FeatureBank(root=root, pooled=True, cache_dir=toy / "cache")
    assert pooled.shape == (1, 16) and (toy / "cache" / "mock_v1" / "bank_tokens_mean.npy").is_file()
    again = FeatureBank(root=root, pooled=True, cache_dir=toy / "cache")      # served from cache
    e = Event("x", "P01", "P01_00", 30.0, 31.0, 29.0, 0, 0)
    h_full, h_pool = build_history(e, full["P01_00"], steps=8), build_history(e, pooled["P01_00"], steps=8)
    assert torch.allclose(h_full["feats"].mean(1, keepdim=True), h_pool["feats"], atol=1e-2)
    assert torch.equal(build_history(e, again["P01_00"], steps=8)["feats"], h_pool["feats"])
    a_full, a_pool = Anchors(root), Anchors(root, pooled=True)
    first = load_events(toy / "train.csv")[30]
    assert torch.allclose(a_full.for_event(first).float().mean(0), a_pool.for_event(first)[0].float(), atol=1e-2)


@pytest.mark.parametrize("mode,fusion", [("single", "aga"), ("decoupled", "aga"), ("decoupled", "three_way")])
def test_training_run_writes_everything(toy, mode, fusion):
    if fusion == "three_way":
        pytest.importorskip("epic_starter.models.gate")
    train = script("train_sta_memory")
    out = toy / "runs" / f"{mode}_{fusion}_S8"
    summary = train.main(["--train-csv", str(toy / "train.csv"), "--test-csv", str(toy / "test.csv"),
                          "--features-dir", str(toy / "features" / "mock_v1"), "--pool-cache", str(toy / "cache"),
                          "--out-dir", str(out), "--mode", mode, "--fusion", fusion, "--steps", "8", "--epochs", "3",
                          "--d-model", "32", "--heads", "4", "--batch-size", "32", "--num-workers", "0",
                          "--device", "cpu"])
    for name in ("config.json", "metrics.json", "best.pt", "val_outputs.npz", "test_outputs.npz",
                 "submission_sta.pt"):
        assert (out / name).is_file(), name
    assert 0 <= summary["best_val_action_mt5r"] <= 100
    sub = torch.load(out / "submission_sta.pt")
    assert [r["narration_id"] for r in sub] == ["T_0", "T_1", "T_2"]
    assert all(r["verb_output"].shape == (97,) and r["noun_output"].shape == (300,) for r in sub)
    z = np.load(out / "test_outputs.npz")
    assert z["use_prior"].tolist() == [True, False, False]
    val = np.load(out / "val_outputs.npz")
    assert np.allclose(val["verb_logits"][val["use_prior"]], val["prior_verb"])
    assert ("attn" in val.files) == (mode == "single") and ("attn_noun" in val.files) == (mode == "decoupled")
    assert ("verb_gates" in val.files) == (fusion == "three_way")
    with pytest.raises(FileExistsError):                 # finished runs are never overwritten
        train.main(["--train-csv", str(toy / "train.csv"), "--out-dir", str(out), "--epochs", "1"])


def test_report_builds_table_and_figures(toy):
    pytest.importorskip("matplotlib")
    train = script("train_sta_memory")
    for mode in ("single", "decoupled"):
        train.main(["--train-csv", str(toy / "train.csv"), "--features-dir", str(toy / "features" / "mock_v1"),
                    "--out-dir", str(toy / "runs" / f"{mode}_S4"), "--mode", mode, "--steps", "4",
                    "--epochs", "1", "--d-model", "16", "--heads", "2", "--num-workers", "0", "--device", "cpu"])
    script("report_sta_memory").main(["--runs-dir", str(toy / "runs"), "--out-dir", str(toy / "report")])
    rep = toy / "report"
    assert (rep / "results.md").read_text().count("|") > 10
    for name in ("horizon.png", "single_S4_attention.png", "decoupled_S4_gates.png"):
        assert (rep / name).is_file(), name
