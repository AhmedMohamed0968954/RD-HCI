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
    cs = int(round(x * 100))                       # whole centiseconds: never prints "60.00"
    return f"{cs // 360000:02d}:{cs // 6000 % 60:02d}:{cs % 6000 / 100:05.2f}"


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
    write_mock_consolidated(events, tmp_path / "features" / "mock_v1", tokens=24, dim=16)
    return tmp_path


def test_pooled_bank_equals_token_mean(toy):
    root = toy / "features" / "mock_v1"
    full, pooled = FeatureBank(root=root), FeatureBank(root=root, pooled=True, cache_dir=toy / "cache")
    assert pooled.shape == (1, 16) and len(list((toy / "cache" / "mock_v1").glob("bank_tokens_mean_*.npy"))) == 1
    again = FeatureBank(root=root, pooled=True, cache_dir=toy / "cache")      # served from cache
    e = Event("x", "P01", "P01_00", 30.0, 31.0, 29.0, 0, 0)
    h_full, h_pool = build_history(e, full["P01_00"], steps=8), build_history(e, pooled["P01_00"], steps=8)
    assert torch.allclose(h_full["feats"].mean(1, keepdim=True), h_pool["feats"], atol=1e-2)
    assert torch.equal(build_history(e, again["P01_00"], steps=8)["feats"], h_pool["feats"])
    # regenerated features (new file) must not reuse the old cache
    arr = np.load(root / "bank_tokens.npy")
    np.save(root / "bank_tokens.npy", arr + np.float16(1))
    fresh = FeatureBank(root=root, pooled=True, cache_dir=toy / "cache")
    assert not torch.equal(build_history(e, fresh["P01_00"], steps=8)["feats"], h_pool["feats"])
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


def test_present_clip_source(toy):
    """Exact STA clip when extracted, else the last bank clip (<= cutoff), else nothing (cold start)."""
    from epic_starter.memory import HistoryDataset
    root = toy / "features" / "mock_v1"
    bank = FeatureBank(root=root, pooled=True)
    train = load_events(toy / "train.csv")
    test = load_events(toy / "test.csv", labeled=False)
    with_anchor = HistoryDataset(train[:5], bank, steps=4, anchors=Anchors(root), present_tokens=True)
    item = with_anchor[3]
    assert int(item["present_source"]) == 1 and item["present_tokens"].shape == (24, 16)
    # the memory's last step is the token mean of the same exact clip
    assert torch.allclose(item["feats"][-1, 0], item["present_tokens"].mean(0), atol=1e-2)
    without = HistoryDataset(test, bank, steps=4, anchors={}, present_tokens=True)
    assert [int(without[i]["present_source"]) for i in range(3)] == [0, 2, 2]
    late = without[2]
    assert late["used_times"][-1] <= test[2].sta_cutoff + 1e-6                     # bank clip is legal
    assert torch.allclose(late["present_tokens"].mean(0), late["feats"][-1, 0], atol=1e-2)


def test_present_probe_initialisation_and_freezing(toy, tmp_path):
    from epic_starter.models.aga import AGAMemoryModel
    from epic_starter.models.factorized_probe import FactorizedAttentiveProbe
    torch.manual_seed(0)
    probe = FactorizedAttentiveProbe(embed_dim=16, mode="factorized").eval()
    ckpt = tmp_path / "probe.pt"
    torch.save({"model": probe.state_dict(), "mode": "factorized", "task": "STA", "epoch": 7,
                "metrics": {"all": {"action_mt5r": 7.63}}}, ckpt)
    model = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="probe")
    assert model.load_present(ckpt) == (7, 7.63)
    model.eval()
    tokens, valid = torch.randn(3, 24, 16), torch.ones(3, 5, dtype=torch.bool)
    out = model(torch.randn(3, 5, 1, 16), valid, tokens, torch.ones(3, dtype=torch.bool))
    ref = probe(tokens)
    # memory heads start as a zero correction: the model starts exactly at the probe's prediction
    assert torch.allclose(out["verb_logits"], ref["verb_output"], atol=1e-5)
    assert torch.allclose(out["noun_logits"], ref["noun_output"], atol=1e-5)
    torch.save({"model": probe.state_dict(), "mode": "factorized", "task": "AR"}, tmp_path / "ar.pt")
    with pytest.raises(ValueError):
        model.load_present(tmp_path / "ar.pt")
    # end to end: a frozen, initialised probe is not changed by training
    train = script("train_sta_memory")
    out_dir = toy / "runs" / "frozen"
    feats = toy / "features" / "mock_v1"
    write_mock_consolidated(load_events(toy / "train.csv"), tmp_path / "f16", tokens=24, dim=16)
    train.main(["--train-csv", str(toy / "train.csv"), "--features-dir", str(tmp_path / "f16"),
                "--present-init", str(ckpt), "--freeze-present", "--out-dir", str(out_dir), "--steps", "4",
                "--epochs", "2", "--d-model", "16", "--heads", "2", "--num-workers", "0", "--device", "cpu"])
    trained = torch.load(out_dir / "best.pt")["state_dict"]
    for k, v in probe.state_dict().items():
        assert torch.equal(trained[f"present.{k}"], v), k
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["present_init"]["val_action_mt5r"] == 7.63
    assert cfg["trainable_parameters"] < cfg["parameters"]
    # the pre-training score is the probe alone and becomes the report's baseline row
    init = json.loads((out_dir / "metrics.json").read_text())["init"]
    assert init["probe_alone"] and 0 <= init["val"]["all"]["action_mt5r"] <= 100
    script("report_sta_memory").main(["--runs-dir", str(toy / "runs"), "--out-dir", str(tmp_path / "rep")])
    table = (tmp_path / "rep" / "results.md").read_text()
    assert table.count("probe alone (before training)") == 1
    assert f"{init['val']['all']['action_mt5r']:.2f}" in table.splitlines()[2]


def test_mean_present_ablation_runs(toy):
    train = script("train_sta_memory")
    out = toy / "runs" / "mean_present"
    s = train.main(["--train-csv", str(toy / "train.csv"), "--features-dir", str(toy / "features" / "mock_v1"),
                    "--present-encoder", "mean", "--out-dir", str(out), "--steps", "4", "--epochs", "1",
                    "--d-model", "16", "--heads", "2", "--num-workers", "0", "--device", "cpu"])
    assert 0 <= s["best_val_action_mt5r"] <= 100
    assert "present_source" not in np.load(out / "val_outputs.npz").files
