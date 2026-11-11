"""Member 4's future branch: FutureBranch, its wiring into GatedAGAModel / HistoryDataset, and the
gate ablation options of scripts/train_sta_memory.py (mock features, CPU, seconds)."""
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
from torch.utils.data import DataLoader

from epic_starter.data.events import Event, load_events
from epic_starter.memory import Anchors, FeatureBank, HistoryDataset, write_mock_consolidated
from epic_starter.models.aga import AGAMemoryModel, GatedAGAModel
from epic_starter.models.factorized_probe import FactorizedAttentiveProbe
from epic_starter.models.future_branch import PREDICTED_TOKENS, FutureBranch
from epic_starter.models.gate import GatedVerbNounHead


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pred_probe_checkpoint(path, dim=16, task="STA", mode="factorized_pred"):
    torch.manual_seed(0)
    probe = FactorizedAttentiveProbe(embed_dim=dim, mode=mode).eval()
    torch.save({"model": probe.state_dict(), "mode": mode, "task": task, "epoch": 3,
                "metrics": {"all": {"action_mt5r": 7.23}}}, path)
    return probe


def test_mean_encoder_reads_only_the_predicted_tokens():
    assert PREDICTED_TOKENS == (7,) + tuple(range(8, 24))
    torch.manual_seed(0)
    branch = FutureBranch(16, 8, encoder="mean").eval()
    x = torch.randn(3, 24, 16)
    fv, fn = branch(x)
    assert fv.shape == fn.shape == (3, 8)
    y = x.clone()
    y[:, :7] = 100.0                                     # observed tokens copied in by the extractor
    assert torch.allclose(branch(y)[0], fv)
    y[:, 7] += torch.randn(16)                          # the predicted temporal token does matter
    assert not torch.allclose(branch(y)[0], fv)
    x[1] = float("nan")                                  # a missing row may hold anything
    fv, fn = branch(x, torch.tensor([True, False, True]))
    assert torch.isfinite(fv).all() and torch.equal(fv[1], torch.zeros(8)) and torch.equal(fn[1], torch.zeros(8))


def test_probe_encoder_is_member2_predictor_pathway(tmp_path):
    probe = pred_probe_checkpoint(tmp_path / "pred.pt")
    branch = FutureBranch(16, 8, encoder="probe", freeze_probe=True)
    assert branch.load_probe(tmp_path / "pred.pt") == (3, 7.23)
    branch.train()
    assert not branch.probe.training and branch.proj_v.training           # frozen probe: no dropout
    assert not any(p.requires_grad for p in branch.probe.parameters())
    assert all(p.requires_grad for p in branch.proj_v.parameters())
    # Member 2's own code: force its obs/pred gate to take only the predicted features
    with torch.no_grad():
        for gate in (probe.gate_v, probe.gate_n):
            gate.weight.zero_()
            gate.bias.copy_(torch.tensor([-1e4, 1e4]))
    obs, pred = torch.randn(4, 24, 16), torch.randn(4, 24, 16)
    ref_v, ref_n = probe.encode_tokens(obs, pred_tokens=pred)
    got_v, got_n = branch.pooled(pred)
    assert torch.allclose(got_v, ref_v, atol=1e-5) and torch.allclose(got_n, ref_n, atol=1e-5)
    pred_probe_checkpoint(tmp_path / "obs.pt", mode="factorized")
    with pytest.raises(ValueError):
        branch.load_probe(tmp_path / "obs.pt")
    with pytest.raises(ValueError):
        FutureBranch(16, 8, encoder="mean").load_probe(tmp_path / "pred.pt")


def gated(branches=GatedAGAModel.BRANCHES, future=True):
    torch.manual_seed(0)
    return GatedAGAModel(AGAMemoryModel(feat_dim=16, d=16, heads=2), GatedVerbNounHead(in_dim=16),
                         future_encoder=FutureBranch(16, 16) if future else None, branches=branches)


def test_future_branch_gets_gate_weight_only_where_available():
    model = gated()
    feats, valid = torch.randn(3, 5, 1, 16), torch.ones(3, 5, dtype=torch.bool)
    tokens, avail = torch.randn(3, 24, 16), torch.tensor([True, False, True])
    out = model(feats, valid, future_tokens=tokens, future_available=avail)
    w = out["verb_gates"]
    assert (w[[0, 2], 2] > 0).all() and w[1, 2] == 0
    assert torch.allclose(w.sum(1), torch.ones(3)) and torch.allclose(out["noun_gates"].sum(1), torch.ones(3))
    out["verb_logits"].sum().backward()
    assert model.future_encoder.proj_v[1].weight.grad.abs().sum() > 0
    # without an encoder, future tokens are ignored (Member 3's behaviour so far)
    w = gated(future=False)(feats, valid, future_tokens=tokens, future_available=avail)["verb_gates"]
    assert torch.equal(w[:, 2], torch.zeros(3))


@pytest.mark.parametrize("branches", [("present",), ("history", "present"), ("present", "future")])
def test_switched_off_branches_get_no_weight(branches):
    model = gated(branches).eval()
    assert model.branches == branches
    out = model(torch.randn(4, 5, 1, 16), torch.ones(4, 5, dtype=torch.bool),
                future_tokens=torch.randn(4, 24, 16), future_available=torch.ones(4, dtype=torch.bool))
    for name, w in zip(GatedAGAModel.BRANCHES, out["verb_gates"].T):
        assert (w > 0).all() if name in branches else torch.equal(w, torch.zeros(4)), name
    assert "branch_mask" not in model.state_dict()             # old checkpoints still load
    with pytest.raises(ValueError):
        gated(("history", "past"))


def test_dataset_returns_predictor_tokens_of_the_exact_clip(tmp_path):
    events = [Event("a", "P01", "P01_01", 12.0, 13.0, 11.0, 1, 2),
              Event("cold", "P01", "P01_01", 0.5, 1.5, -0.5, 1, 2)]
    write_mock_consolidated(events, tmp_path, tokens=24, dim=4)
    future = Anchors(tmp_path, tokens_file="sta_pred_tokens.npy")
    ds = HistoryDataset(events, FeatureBank(root=tmp_path), steps=4, anchors=Anchors(tmp_path),
                        present_tokens=True, future=future)
    pred = np.load(tmp_path / "sta_pred_tokens.npy")
    a, cold = ds[0], ds[1]
    assert a["future_available"] and torch.equal(a["future_tokens"], torch.from_numpy(pred[0]).float())
    assert not torch.equal(a["future_tokens"], a["present_tokens"])        # not the observed clip
    assert not cold["future_available"] and torch.equal(cold["future_tokens"], torch.zeros(24, 4))
    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch["future_tokens"].shape == (2, 24, 4) and batch["future_available"].tolist() == [True, False]
    with pytest.raises(AssertionError):                                   # same leakage check as anchors
        future.for_event(Event("a", "P01", "P01_01", 11.5, 12.5, 10.5, 1, 2))
    # dict input; an all-zero block (failed extraction) is not a future
    ds = HistoryDataset(events[:1], FeatureBank(root=tmp_path), steps=4, future={"a": torch.zeros(24, 4)})
    assert not ds[0]["future_available"]


def stamp(x):
    cs = int(round(x * 100))                       # whole centiseconds: never prints "60.00"
    return f"{cs // 360000:02d}:{cs // 6000 % 60:02d}:{cs % 6000 / 100:05.2f}"


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for v in range(6):
        vid, t = f"P0{v % 3 + 1}_{v:02d}", 0.5
        for i in range(20):
            t += rng.uniform(1, 4)
            rows.append(dict(narration_id=f"{vid}_{i}", participant_id=vid[:3], video_id=vid,
                             start_timestamp=stamp(t), stop_timestamp=stamp(t + 1.5),
                             verb_class=int(rng.integers(5)), noun_class=int(rng.integers(8))))
    with open(tmp_path / "train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(tmp_path / "test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])[:5])
        w.writeheader()
        w.writerow(dict(narration_id="T_0", participant_id="P09", video_id="P09_01",
                        start_timestamp="00:00:06.00", stop_timestamp="00:00:08.00"))
    events = load_events(tmp_path / "train.csv") + load_events(tmp_path / "test.csv", labeled=False)
    write_mock_consolidated(load_events(tmp_path / "train.csv"), tmp_path / "f16", tokens=24, dim=16)
    write_mock_consolidated(events, tmp_path / "f16_test", tokens=24, dim=16)
    return tmp_path


def run(toy, name, *extra, features="f16"):
    out = toy / "runs" / name
    args = ["--train-csv", str(toy / "train.csv"), "--out-dir", str(out), "--steps", "4", "--epochs", "2",
            "--d-model", "16", "--heads", "2", "--batch-size", "32", "--num-workers", "0", "--device", "cpu"]
    if features:
        args += ["--features-dir", str(toy / features)]
    return script("train_sta_memory").main(args + list(extra)), out


def test_training_with_future_branch_and_ablations(toy):
    s, out = run(toy, "hpf_mean", "--fusion", "three_way", "--future-encoder", "mean",
                 "--test-csv", str(toy / "test.csv"), features="f16_test")
    assert 0 <= s["best_val_action_mt5r"] <= 100
    val = np.load(out / "val_outputs.npz")
    has = val["future_available"].astype(bool)
    assert has.any() and (val["verb_gates"][has, 2] > 0).all() and (val["verb_gates"][~has, 2] == 0).all()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["events_with_future"]["val"] == int(has.sum())
    ckpt = torch.load(out / "best.pt", weights_only=False)
    assert ckpt["branches"] == ("history", "present", "future") and ckpt["future_encoder"] == "mean"
    assert any(k.startswith("future_encoder.proj_v") for k in ckpt["state_dict"])
    assert (out / "submission_sta.pt").is_file()

    _, out = run(toy, "present_only", "--fusion", "three_way", "--branches", "present")
    g = np.load(out / "val_outputs.npz")["verb_gates"]
    assert (g[:, [0, 2]] == 0).all()
    run(toy, "hp", "--fusion", "three_way", "--branches", "history,present")
    run(toy, "mock_pf", "--fusion", "three_way", "--future-encoder", "mean", "--branches", "present,future",
        "--epochs", "1", features=None)

    script("report_sta_memory").main(["--runs-dir", str(toy / "runs"), "--out-dir", str(toy / "report")])
    table = (toy / "report" / "results.md").read_text()
    for label in ("three_way H+P+F (future: mean)", "three_way P |", "three_way H+P |", "three_way P+F (future: mean)"):
        assert label in table, label
    summary = script("member4_gate_summary").main(["--runs-dir", str(toy / "runs"), "--steps", "4",
                                                   "--out", str(toy / "m4" / "gate.md"), "--any-present"])
    rows = {e["label"]: e for e in summary}
    assert set(rows) == {"three_way P", "three_way H+P", "three_way P+F (future: mean)", "three_way H+P+F (future: mean)"}
    assert rows["three_way P"]["verb_gates"]["history"] == 0 and rows["three_way P"]["verb_gates"]["future"] == 0
    assert rows["three_way H+P+F (future: mean)"]["verb_gates_with_future"]["future"] > 0
    assert rows["three_way P"]["delta_vs_HP"][2] == 1                 # same seed as the H+P run
    assert (toy / "m4" / "gate.json").is_file() and "history + present" in (toy / "m4" / "gate.md").read_text()


def test_frozen_member2_future_probe_is_not_trained(toy, tmp_path):
    probe = pred_probe_checkpoint(tmp_path / "pred.pt")
    _, out = run(toy, "hpf_probe", "--fusion", "three_way", "--future-encoder", "probe",
                 "--future-init", str(tmp_path / "pred.pt"), "--freeze-future")
    state = torch.load(out / "best.pt", weights_only=False)["state_dict"]
    for k, v in probe.state_dict().items():
        assert torch.equal(state[f"future_encoder.probe.{k}"], v), k
    assert json.loads((out / "config.json").read_text())["future_init"]["val_action_mt5r"] == 7.23


@pytest.mark.parametrize("bad", [
    ["--future-encoder", "mean"],                                            # needs three_way
    ["--fusion", "three_way", "--branches", "history,past"],
    ["--fusion", "three_way", "--branches", ""],
    ["--fusion", "three_way", "--future-encoder", "mean", "--branches", "history,present"],
    ["--fusion", "three_way", "--future-encoder", "mean", "--no-anchor"],
    ["--fusion", "three_way", "--future-encoder", "mean", "--freeze-future"],
    ["--fusion", "three_way", "--future-encoder", "probe", "--freeze-future"],
])
def test_inconsistent_options_are_refused(bad):
    with pytest.raises(SystemExit):
        script("train_sta_memory").parse_args(["--train-csv", "x.csv", "--out-dir", "o", *bad])
