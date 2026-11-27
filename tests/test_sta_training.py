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


def probe_checkpoint(path, dim=16):
    from epic_starter.models.factorized_probe import FactorizedAttentiveProbe
    torch.manual_seed(0)
    probe = FactorizedAttentiveProbe(embed_dim=dim, mode="factorized").eval()
    torch.save({"model": probe.state_dict(), "mode": "factorized", "task": "STA", "epoch": 7,
                "metrics": {"all": {"action_mt5r": 7.63}}}, path)
    return probe


def test_memory_pretraining_two_stages(toy, tmp_path):
    """Stage 1 trains the memory alone; stage 2 starts the probe model from that memory."""
    train = script("train_sta_memory")
    feats = tmp_path / "f16"
    write_mock_consolidated(load_events(toy / "train.csv"), feats, tokens=24, dim=16)
    common = ["--train-csv", str(toy / "train.csv"), "--features-dir", str(feats), "--steps", "4",
              "--d-model", "16", "--heads", "2", "--num-workers", "0", "--device", "cpu"]
    runs = toy / "runs"
    s1 = train.main(common + ["--present-encoder", "none", "--epochs", "2", "--out-dir", str(runs / "memonly")])
    assert 0 <= s1["best_val_action_mt5r"] <= 100
    stage1 = runs / "memonly" / "best.pt"
    mem = torch.load(stage1)["state_dict"]
    assert not any(k.startswith(("gate_", "present")) for k in mem)       # no gate, no probe
    memory_keys = [k for k in mem if k.startswith(("fx.", "aux_verb.", "aux_noun.", "mem_v.", "mem_n."))]
    assert memory_keys
    ckpt = tmp_path / "probe.pt"
    probe = probe_checkpoint(ckpt)
    warm = ["--present-init", str(ckpt), "--freeze-present", "--memory-init", str(stage1)]

    # frozen memory: it ends exactly as stage 1 left it, and the probe is untouched
    train.main(common + warm + ["--freeze-memory", "--epochs", "2", "--out-dir", str(runs / "warm_frozen")])
    got = torch.load(runs / "warm_frozen" / "best.pt")["state_dict"]
    for k in memory_keys:
        assert torch.equal(got[k], mem[k]), k
    for k, v in probe.state_dict().items():
        assert torch.equal(got[f"present.{k}"], v), k
    cfg = json.loads((runs / "warm_frozen" / "config.json").read_text())
    assert cfg["memory_init"]["source_present_encoder"] == "none" and cfg["memory_init"]["frozen"]
    # the correction still starts at zero, so the score before training is the probe alone
    assert json.loads((runs / "warm_frozen" / "metrics.json").read_text())["init"]["probe_alone"]

    # fine-tuned memory, with the AGA gate and with Member 4's three-way gate
    train.main(common + warm + ["--epochs", "1", "--out-dir", str(runs / "warm")])
    train.main(common + warm + ["--fusion", "three_way", "--epochs", "1", "--out-dir", str(runs / "3way_warm")])
    cfg_ft = json.loads((runs / "warm" / "config.json").read_text())
    assert cfg_ft["trainable_parameters"] - cfg["trainable_parameters"] == cfg["memory_init"]["parameters"]

    # the report keeps pre-trained runs apart from scratch runs
    script("report_sta_memory").main(["--runs-dir", str(runs), "--out-dir", str(tmp_path / "rep")])
    table = (tmp_path / "rep" / "results.md").read_text()
    assert "none (memory only)" in table and ", pre-trained (frozen)" in table and ", pre-trained |" in table
    rows = script("member4_gate_summary").main(["--runs-dir", str(runs), "--steps", "4",
                                                "--out", str(tmp_path / "gate.md")])
    labels = {r["label"] for r in rows}
    assert {"aga [memory pre-trained]", "aga [memory pre-trained (frozen)]",
            "three_way H+P [memory pre-trained]"} <= labels


def test_load_memory_checks_shapes_and_reads_gated_checkpoints(tmp_path):
    from epic_starter.models.aga import AGAMemoryModel, GatedAGAModel
    from epic_starter.models.gate import GatedVerbNounHead
    torch.manual_seed(1)
    src = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="none")
    ctor = dict(feat_dim=16, d=16, heads=2, alpha=0.8, mode="decoupled", present_encoder="none")
    torch.save({"state_dict": src.state_dict(), "ctor": ctor, "epoch": 3, "val_action_mt5r": 5.0},
               tmp_path / "memonly.pt")
    dst = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="probe")
    assert dst.load_memory(tmp_path / "memonly.pt") == (3, 5.0, "none")
    for name, part in src.memory_parts().items():
        for k, v in part.state_dict().items():
            assert torch.equal(dst.memory_parts()[name].state_dict()[k], v), f"{name}.{k}"
    # a three-way (GatedAGAModel) checkpoint stores the memory under 'memory.'
    gated = GatedAGAModel(dst, GatedVerbNounHead(in_dim=16))
    torch.save({"state_dict": gated.state_dict(), "ctor": {**ctor, "present_encoder": "probe"}},
               tmp_path / "gated.pt")
    again = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="mean")
    again.load_memory(tmp_path / "gated.pt")
    assert torch.equal(again.fx[-1].weight, src.fx[-1].weight)
    with pytest.raises(ValueError, match="does not match"):
        AGAMemoryModel(feat_dim=16, d=32, heads=2, present_encoder="probe").load_memory(tmp_path / "memonly.pt")
    with pytest.raises(ValueError, match="does not match"):
        AGAMemoryModel(feat_dim=16, d=16, heads=2, mode="single").load_memory(tmp_path / "memonly.pt")
    with pytest.raises(ValueError):
        GatedAGAModel(src, GatedVerbNounHead(in_dim=16))       # memory only has no present branch


def test_memory_only_ignores_the_present_and_frozen_memory_has_no_dropout():
    from epic_starter.models.aga import AGAMemoryModel
    torch.manual_seed(2)
    model = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="none", dropout=0.5).eval()
    feats, valid = torch.randn(4, 5, 1, 16), torch.ones(4, 5, dtype=torch.bool)
    out = model(feats, valid)
    assert out["verb_logits"].shape == (4, 97) and out["noun_logits"].shape == (4, 300)
    assert "gate_verb" not in out
    frozen = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="mean", freeze_memory=True).train()
    assert all(not p.requires_grad for part in frozen.memory_parts().values() for p in part.parameters())
    assert all(not m.training for part in frozen.memory_parts().values() for m in part.modules())
    assert frozen.gate_v.training and any(p.requires_grad for p in frozen.gate_v.parameters())


@pytest.mark.parametrize("bad", [["--freeze-memory"], ["--present-encoder", "none", "--memory-init", "x.pt"],
                                 ["--present-encoder", "none", "--fusion", "three_way"]])
def test_pretraining_options_are_checked(bad):
    with pytest.raises(SystemExit):
        script("train_sta_memory").parse_args(["--train-csv", "x.csv", "--out-dir", "o"] + bad)


def test_probe_steps_precompute_loader_and_training(toy, tmp_path):
    """Member 2's probe outputs per clip as memory steps: precompute, load, train both stages."""
    from epic_starter.memory import HistoryDataset, make_mock_bank
    from epic_starter.memory.probe_steps import encode, layout, precompute, step_dim
    from epic_starter.models.aga import AGAMemoryModel
    events = load_events(toy / "train.csv")
    feats = tmp_path / "f16"
    write_mock_consolidated(events, feats, tokens=24, dim=16)
    ckpt = tmp_path / "probe.pt"
    probe = probe_checkpoint(ckpt)
    steps = precompute(feats, ckpt, tmp_path / "psteps", batch_size=64, log=lambda m: None)
    src, out = np.load(feats / "bank_tokens.npy"), np.load(steps / "bank_tokens.npy")
    assert out.shape == (len(src), 1, step_dim(16))
    with torch.no_grad():
        ref = probe(torch.from_numpy(src[:10].astype(np.float32)))
    L = layout(16)
    assert np.allclose(out[:10, 0, slice(*L["verb_logits"])], ref["verb_output"].numpy(), atol=2e-2)
    assert np.allclose(out[:10, 0, slice(*L["noun_logits"])], ref["noun_output"].numpy(), atol=2e-2)
    assert precompute(feats, ckpt, steps, log=lambda m: None) == steps             # same probe: no-op
    other = tmp_path / "other.pt"
    torch.save({**torch.load(ckpt), "note": "retrained"}, other)
    with pytest.raises(FileExistsError):
        precompute(feats, other, steps, log=lambda m: None)

    # loader: steps from the probe bank, present clip from the token bank, same clips
    bank, sbank = FeatureBank(root=feats), FeatureBank(root=steps)
    ds = HistoryDataset(events[:20], bank, steps=4, anchors=Anchors(feats), present_tokens=True,
                        step_bank=sbank, step_anchors=Anchors(steps))
    item = ds[10]
    assert item["feats"].shape == (4, 1, step_dim(16)) and int(item["present_source"]) == 1
    # the last step is the probe on the same exact STA clip that the present branch reads
    assert torch.allclose(item["feats"][-1, 0].float(), encode(probe, item["present_tokens"][None])[0],
                          atol=2e-2)
    with pytest.raises(ValueError, match="disagree"):
        HistoryDataset(events[:20], bank, steps=4, step_bank=make_mock_bank(events, stride=0.5, dim=16))

    # both pre-training stages on probe steps
    train = script("train_sta_memory")
    common = ["--train-csv", str(toy / "train.csv"), "--features-dir", str(feats), "--step-features", str(steps),
              "--steps", "4", "--d-model", "16", "--heads", "2", "--num-workers", "0", "--device", "cpu"]
    runs = toy / "runs"
    train.main(common + ["--present-encoder", "none", "--epochs", "2", "--out-dir", str(runs / "ps_memonly")])
    warm = ["--present-init", str(ckpt), "--freeze-present", "--memory-init", str(runs / "ps_memonly" / "best.pt")]
    train.main(common + warm + ["--epochs", "1", "--out-dir", str(runs / "ps_warm")])
    train.main(common + warm + ["--fusion", "three_way", "--epochs", "1", "--out-dir", str(runs / "ps_3way_warm")])
    cfg = json.loads((runs / "ps_warm" / "config.json").read_text())
    assert cfg["step_features"]["dir"] == str(steps)
    assert json.loads((runs / "ps_warm" / "metrics.json").read_text())["init"]["probe_alone"]
    # token-mean and probe-step memories are not interchangeable
    with pytest.raises(ValueError, match="does not match"):
        AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="probe").load_memory(runs / "ps_memonly" / "best.pt")
    # steps computed with another probe than the present one are refused
    with pytest.raises(ValueError, match="another probe checkpoint"):
        train.main(common + ["--present-init", str(other), "--epochs", "1", "--out-dir", str(runs / "bad")])
    rows = script("member4_gate_summary").main(["--runs-dir", str(runs), "--steps", "4",
                                                "--out", str(tmp_path / "gate.md")])
    assert {"aga [memory probe steps, pre-trained]", "three_way H+P [memory probe steps, pre-trained]"} <= \
           {r["label"] for r in rows}


def test_probe_steps_keys_are_the_probe_predictions():
    from epic_starter.memory.probe_steps import layout, step_dim
    from epic_starter.models.aga import AGAMemoryModel
    torch.manual_seed(3)
    with pytest.raises(ValueError):
        AGAMemoryModel(feat_dim=16, d=16, heads=2, mode="single", step_encoder="probe")
    model = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="none", step_encoder="probe").eval()
    x = torch.randn(2, 5, 1, step_dim(16))
    valid = torch.ones(2, 5, dtype=torch.bool)
    base = model(x, valid)
    # changing only the stored noun logits changes the noun memory's attention, not the verb memory's
    y = x.clone()
    y[..., slice(*layout(16)["noun_logits"])] = torch.randn(2, 5, 1, 300) * 5
    moved = model(y, valid)
    assert torch.allclose(base["attn_verb"], moved["attn_verb"])
    assert not torch.allclose(base["attn_noun"], moved["attn_noun"])
    with pytest.raises(ValueError, match="probe steps"):
        model(torch.randn(2, 5, 1, 16), valid)


def test_feature_keys_baseline_and_summary(toy, tmp_path):
    """Plain-attention baseline (--step-keys features) trains end to end, never mixes with AGA keys,
    and the summary groups runs over seeds."""
    from epic_starter.memory.probe_steps import precompute
    from epic_starter.models.aga import AGAMemoryModel
    events = load_events(toy / "train.csv")
    feats = tmp_path / "f16"
    write_mock_consolidated(events, feats, tokens=24, dim=16)
    ckpt = tmp_path / "probe.pt"
    probe_checkpoint(ckpt)
    steps = precompute(feats, ckpt, tmp_path / "psteps", batch_size=64, log=lambda m: None)
    train = script("train_sta_memory")
    common = ["--train-csv", str(toy / "train.csv"), "--features-dir", str(feats), "--step-features", str(steps),
              "--steps", "4", "--d-model", "16", "--heads", "2", "--num-workers", "0", "--device", "cpu",
              "--epochs", "1"]
    runs = toy / "runs"
    for seed in (1, 2):
        mo = runs / f"decoupled_S4_seed{seed}_psteps_fkeys_memonly"
        train.main(common + ["--seed", str(seed), "--step-keys", "features", "--present-encoder", "none",
                             "--out-dir", str(mo)])
        train.main(common + ["--seed", str(seed), "--step-keys", "features", "--present-init", str(ckpt),
                             "--freeze-present", "--memory-init", str(mo / "best.pt"), "--freeze-memory",
                             "--out-dir", str(runs / f"decoupled_S4_seed{seed}_psteps_fkeys_warm_frozenmem")])
    state = torch.load(runs / "decoupled_S4_seed1_psteps_fkeys_memonly" / "best.pt")["state_dict"]
    assert state["mem_v.ek.1.weight"].shape == (16, 16)        # keys read the 16-dim embeddings
    with pytest.raises(ValueError, match="does not match"):     # AGA-key model cannot load it
        AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="probe", step_encoder="probe").load_memory(
            runs / "decoupled_S4_seed1_psteps_fkeys_memonly" / "best.pt")
    with pytest.raises(ValueError):
        AGAMemoryModel(feat_dim=16, d=16, heads=2, step_keys="features")      # needs probe steps
    with pytest.raises(SystemExit):
        train.parse_args(["--train-csv", "x.csv", "--out-dir", "o", "--step-keys", "features"])
    groups = script("sta_memory_summary").main(["--runs-dir", str(runs), "--match", "fkeys",
                                                "--out", str(tmp_path / "summary.md")])
    assert set(groups) == {"decoupled_S4_psteps_fkeys_memonly", "decoupled_S4_psteps_fkeys_warm_frozenmem"}
    assert set(groups["decoupled_S4_psteps_fkeys_warm_frozenmem"]) == {1, 2}
    table = (tmp_path / "summary.md").read_text()
    assert "| decoupled_S4_psteps_fkeys_warm_frozenmem | 1,2 |" in table
    labels = {r["label"] for r in script("member4_gate_summary").main(
        ["--runs-dir", str(runs), "--steps", "4", "--out", str(tmp_path / "gate.md")])}
    assert "aga [memory probe steps (feature keys), pre-trained (frozen)]" in labels


def split_off_test_files(feats, test_events):
    """Move the test events' exact STA clips to Member 2's separate hidden-test files."""
    import csv as _csv
    rows = list(_csv.DictReader((feats / "events_index.csv").open(newline="")))
    ids = {e.narration_id for e in test_events}
    obs = np.load(feats / "sta_obs_tokens.npy")
    keep, test = [r for r in rows if r["narration_id"] not in ids], [r for r in rows if r["narration_id"] in ids]
    for name, part in (("events_index.csv", keep), ("test_events_index.csv", test)):
        with (feats / name).open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for i, r in enumerate(part):
                w.writerow({**r, "row": r["row"] if name == "events_index.csv" else i})
    np.save(feats / "sta_obs_test_tokens.npy", obs[[int(r["row"]) for r in test]])
    return len(test)


def test_cached_present_equals_frozen_probe_and_reads_test_clips(toy, tmp_path):
    from torch.utils.data import DataLoader
    from epic_starter.memory import HistoryDataset
    from epic_starter.memory.bank import anchors_with_test
    from epic_starter.memory.probe_steps import precompute
    from epic_starter.models.aga import AGAMemoryModel
    train_ev = load_events(toy / "train.csv")
    test_ev = load_events(toy / "test.csv", labeled=False)
    feats = tmp_path / "f16"
    write_mock_consolidated(train_ev + test_ev, feats, tokens=24, dim=16)
    n_test = split_off_test_files(feats, test_ev)
    assert n_test == len(test_ev)
    ckpt = tmp_path / "probe.pt"
    probe_checkpoint(ckpt)
    steps = precompute(feats, ckpt, tmp_path / "ps", batch_size=64, log=lambda m: None)
    assert json.loads((steps / "meta.json").read_text())["test_events"]
    assert (steps / "sta_obs_test_tokens.npy").is_file() and (steps / "test_events_index.csv").is_file()

    # the present branch and the memory steps both read the hidden-test STA clips
    bank, sbank = FeatureBank(root=feats), FeatureBank(root=steps, pooled=True)
    obs, sobs = anchors_with_test(feats, "sta_obs_tokens.npy"), anchors_with_test(steps, "sta_obs_tokens.npy")
    live = [e for e in test_ev if e.sta_cutoff >= 0]
    ds_probe = HistoryDataset(train_ev[:40] + live, bank, steps=4, anchors=obs, present_tokens=True,
                              step_bank=sbank, step_anchors=sobs)
    assert all(int(ds_probe[40 + i]["present_source"]) == 1 for i in range(len(live)))   # exact test clips
    ds_cached = HistoryDataset(train_ev[:40] + live, sbank, steps=4, anchors=sobs)

    # cached present == frozen probe on the present clip (same weights, fp16 storage)
    torch.manual_seed(5)
    m_probe = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="probe", freeze_present=True,
                             step_encoder="probe")
    m_probe.load_present(ckpt)
    for head in (m_probe.head_verb, m_probe.head_noun):          # a non-zero correction, to compare it too
        torch.nn.init.normal_(head[-1].weight, std=0.1)
    m_cached = AGAMemoryModel(feat_dim=16, d=16, heads=2, present_encoder="cached", step_encoder="probe")
    m_cached.load_state_dict({k: v for k, v in m_probe.state_dict().items() if not k.startswith("present.")})
    m_probe.eval(), m_cached.eval()
    bp = next(iter(DataLoader(ds_probe, batch_size=len(ds_probe))))
    bc = next(iter(DataLoader(ds_cached, batch_size=len(ds_cached))))
    assert torch.equal(bp["feats"], bc["feats"])
    with torch.no_grad():
        op = m_probe(bp["feats"], bp["valid"], bp["present_tokens"], bp["present_source"] > 0)
        oc = m_cached(bc["feats"], bc["valid"])
    for k in ("verb_logits", "noun_logits"):
        assert torch.allclose(op[k], oc[k], atol=5e-2), k

    # training: cached present without the token folder, starting at the probe alone
    train = script("train_sta_memory")
    common = ["--train-csv", str(toy / "train.csv"), "--test-csv", str(toy / "test.csv"),
              "--step-features", str(steps), "--steps", "4", "--d-model", "16", "--heads", "2",
              "--num-workers", "0", "--device", "cpu", "--epochs", "1"]
    runs = toy / "runs"
    train.main(common + ["--present-encoder", "none", "--out-dir", str(runs / "mo")])
    train.main(common + ["--present-encoder", "cached", "--memory-init", str(runs / "mo" / "best.pt"),
                         "--freeze-memory", "--out-dir", str(runs / "cached")])
    m = json.loads((runs / "cached" / "metrics.json").read_text())
    assert m["init"]["probe_alone"] and (runs / "cached" / "test_outputs.npz").is_file()
    train.main(common + ["--present-encoder", "cached", "--fusion", "three_way", "--branches", "present",
                         "--out-dir", str(runs / "cached_3way_P")])
    for bad in (["--present-encoder", "cached"], ["--present-encoder", "probe"]):
        with pytest.raises(SystemExit):              # cached needs probe steps; probe needs the tokens
            train.parse_args(["--train-csv", "x.csv", "--out-dir", "o"] + bad +
                             ([] if bad[1] == "cached" else ["--step-features", str(steps)]))
