from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from epic_starter.data.events import Event
from epic_starter.memory import (Anchors, AuxLabeler, FeatureBank, HistoryDataset, build_history,
                                 make_mock_bank, step_times, write_mock_consolidated)
from epic_starter.models.aga import AGAMemoryModel, GatedAGAModel, aga_loss


def ev(start, stop=None, nid="e", video="P01_01", verb=1, noun=2):
    return Event(nid, "P01", video, start, stop or start + 1, start - 1, verb, noun)


def dense_bank(end=60.0, stride=0.1):
    return make_mock_bank([ev(end - 1, end)], stride=stride, tokens=3, dim=4)


def test_steps_ascending_and_last_is_cutoff():
    taus = step_times(19.73, steps=16, stride=1.0)
    assert np.all(np.diff(taus) > 0) and taus[-1] == pytest.approx(19.73)
    assert taus[0] == pytest.approx(4.73)


def test_never_uses_a_clip_after_its_step_or_the_cutoff():
    bank = dense_bank()
    e = ev(20.73)                                   # cutoff 19.73, between grid points
    h = build_history(e, bank["P01_01"], steps=16)
    assert h["valid"].all()
    assert (h["used_times"] <= h["step_times"] + 1e-6).all()
    assert h["used_times"].max() <= e.sta_cutoff + 1e-6
    # the mock encodes each clip's end time in channel 0: check the features really are those clips
    assert np.allclose(h["feats"][:, 0, 0].numpy(), h["used_times"], atol=1e-2)
    # snapping is to the latest legal clip (grid 0.1 s -> 19.7, not 19.8)
    assert h["used_times"][-1] == pytest.approx(19.7)


def test_negative_cutoff_has_no_history():
    h = build_history(ev(0.6), None, steps=16, shape=(3, 4))
    assert not h["valid"].any() and not h["feats"].any()


def test_early_event_masks_steps_before_video_start():
    h = build_history(ev(4.0), dense_bank()["P01_01"], steps=16)   # cutoff 3.0
    assert h["valid"].sum() == 3                                  # taus 1.0, 2.0, 3.0 are >= first clip (0.1)
    assert not h["valid"][:13].any()


def test_gap_in_bank_is_masked_not_filled_with_old_clip():
    bank = FeatureBank({"P01_01": {"times": np.array([1.0, 2.0, 10.0]),
                                   "feats": torch.zeros(3, 3, 4), "clip_seconds": 2.0}})
    h = build_history(ev(9.0), bank["P01_01"], steps=4, stride=1.0)  # taus 5,6,7,8
    assert not h["valid"].any()


def test_anchor_replaces_last_step_with_exact_clip():
    anchor = torch.full((3, 4), 7.0)
    h = build_history(ev(20.73), dense_bank()["P01_01"], steps=4, anchor=anchor)
    assert h["used_times"][-1] == pytest.approx(19.73)
    assert torch.all(h["feats"][-1] == 7.0)


def test_aux_label_is_first_action_starting_in_window():
    events = [ev(10.0, nid="a", verb=3, noun=4), ev(11.0, nid="b", verb=5, noun=6), ev(30.0, nid="c")]
    lab = AuxLabeler(events, lo=0.5, hi=2.5)
    verbs, nouns = lab("P01_01", np.array([7.0, 8.0, 9.0, 20.0]))
    # tau 7 -> window [7.5, 9.5]: none; tau 8 -> [8.5,10.5]: a; tau 9 -> [9.5,11.5]: a first; tau 20: none
    assert verbs.tolist() == [-1, 3, 3, -1] and nouns.tolist() == [-1, 4, 4, -1]


def test_bank_rejects_unsorted_times():
    with pytest.raises(ValueError):
        FeatureBank({"v": {"times": np.array([2.0, 1.0]), "feats": torch.zeros(2, 3, 4)}})


def test_consolidated_layout_matches_in_memory_bank(tmp_path):
    events = [ev(12.0, nid="a"), ev(6.0, nid="b", video="P02_01")]
    write_mock_consolidated(events, tmp_path, stride=0.5, tokens=3, dim=4)
    mem = make_mock_bank(events, stride=0.5, tokens=3, dim=4)
    disk = FeatureBank(root=tmp_path)
    assert disk.shape == (3, 4) and "P02_01" in disk
    h_disk = build_history(events[0], disk["P01_01"], steps=8)
    h_mem = build_history(events[0], mem["P01_01"], steps=8)
    assert np.allclose(h_disk["used_times"], h_mem["used_times"], equal_nan=True)
    assert torch.allclose(h_disk["feats"], h_mem["feats"])


def test_consolidated_anchor_is_used_and_checked(tmp_path):
    events = [ev(12.0, nid="a")]
    write_mock_consolidated(events, tmp_path, tokens=3, dim=4)
    anchors = Anchors(tmp_path)
    ds = HistoryDataset(events, FeatureBank(root=tmp_path), steps=4, anchors=anchors)
    item = ds[0]
    assert item["used_times"][-1] == pytest.approx(11.0)
    assert torch.allclose(item["feats"][-1, :, 0], torch.full((3,), 11.0))
    # a clip that ends after the cutoff must be refused
    late = Event("a", "P01", "P01_01", 11.5, 12.5, 10.5, 1, 2)
    with pytest.raises(AssertionError):
        anchors.for_event(late)


def test_consolidated_bank_works_with_dataloader_workers(tmp_path):
    events = [ev(10.0 + i, nid=f"e{i}") for i in range(8)]
    write_mock_consolidated(events, tmp_path, tokens=3, dim=4)
    ds = HistoryDataset(events, FeatureBank(root=tmp_path), steps=4, anchors=Anchors(tmp_path))
    batches = list(DataLoader(ds, batch_size=4, num_workers=2))
    assert sum(len(b["narration_id"]) for b in batches) == 8


@pytest.mark.parametrize("mode", ["single", "decoupled"])
def test_dataset_to_model_end_to_end(mode):
    events = [ev(0.5, nid="cold"), ev(4.0, nid="early"), ev(25.0, nid="full"), ev(40.0, nid="late", verb=96, noun=299)]
    bank = make_mock_bank(events, stride=0.5, tokens=24, dim=16)
    ds = HistoryDataset(events, bank, steps=16, aux=AuxLabeler(events))
    batch = next(iter(DataLoader(ds, batch_size=4)))
    assert batch["feats"].shape == (4, 16, 24, 16)
    assert batch["use_prior"].tolist() == [True, False, False, False]
    model = AGAMemoryModel(feat_dim=16, d=32, heads=4, mode=mode)
    out = model(batch["feats"], batch["valid"])
    assert out["verb_logits"].shape == (4, 97) and out["noun_logits"].shape == (4, 300)
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))
    loss = aga_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_model_output_does_not_depend_on_masked_steps():
    torch.manual_seed(0)
    model = AGAMemoryModel(feat_dim=8, d=16, heads=2).eval()
    feats = torch.randn(1, 6, 3, 8)
    valid = torch.tensor([[False, False, True, True, True, True]])
    a = model(feats, valid)["verb_logits"]
    feats[:, :2] = torch.randn(1, 2, 3, 8) * 100   # garbage in masked steps must not matter
    b = model(feats, valid)["verb_logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_legacy_ambiguous_clip_end_column_is_rejected(tmp_path):
    events = [ev(12.0, nid="a")]
    write_mock_consolidated(events, tmp_path, tokens=3, dim=4)
    idx = tmp_path / "events_index.csv"
    idx.write_text("narration_id,row,video_id,clip_end_s\na,0,P01_01,11.0\n")
    with pytest.raises(ValueError, match="sta_clip_end_s"):
        Anchors(tmp_path)


@pytest.mark.parametrize("mode", ["single", "decoupled"])
def test_encode_exposes_history_and_present_per_field(mode):
    model = AGAMemoryModel(feat_dim=8, d=16, heads=2, mode=mode).eval()
    feats = torch.randn(3, 5, 3, 8)
    valid = torch.tensor([[True] * 5, [False, False, False, False, True], [False] * 5])
    z = model.encode(feats, valid)
    for k in ("verb_history", "noun_history", "present"):
        assert z[k].shape == (3, 16) and torch.isfinite(z[k]).all()
    assert z["history_available"].tolist() == [True, False, False]
    assert z["present_available"].tolist() == [True, True, False]


def test_member4_gate_plugs_into_memory():
    gate = pytest.importorskip("epic_starter.models.gate")
    model = GatedAGAModel(AGAMemoryModel(feat_dim=8, d=16, heads=2), gate.GatedVerbNounHead(in_dim=16))
    valid = torch.tensor([[True] * 5, [False] * 4 + [True]])
    out = model(torch.randn(2, 5, 3, 8), valid)
    assert out["verb_logits"].shape == (2, 97) and out["noun_logits"].shape == (2, 300)
    w = out["verb_gates"]
    assert torch.allclose(w[:, 2], torch.zeros(2))                    # no future branch yet
    assert w[1, 0] == 0 and torch.allclose(w.sum(1), torch.ones(2))   # no history -> present only
    out["verb_logits"].sum().backward()


def test_extractor_edge_rows_are_ignored(tmp_path):
    """Member 2's extractor writes t_end_s = -1 for undecodable clips and may repeat the last
    frame time at the video end; sta_use_prior rows keep a zero block that must not be used."""
    events = [ev(12.0, nid="a"), ev(12.5, nid="b")]
    write_mock_consolidated(events, tmp_path, tokens=3, dim=4)
    lines = (tmp_path / "bank_index.csv").read_text().splitlines()
    lines[1] = lines[1].replace(lines[1].split(",")[1], "-1.0")          # first clip undecodable
    last_t = lines[-1].split(",")[1]
    lines[-2] = ",".join([lines[-2].split(",")[0], last_t, lines[-2].split(",")[2]])  # duplicate end time
    (tmp_path / "bank_index.csv").write_text("\n".join(lines) + "\n")
    bank = FeatureBank(root=tmp_path)
    assert bank.dropped == {"undecodable": 1, "duplicate_time": 1}
    assert (np.diff(bank["P01_01"]["times"]) > 0).all() and bank["P01_01"]["times"][0] > 0
    idx = (tmp_path / "events_index.csv").read_text()
    (tmp_path / "events_index.csv").write_text(idx.replace(",11.500000,False", ",-1.0,True"))  # event b
    anchors = Anchors(tmp_path)
    assert anchors.for_event(events[1]) is None          # cutoff >= 0 but no observable frame
    assert anchors.for_event(events[0]) is not None
