"""Member 3 checkpoint demo: STA history loader + AGA memory on a mock feature bank.

Uses the real annotation CSV and the shared grouped split, but random features
(make_mock_bank), so it runs before feature extraction exists. When real features
are ready, pass --features-dir team/features/<version> and the same code reads the
consolidated arrays (bank_tokens.npy, sta_obs_tokens.npy) instead.

    python scripts/demo_memory.py --train-csv "$EPIC_PROJECT/annotations/EPIC_100_train.csv"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from epic_starter.data.events import load_events
from epic_starter.data.manifests import grouped_split
from epic_starter.epic_metrics import evaluate
from epic_starter.memory import Anchors, AuxLabeler, FeatureBank, HistoryDataset, make_mock_bank
from epic_starter.models.aga import AGAMemoryModel, aga_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--features-dir", help="consolidated feature version folder "
                   "(team/features/<version>); omit to use a mock bank")
    p.add_argument("--limit-videos", type=int, default=20, help="keep the demo small")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--mode", choices=["single", "decoupled"], default="decoupled")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    torch.manual_seed(a.seed)

    events = load_events(Path(a.train_csv))
    train, val = grouped_split(events, .2, a.seed)            # the team's shared split
    keep_t = sorted({e.video_id for e in train})[: a.limit_videos]
    keep_v = sorted({e.video_id for e in val})[: max(1, a.limit_videos // 4)]
    train = [e for e in train if e.video_id in keep_t]
    val = [e for e in val if e.video_id in keep_v]
    if a.features_dir:
        bank, anchors = FeatureBank(root=a.features_dir), Anchors(a.features_dir)
        print("feature version:", bank.meta)
    else:
        bank, anchors = make_mock_bank(train + val, stride=1.0, tokens=24, dim=64, encode_time=False), None
    print(f"train {len(train)} events / {len(keep_t)} videos, val {len(val)} / {len(keep_v)}; "
          f"bank [T,D] = {bank.shape}; {'folder' if a.features_dir else 'in-memory MOCK'} features")

    tr = HistoryDataset(train, bank, a.steps, a.stride, aux=AuxLabeler(train), anchors=anchors)
    va = HistoryDataset(val, bank, a.steps, a.stride, anchors=anchors)  # no aux labels needed
    model = AGAMemoryModel(feat_dim=bank.shape[1], mode=a.mode)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    for epoch in range(a.epochs):
        model.train()
        t0, losses = time.time(), []
        for batch in DataLoader(tr, batch_size=a.batch_size, shuffle=True):
            loss = aga_loss(model(batch["feats"], batch["valid"]), batch)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"epoch {epoch}: loss {np.mean(losses):.3f} ({time.time()-t0:.1f}s)")

    model.eval()
    vs, ns, yv, yn = [], [], [], []
    with torch.no_grad():
        for batch in DataLoader(va, batch_size=a.batch_size):
            out = model(batch["feats"], batch["valid"])
            vs.append(out["verb_logits"]); ns.append(out["noun_logits"])
            yv.append(batch["verb_label"]); yn.append(batch["noun_label"])
    res = evaluate(torch.cat(vs), torch.cat(ns), torch.cat(yv), torch.cat(yn), task="STA")["all"]
    print({k: round(v, 2) for k, v in res.items() if isinstance(v, float)})
    if not a.features_dir or bank.meta.get("backbone") == "MOCK":
        print("Mock features carry no visual signal: these numbers only prove the pipeline runs.")


if __name__ == "__main__":
    main()
