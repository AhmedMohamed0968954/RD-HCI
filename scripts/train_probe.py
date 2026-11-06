"""Train & Evaluate Factorized Spatiotemporal Attentive Probe on TokenBank (AR & STA)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.data.manifests import read_manifest
from epic_starter.epic_metrics import evaluate
from epic_starter.models.factorized_probe import TokenBank, FactorizedAttentiveProbe


class ManifestTokenDataset(Dataset):
    def __init__(self, manifest_path: Path, bank: TokenBank, task: str):
        self.rows = [r for r in read_manifest(manifest_path) if r["narration_id"] in bank.row]
        self.bank = bank
        self.task = task

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        nid = r["narration_id"]
        if self.task == "AR":
            tokens = self.bank.ar_tokens(nid)
            use_prior = False
        else:
            tokens, use_prior = self.bank.sta_obs_tokens(nid, sta_cutoff=r["sta_cutoff"])
        verb = -1 if r["verb"] is None else int(r["verb"])
        noun = -1 if r["noun"] is None else int(r["noun"])
        return tokens, verb, noun, use_prior, nid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--task", choices=["AR", "STA"], default="AR")
    parser.add_argument("--mode", choices=["factorized", "full_attn", "mean_pool"], default="factorized")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--save-ckpt", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_dir = Path(args.manifest_dir)
    bank = TokenBank(args.feature_dir, verify_leakage=True)

    train_ds = ManifestTokenDataset(m_dir / f"train_{args.task.lower()}.jsonl", bank, args.task)
    val_ds = ManifestTokenDataset(m_dir / f"val_{args.task.lower()}.jsonl", bank, args.task)
    print(f"[{args.task} | {args.mode}] Train events: {len(train_ds)} | Val events: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    priors = json.loads((m_dir / "priors.json").read_text(encoding="utf-8"))
    prior_v_logits = torch.log(torch.tensor(priors["verb_output"], dtype=torch.float32) + 1e-8)
    prior_n_logits = torch.log(torch.tensor(priors["noun_output"], dtype=torch.float32) + 1e-8)

    model = FactorizedAttentiveProbe(mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for tokens, yv, yn, use_prior, _ in train_loader:
            valid = ~use_prior
            if not valid.any():
                continue
            tokens, yv, yn = tokens[valid].to(device), yv[valid].to(device), yn[valid].to(device)
            optimizer.zero_grad()
            out = model(tokens)
            loss = criterion(out["verb_output"], yv) + criterion(out["noun_output"], yn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        scheduler.step()

        # Validation
        model.eval()
        all_v_scores, all_n_scores, all_yv, all_yn = [], [], [], []
        with torch.no_grad():
            for tokens, yv, yn, use_prior, _ in val_loader:
                out = model(tokens.to(device))
                v_log = out["verb_output"].cpu()
                n_log = out["noun_output"].cpu()
                if use_prior.any():
                    v_log[use_prior] = prior_v_logits
                    n_log[use_prior] = prior_n_logits
                all_v_scores.append(v_log.numpy())
                all_n_scores.append(n_log.numpy())
                all_yv.append(yv.numpy())
                all_yn.append(yn.numpy())

        metrics = evaluate(
            np.concatenate(all_v_scores),
            np.concatenate(all_n_scores),
            np.concatenate(all_yv),
            np.concatenate(all_yn),
            task=args.task,
            score_type="logits",
        )["all"]
        primary = metrics["primary_score"]
        print(
            f"Epoch {epoch:02d} | Loss: {total_loss/max(1,steps):.4f} | "
            f"Verb Top1: {metrics['verb_top1']:.2f}% | Noun Top1: {metrics['noun_top1']:.2f}% | "
            f"Action Top1: {metrics['action_top1']:.2f}% | Action MT5R: {metrics['action_mt5r']:.2f}%"
        )
        if primary > best_score:
            best_score = primary
            if args.save_ckpt:
                Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "metrics": metrics, "epoch": epoch}, args.save_ckpt)


if __name__ == "__main__":
    main()
