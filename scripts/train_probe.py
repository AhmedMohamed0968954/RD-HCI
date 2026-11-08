"""
Member 2 Deliverable 2B: Train & Evaluate Factorized Spatiotemporal Attentive Probe (AR & STA)
on the full 67,217-event V-JEPA 2.1 TokenBank.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import json.encoder
json.encoder.c_make_encoder = None
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
    def __init__(self, rows: list[dict], bank: TokenBank, task: str, load_pred: bool = False):
        self.rows = rows
        self.bank = bank
        self.task = task
        self.load_pred = load_pred

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        nid = r["narration_id"]
        if self.task == "AR":
            tokens = self.bank.ar_tokens(nid)
            pred_tokens = tokens
            use_prior = False
        else:
            tokens, use_prior = self.bank.sta_obs_tokens(nid, sta_cutoff=r["sta_cutoff"])
            if self.load_pred:
                pred_tokens, _ = self.bank.sta_pred_tokens(nid, sta_cutoff=r["sta_cutoff"])
            else:
                pred_tokens = tokens
        verb = -1 if r["verb"] is None else int(r["verb"])
        noun = -1 if r["noun"] is None else int(r["noun"])
        return tokens, pred_tokens, verb, noun, use_prior, nid


def compute_subset_masks(train_rows: list[dict], val_rows: list[dict]) -> dict[str, np.ndarray]:
    train_participants = {r["participant_id"] for r in train_rows}
    unseen_mask = np.array([r["participant_id"] not in train_participants for r in val_rows], dtype=bool)

    action_counts = Counter(int(r["verb"]) * 300 + int(r["noun"]) for r in train_rows if r["verb"] is not None)
    total_train = sum(action_counts.values())
    head_actions = set()
    running = 0
    for act_id, count in action_counts.most_common():
        if running < 0.80 * total_train:
            head_actions.add(act_id)
            running += count
        else:
            break
    tail_mask = np.array(
        [(int(r["verb"]) * 300 + int(r["noun"])) not in head_actions for r in val_rows],
        dtype=bool,
    )
    return {"unseen": unseen_mask, "tail": tail_mask}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task", choices=["AR", "STA"], default="AR")
    parser.add_argument("--mode", choices=["factorized", "factorized_pred", "full_attn", "mean_pool"], default="factorized")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_dir = Path(args.manifest_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = TokenBank(args.feature_dir, verify_leakage=True)
    all_train_rows = read_manifest(m_dir / f"train_{args.task.lower()}.jsonl")
    all_val_rows = read_manifest(m_dir / f"val_{args.task.lower()}.jsonl")

    train_rows = [r for r in all_train_rows if r["narration_id"] in bank.row]
    val_rows = [r for r in all_val_rows if r["narration_id"] in bank.row]
    if not train_rows or not val_rows:
        raise RuntimeError(f"Insufficient matched events: train={len(train_rows)}, val={len(val_rows)}")

    load_pred = args.mode == "factorized_pred" and args.task == "STA"
    subsets = compute_subset_masks(all_train_rows, val_rows)
    train_ds = ManifestTokenDataset(train_rows, bank, args.task, load_pred=load_pred)
    val_ds = ManifestTokenDataset(val_rows, bank, args.task, load_pred=load_pred)
    print(
        f"[{args.task} | {args.mode}] Train: {len(train_ds)}/{len(all_train_rows)} | "
        f"Val: {len(val_ds)}/{len(all_val_rows)} (Unseen: {subsets['unseen'].sum()}, Tail: {subsets['tail'].sum()})"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    priors = json.loads((m_dir / "priors.json").read_text(encoding="utf-8"))
    prior_v_logits = torch.log(torch.tensor(priors["verb_output"], dtype=torch.float32) + 1e-8)
    prior_n_logits = torch.log(torch.tensor(priors["noun_output"], dtype=torch.float32) + 1e-8)

    model = FactorizedAttentiveProbe(mode=args.mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_score = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for tokens, pred_tokens, yv, yn, use_prior, _ in train_loader:
            valid = ~use_prior
            if not valid.any():
                continue
            tokens = tokens[valid].to(device)
            pred_t = pred_tokens[valid].to(device) if load_pred else None
            yv, yn = yv[valid].to(device), yn[valid].to(device)
            optimizer.zero_grad()
            out = model(tokens, pred_tokens=pred_t)
            loss = criterion(out["verb_output"], yv) + criterion(out["noun_output"], yn)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        scheduler.step()

        model.eval()
        all_v_scores, all_n_scores, all_yv, all_yn, all_prior, all_nids = [], [], [], [], [], []
        with torch.no_grad():
            for tokens, pred_tokens, yv, yn, use_prior, nids in val_loader:
                pred_t = pred_tokens.to(device) if load_pred else None
                out = model(tokens.to(device), pred_tokens=pred_t)
                v_log = out["verb_output"].cpu()
                n_log = out["noun_output"].cpu()
                if use_prior.any():
                    v_log[use_prior] = prior_v_logits
                    n_log[use_prior] = prior_n_logits
                all_v_scores.append(v_log.numpy())
                all_n_scores.append(n_log.numpy())
                all_yv.append(yv.numpy())
                all_yn.append(yn.numpy())
                all_prior.append(use_prior.numpy())
                all_nids.extend(nids)

        v_scores_np = np.concatenate(all_v_scores)
        n_scores_np = np.concatenate(all_n_scores)
        yv_np = np.concatenate(all_yv)
        yn_np = np.concatenate(all_yn)
        prior_np = np.concatenate(all_prior)

        metrics = evaluate(
            v_scores_np,
            n_scores_np,
            yv_np,
            yn_np,
            task=args.task,
            score_type="logits",
            subsets=subsets,
        )
        m_all = metrics["all"]
        m_tail = metrics["tail"]
        primary = float(m_all["primary_score"])
        epoch_record = {"epoch": int(epoch), "train_loss": float(total_loss / max(1, steps)), "metrics": metrics}
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | Loss: {epoch_record['train_loss']:.4f} | "
            f"Act Top1: {m_all['action_top1']:.2f}% | Act MT5R: {m_all['action_mt5r']:.2f}% | "
            f"Verb Top1: {m_all['verb_top1']:.2f}% | Noun Top1: {m_all['noun_top1']:.2f}% | "
            f"Tail Top1: {m_tail['action_top1']:.2f}% | Tail MT5R: {m_tail['action_mt5r']:.2f}%"
        )

        if primary > best_score:
            best_score = primary
            torch.save(
                {"model": model.state_dict(), "mode": args.mode, "task": args.task, "epoch": int(epoch), "metrics": metrics},
                out_dir / "best.pt",
            )
            np.savez(
                out_dir / "val_outputs.npz",
                narration_id=np.array(all_nids),
                verb_logits=v_scores_np,
                noun_logits=n_scores_np,
                verb_labels=yv_np,
                noun_labels=yn_np,
                use_prior=prior_np,
            )
            (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Finished {args.task} ({args.mode}) -> Best primary score: {best_score:.2f}% saved to {out_dir}")


if __name__ == "__main__":
    main()
