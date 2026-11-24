"""
Member 2 Final Hidden-Test Inference & Codabench Packaging:
Runs 3-seed ensemble of AR_factorized and STA_factorized on extracted hidden-test tokens,
applies Member 1's marginal logit adjustment + cold_start_prior.json for cutoff < 0 events,
and saves a validated submission.zip via epic_starter.epic_metrics.save_submission().
"""
from __future__ import annotations
import argparse
import json
import json.encoder
json.encoder.c_make_encoder = None
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.calibration import NUM_VERBS, NUM_NOUNS, adjust, class_log_prior, cold_start_logits
from epic_starter.data.manifests import read_manifest
from epic_starter.epic_metrics import save_submission
from epic_starter.models.factorized_probe import FactorizedAttentiveProbe


@torch.no_grad()
def predict_ensemble(
    tokens_npy: Path,
    ckpt_paths: list[Path],
    mode: str,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(tokens_npy, mmap_mode="r")
    n = len(arr)
    v_sum = np.zeros((n, NUM_VERBS), dtype=np.float64)
    n_sum = np.zeros((n, NUM_NOUNS), dtype=np.float64)

    for ckpt_p in ckpt_paths:
        model = FactorizedAttentiveProbe(mode=mode).to(device)
        state = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(np.array(arr[start:end], dtype=np.float32)).to(device)
            out = model(x)
            v_sum[start:end] += out["verb_output"].cpu().numpy()
            n_sum[start:end] += out["noun_output"].cpu().numpy()

    k = float(len(ckpt_paths))
    return (v_sum / k).astype(np.float32), (n_sum / k).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", default="manifests/hybrid-v3")
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--results-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    p.add_argument("--sta-tau-v", type=float, default=0.25)
    p.add_argument("--sta-tau-n", type=float, default=0.50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_dir = Path(args.manifest_dir)
    f_dir = Path(args.feature_dir)
    r_root = Path(args.results_root)

    train_sta = read_manifest(m_dir / "train_sta.jsonl")
    test_rows = read_manifest(m_dir / "test_sta.jsonl")
    expected_ids = [r["narration_id"] for r in test_rows]

    test_idx = pd.read_csv(f_dir / "test_events_index.csv")
    assert list(test_idx["narration_id"]) == expected_ids, "test_events_index.csv order mismatch!"
    use_prior = test_idx["sta_use_prior"].astype(bool).to_numpy()

    # 1. AR 3-seed ensemble (tau = 0, 0)
    ar_ckpts = [r_root / f"AR_factorized_seed{s}" / "best.pt" for s in args.seeds]
    ar_v, ar_n = predict_ensemble(f_dir / "ar_test_tokens.npy", ar_ckpts, "factorized", device)

    # 2. STA 3-seed ensemble + marginal logit adjustment + cold-start prior (tau=0 for cutoff < 0)
    sta_ckpts = [r_root / f"STA_factorized_seed{s}" / "best.pt" for s in args.seeds]
    sta_v, sta_n = predict_ensemble(f_dir / "sta_obs_test_tokens.npy", sta_ckpts, "factorized", device)

    log_pi_v = class_log_prior([r["verb"] for r in train_sta], NUM_VERBS)
    log_pi_n = class_log_prior([r["noun"] for r in train_sta], NUM_NOUNS)
    sta_v = adjust(sta_v, log_pi_v, tau=args.sta_tau_v).astype(np.float32)
    sta_n = adjust(sta_n, log_pi_n, tau=args.sta_tau_n).astype(np.float32)

    cs_prior = json.loads((m_dir / "cold_start_prior.json").read_text(encoding="utf-8"))
    cs_v, cs_n = cold_start_logits(cs_prior, log_pi_v, log_pi_n, tau_v=0.0, tau_n=0.0)
    sta_v[use_prior] = cs_v
    sta_n[use_prior] = cs_n

    ar_preds = [{"narration_id": nid, "verb_output": ar_v[i], "noun_output": ar_n[i]} for i, nid in enumerate(expected_ids)]
    sta_preds = [{"narration_id": nid, "verb_output": sta_v[i], "noun_output": sta_n[i]} for i, nid in enumerate(expected_ids)]

    zip_path = save_submission(ar_preds, sta_preds, expected_ids, args.out_dir)
    print(f"Validated Codabench submission saved to: {zip_path} ({len(expected_ids)} events, {use_prior.sum()} cold-start STA)")


if __name__ == "__main__":
    main()
