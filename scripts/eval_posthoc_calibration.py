"""
Plan Item 3 (Members 1 + 2):
1. Stratified Seen/Unseen half-validation split (val_tune vs val_test).
2. Diagnostic count of validation action classes unseen in hybrid_train.csv.
3. Controlled 3-seed comparison on val_test (hyperparameters tuned on val_tune):
   - Raw (tau_v=0, tau_n=0)
   - Marginal Logit Adjustment (tau_v, tau_n)
   - + Finite-beta Feasibility Mask: -beta * I[C_train(v, n) == 0]
   - + Smoothed PMI: +gamma * log(P_a(v, n) / (P_a(v) P_a(n)))
   - + Smoothed Joint Prior: +gamma * log P_a(v, n)
"""
from __future__ import annotations
import json
import json.encoder
json.encoder.c_make_encoder = None
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.calibration import NUM_VERBS, NUM_NOUNS, adjust, class_log_prior
from epic_starter.data.manifests import read_manifest
from epic_starter.epic_metrics import action_top5, probabilities, scores_for


def stratified_split_halves(val_rows: list[dict], unseen_mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Split validation videos 50/50 within Seen and Unseen participant groups separately."""
    rng = np.random.default_rng(seed)
    seen_vids = sorted({r["video_id"] for r, u in zip(val_rows, unseen_mask) if not u})
    unseen_vids = sorted({r["video_id"] for r, u in zip(val_rows, unseen_mask) if u})
    rng.shuffle(seen_vids)
    rng.shuffle(unseen_vids)
    tune_vids = set(seen_vids[: len(seen_vids) // 2]) | set(unseen_vids[: len(unseen_vids) // 2])
    return np.array([r["video_id"] in tune_vids for r in val_rows], dtype=bool)


def build_cooccurrence_matrices(train_rows: list[dict], alpha: float = 0.05):
    counts = np.zeros((NUM_VERBS, NUM_NOUNS), dtype=np.float64)
    for r in train_rows:
        if r["verb"] is not None and r["noun"] is not None:
            counts[int(r["verb"]), int(r["noun"])] += 1.0
    unseen_pair = counts == 0.0
    sm = counts + alpha
    p_vn = sm / sm.sum()
    p_v = p_vn.sum(axis=1, keepdims=True)
    p_n = p_vn.sum(axis=0, keepdims=True)
    pmi = np.log(p_vn / (p_v * p_n))
    log_joint = np.log(p_vn)
    return counts, unseen_pair, pmi, log_joint


def pairwise_action_top5(vp: np.ndarray, np_: np.ndarray, pair_bonus: np.ndarray | None = None, k: int = 20) -> np.ndarray:
    """Top-5 joint action IDs (verb*300+noun) under log p_v + log p_n + pair_bonus[v, n]."""
    if pair_bonus is None:
        return action_top5(vp, np_, k=10)
    n = len(vp)
    log_vp = np.log(np.maximum(vp, 1e-12))
    log_np = np.log(np.maximum(np_, 1e-12))
    cv = np.argpartition(-log_vp, k, axis=1)[:, :k]
    cn = np.argpartition(-log_np, k, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    base = log_vp[rows, cv][:, :, None] + log_np[rows, cn][:, None, :]
    bonus = pair_bonus[cv[:, :, None], cn[:, None, :]]
    score = (base + bonus).reshape(n, -1)
    ids = (cv[:, :, None] * NUM_NOUNS + cn[:, None, :]).reshape(n, -1)
    top5_idx = np.argpartition(-score, 5, axis=1)[:, :5]
    top5_scores = np.take_along_axis(score, top5_idx, axis=1)
    top5_ids = np.take_along_axis(ids, top5_idx, axis=1)
    order = np.lexsort((top5_ids, -top5_scores), axis=1)
    return np.take_along_axis(top5_ids, order, axis=1)


def eval_action_subsets(topa: np.ndarray, ya: np.ndarray, unseen_m: np.ndarray, tail_m: np.ndarray, metric: str):
    out = {}
    for name, m in [("all", np.ones(len(ya), dtype=bool)), ("seen", ~unseen_m), ("unseen", unseen_m), ("tail", tail_m)]:
        s = scores_for(ya[m], topa[m])
        out[name] = s[metric]
    return out


def main():
    m_dir = Path("manifests/hybrid-v3")
    res_root = Path("/lustre1/scratch/394/vsc39484/epic-kitchens/team/results/member2_probe_hybrid")
    tail_info = json.loads((m_dir / "tail_classes.json").read_text(encoding="utf-8"))
    tail_verbs = set(tail_info["verbs"])
    tail_nouns = set(tail_info["nouns"])

    for task, model_name, metric in [
        ("AR", "AR_factorized", "top1"),
        ("STA", "STA_factorized", "mt5r"),
        ("STA", "STA_factorized_pred_1664", "mt5r"),
    ]:
        train_rows = read_manifest(m_dir / f"train_{task.lower()}.jsonl")
        val_rows = read_manifest(m_dir / f"val_{task.lower()}.jsonl")
        val_by_id = {r["narration_id"]: r for r in val_rows}
        train_parts = {r["participant_id"] for r in train_rows}

        log_pi_v = class_log_prior([r["verb"] for r in train_rows], NUM_VERBS)
        log_pi_n = class_log_prior([r["noun"] for r in train_rows], NUM_NOUNS)
        _, unseen_pair, pmi, log_joint = build_cooccurrence_matrices(train_rows, alpha=0.05)

        if model_name in {"AR_factorized", "STA_factorized"}:
            val_ya = np.array([int(r["verb"]) * NUM_NOUNS + int(r["noun"]) for r in val_rows])
            val_u = np.array([r["participant_id"] not in train_parts for r in val_rows])
            val_t = np.array([(int(r["verb"]) in tail_verbs) or (int(r["noun"]) in tail_nouns) for r in val_rows])
            print(f"\n=== [{task}] Unseen (v, n) Pairs Diagnostic on Validation ===")
            for sname, smask in [("All", np.ones(len(val_ya), dtype=bool)), ("Seen", ~val_u), ("Unseen", val_u), ("Tail", val_t)]:
                uniq_a = np.unique(val_ya[smask])
                unseen_cls = sum(1 for a in uniq_a if unseen_pair[a // NUM_NOUNS, a % NUM_NOUNS])
                unseen_ev = sum(1 for a in val_ya[smask] if unseen_pair[a // NUM_NOUNS, a % NUM_NOUNS])
                print(f"  {sname:7s}: {unseen_cls}/{len(uniq_a)} unique action classes ({100*unseen_cls/len(uniq_a):.1f}%) | "
                      f"{unseen_ev}/{smask.sum()} events ({100*unseen_ev/smask.sum():.1f}%) never seen in train")

        taus = [0.0, 0.25, 0.5, 0.75, 1.0] if task == "STA" else [0.0, 0.1, 0.2, 0.3]
        betas = [0.5, 1.0, 2.0, 3.0, 5.0]
        gammas = [0.1, 0.25, 0.5, 0.75, 1.0]

        methods = [
            "Raw (tau=0)",
            "Logit Adj (tau_v, tau_n)",
            "+ Feasibility Mask (-beta)",
            "+ Smoothed PMI (+gamma)",
            "+ Joint Prior (+gamma)",
        ]
        agg = {m_key: {"all": [], "seen": [], "unseen": [], "tail": []} for m_key in methods}
        chosen_params = []

        for seed in [42, 1, 2]:
            d = np.load(res_root / f"{model_name}_seed{seed}" / "val_outputs.npz")
            ordered_rows = [val_by_id[str(nid)] for nid in d["narration_id"]]
            zv, zn = d["verb_logits"], d["noun_logits"]
            yv, yn = d["verb_labels"], d["noun_labels"]
            ya = yv * NUM_NOUNS + yn
            unseen_m = np.array([r["participant_id"] not in train_parts for r in ordered_rows])
            tail_m = np.array([(int(v) in tail_verbs) or (int(n) in tail_nouns) for v, n in zip(yv, yn)])
            tune_m = stratified_split_halves(ordered_rows, unseen_m, seed=0)
            test_m = ~tune_m

            vp0 = probabilities(zv[test_m], NUM_VERBS, "logits")
            np0 = probabilities(zn[test_m], NUM_NOUNS, "logits")
            for k, v in eval_action_subsets(action_top5(vp0, np0), ya[test_m], unseen_m[test_m], tail_m[test_m], metric).items():
                agg["Raw (tau=0)"][k].append(v)

            best_score, best_tau = -1.0, (0.0, 0.0)
            for tv in taus:
                vp_t = probabilities(adjust(zv[tune_m], log_pi_v, tv), NUM_VERBS, "logits")
                for tn in taus:
                    np_t = probabilities(adjust(zn[tune_m], log_pi_n, tn), NUM_NOUNS, "logits")
                    sc = scores_for(ya[tune_m], action_top5(vp_t, np_t))[metric]
                    if sc > best_score:
                        best_score, best_tau = sc, (tv, tn)

            vp_tune = probabilities(adjust(zv[tune_m], log_pi_v, best_tau[0]), NUM_VERBS, "logits")
            np_tune = probabilities(adjust(zn[tune_m], log_pi_n, best_tau[1]), NUM_NOUNS, "logits")
            vp_test = probabilities(adjust(zv[test_m], log_pi_v, best_tau[0]), NUM_VERBS, "logits")
            np_test = probabilities(adjust(zn[test_m], log_pi_n, best_tau[1]), NUM_NOUNS, "logits")

            for k, v in eval_action_subsets(action_top5(vp_test, np_test), ya[test_m], unseen_m[test_m], tail_m[test_m], metric).items():
                agg["Logit Adj (tau_v, tau_n)"][k].append(v)

            best_b_sc, best_b = -1.0, 0.0
            for b in betas:
                sc = scores_for(ya[tune_m], pairwise_action_top5(vp_tune, np_tune, -b * unseen_pair))[metric]
                if sc > best_b_sc:
                    best_b_sc, best_b = sc, b
            for k, v in eval_action_subsets(pairwise_action_top5(vp_test, np_test, -best_b * unseen_pair), ya[test_m], unseen_m[test_m], tail_m[test_m], metric).items():
                agg["+ Feasibility Mask (-beta)"][k].append(v)

            best_pmi_sc, best_pmi_g = -1.0, 0.0
            for g in gammas:
                sc = scores_for(ya[tune_m], pairwise_action_top5(vp_tune, np_tune, g * pmi))[metric]
                if sc > best_pmi_sc:
                    best_pmi_sc, best_pmi_g = sc, g
            for k, v in eval_action_subsets(pairwise_action_top5(vp_test, np_test, best_pmi_g * pmi), ya[test_m], unseen_m[test_m], tail_m[test_m], metric).items():
                agg["+ Smoothed PMI (+gamma)"][k].append(v)

            best_j_sc, best_j_g = -1.0, 0.0
            for g in gammas:
                sc = scores_for(ya[tune_m], pairwise_action_top5(vp_tune, np_tune, g * log_joint))[metric]
                if sc > best_j_sc:
                    best_j_sc, best_j_g = sc, g
            for k, v in eval_action_subsets(pairwise_action_top5(vp_test, np_test, best_j_g * log_joint), ya[test_m], unseen_m[test_m], tail_m[test_m], metric).items():
                agg["+ Joint Prior (+gamma)"][k].append(v)

            chosen_params.append(f"s{seed}: tau={best_tau}, beta={best_b}, g_pmi={best_pmi_g}, g_joint={best_j_g}")

        print(f"\n=== [{model_name}] Val-Test Half Results (Mean ± SD over 3 seeds, {metric.upper()}) ===")
        print("Tuned params:", " | ".join(chosen_params))
        print("| Post-Hoc Method | All | Seen | Unseen | Tail |")
        print("|---|---|---|---|---|")
        for method_name in methods:
            res = agg[method_name]
            row = [f"{np.mean(res[s]):.2f} ± {np.std(res[s]):.2f}%" for s in ["all", "seen", "unseen", "tail"]]
            print(f"| {method_name} | {' | '.join(row)} |")


if __name__ == "__main__":
    main()
