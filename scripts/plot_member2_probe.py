"""
Generate Member 2 Mid-Term Presentation Figures:
  1. Verb vs. Noun Attention Mass across the 8 Temporal Tubelets and 4x4 Spatial Grid
  2. Learned 4x4 Spatial Grid Heatmap for Noun Queries vs. Verb Queries
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epic_starter.data.manifests import read_manifest
from epic_starter.models.factorized_probe import TokenBank, FactorizedAttentiveProbe


def main():
    epic_work = Path("/lustre1/scratch/394/vsc39484/epic-kitchens")
    feat_dir = epic_work / "team/features/vjepa21_vitl384_f16"
    manifest_dir = epic_work / "team/manifests/split_seed42"
    res_root = epic_work / "team/results/member2_probe_full"
    out_dir = res_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = TokenBank(feat_dir, verify_leakage=True)
    val_rows = read_manifest(manifest_dir / "val_ar.jsonl")[:512]
    tokens = torch.stack([bank.ar_tokens(r["narration_id"]) for r in val_rows])

    ckpt = torch.load(res_root / "AR_factorized/best.pt", map_location="cpu", weights_only=False)
    model = FactorizedAttentiveProbe(mode="factorized")
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        _, _, attn = model.encode_tokens(tokens, return_attn=True)
        # verb_attn, noun_attn: [B, num_queries=4, 24]
        v_attn = attn["verb_attn"].mean(dim=(0, 1)).numpy()  # [24]
        n_attn = attn["noun_attn"].mean(dim=(0, 1)).numpy()  # [24]

    stats = {
        "verb_temporal_mass_pct": float(v_attn[:8].sum() * 100),
        "verb_spatial_mass_pct": float(v_attn[8:].sum() * 100),
        "noun_temporal_mass_pct": float(n_attn[:8].sum() * 100),
        "noun_spatial_mass_pct": float(n_attn[8:].sum() * 100),
        "verb_temporal_8": [round(float(x), 4) for x in v_attn[:8]],
        "noun_spatial_4x4": np.round(n_attn[8:].reshape(4, 4), 4).tolist(),
    }
    (out_dir / "attention_routing_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("=== LEARNED ATTENTION ROUTING STATS (AR Factorized Probe) ===")
    print(f"Verb Queries -> Temporal Tubelets (T1..T8): {stats['verb_temporal_mass_pct']:.1f}% | Spatial Grid: {stats['verb_spatial_mass_pct']:.1f}%")
    print(f"Noun Queries -> Temporal Tubelets (T1..T8): {stats['noun_temporal_mass_pct']:.1f}% | Spatial Grid: {stats['noun_spatial_mass_pct']:.1f}%")
    print("Noun 4x4 Spatial Grid Attention (rows=top->bottom, cols=left->right):")
    print(np.array(stats["noun_spatial_4x4"]))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

        # Panel 1: 8 Temporal Tubelets (Verb vs Noun)
        x = np.arange(1, 9)
        axes[0].plot(x, v_attn[:8] * 100, marker="o", linewidth=2.2, label="Verb Queries (Motion)")
        axes[0].plot(x, n_attn[:8] * 100, marker="s", linewidth=2.2, linestyle="--", label="Noun Queries (Object)")
        axes[0].set_title("Temporal Tubelet Attention (T1 -> T8)")
        axes[0].set_xlabel("Temporal Tubelet Slice (2 frames / slice)")
        axes[0].set_ylabel("Mean Attention Weight (%)")
        axes[0].set_xticks(x)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Panel 2: 4x4 Spatial Grid Heatmap for Noun Queries
        grid = n_attn[8:].reshape(4, 4) * 100
        im = axes[1].imshow(grid, cmap="YlOrRd")
        axes[1].set_title("Noun Query 4x4 Active Spatial Grid (%)")
        axes[1].set_xticks(range(4), ["Left", "Mid-L", "Mid-R", "Right"])
        axes[1].set_yticks(range(4), ["Top", "Upper-M", "Lower-M", "Bottom (Hands)"])
        for i in range(4):
            for j in range(4):
                axes[1].text(j, i, f"{grid[i, j]:.1f}%", ha="center", va="center", color="black", fontweight="bold")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(out_dir / "member2_attention_routing.pdf")
        plt.savefig(out_dir / "member2_attention_routing.png")
        print(f"Saved figure to {out_dir / 'member2_attention_routing.png'}")
    except ImportError:
        print("matplotlib not installed in current shell; JSON stats saved.")


if __name__ == "__main__":
    main()
