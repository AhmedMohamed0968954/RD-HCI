"""Member 2: TokenBank mmap reader + Factorized Spatiotemporal Attentive Probe (AR & STA)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenBank:
    """Zero-copy memory-mapped reader for consolidated V-JEPA 2.1 [N, 24, D] float16 token arrays."""

    def __init__(self, root: str | Path, verify_leakage: bool = True):
        self.root = Path(root)
        self.verify_leakage = verify_leakage
        idx = pd.read_csv(self.root / "events_index.csv")
        self.row = dict(zip(idx["narration_id"], idx["row"]))
        self.sta_clip_end = dict(zip(idx["narration_id"], idx["sta_clip_end_s"]))
        self.use_prior_map = dict(zip(idx["narration_id"], idx["sta_use_prior"].astype(bool)))

        self._ar = None
        self._sta_obs = None
        self._sta_pred = None
        self._bank = None
        self._bank_idx = None

    def ar_tokens(self, narration_id: str) -> torch.Tensor:
        if self._ar is None:
            self._ar = np.load(self.root / "ar_tokens.npy", mmap_mode="r")
        r = self.row[narration_id]
        return torch.from_numpy(np.array(self._ar[r], dtype=np.float32))

    def sta_obs_tokens(self, narration_id: str, sta_cutoff: float | None = None) -> tuple[torch.Tensor, bool]:
        if self._sta_obs is None:
            self._sta_obs = np.load(self.root / "sta_obs_tokens.npy", mmap_mode="r")
        r = self.row[narration_id]
        use_prior = self.use_prior_map[narration_id]
        if self.verify_leakage and sta_cutoff is not None and not use_prior:
            end_s = self.sta_clip_end[narration_id]
            if end_s > sta_cutoff + 1e-5:
                raise AssertionError(f"STA leakage detected for {narration_id}: {end_s} > {sta_cutoff}")
        return torch.from_numpy(np.array(self._sta_obs[r], dtype=np.float32)), use_prior

    def sta_pred_tokens(self, narration_id: str, sta_cutoff: float | None = None) -> tuple[torch.Tensor, bool]:
        if self._sta_pred is None:
            self._sta_pred = np.load(self.root / "sta_pred_tokens.npy", mmap_mode="r")
        r = self.row[narration_id]
        use_prior = self.use_prior_map[narration_id]
        if self.verify_leakage and sta_cutoff is not None and not use_prior:
            end_s = self.sta_clip_end[narration_id]
            if end_s > sta_cutoff + 1e-5:
                raise AssertionError(f"STA leakage detected for {narration_id}: {end_s} > {sta_cutoff}")
        return torch.from_numpy(np.array(self._sta_pred[r], dtype=np.float32)), use_prior


class MultiQueryCrossAttentionPool(nn.Module):
    """Learnable multi-query attentive pooling [B, M, D] attending over [B, N_tok, D] -> [B, D]."""

    def __init__(self, embed_dim: int = 1024, num_queries: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, return_attn: bool = False):
        B = tokens.size(0)
        q = self.norm_q(self.queries.expand(B, -1, -1))
        kv = self.norm_kv(tokens)
        out, attn_weights = self.attn(q, kv, kv, need_weights=return_attn, average_attn_weights=True)
        out = out + self.proj(out)
        pooled = out.mean(dim=1)  # [B, D]
        if return_attn:
            return pooled, attn_weights
        return pooled


class FactorizedAttentiveProbe(nn.Module):
    """
    Factorized Spatiotemporal-Motion Attentive Probe for EPIC-KITCHENS-100 AR & STA.
    Input: tokens of shape [B, 24, D] where:
      - tokens[:, :8, :]  = E_temp (8 temporal tubelet-mean tokens capturing motion progression)
      - tokens[:, 8:, :]  = E_spat (16 spatial 4x4 grid tokens from final 0.5s capturing active objects)

    Modes (for Mid-Term & Final Ablation Table):
      - "factorized": Verb queries attend to E_temp (+ global context); Noun queries attend to E_spat.
      - "full_attn":  Both Verb and Noun queries attend over all 24 tokens (JFAA baseline).
      - "mean_pool":  Simple average pooling over all 24 tokens (Linear/MLP baseline).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_queries: int = 4,
        num_heads: int = 8,
        num_verbs: int = 97,
        num_nouns: int = 300,
        dropout: float = 0.2,
        mode: str = "factorized",
    ):
        super().__init__()
        if mode not in {"factorized", "full_attn", "mean_pool"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.embed_dim = embed_dim
        self.mode = mode

        # Learned positional embeddings for the 8 temporal + 16 spatial tokens
        self.temp_pos = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)
        self.spat_pos = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)

        if mode in {"factorized", "full_attn"}:
            self.verb_pool = MultiQueryCrossAttentionPool(embed_dim, num_queries, num_heads, dropout)
            self.noun_pool = MultiQueryCrossAttentionPool(embed_dim, num_queries, num_heads, dropout)

        # Cross-Interaction MLP so verbs know which object is held and vice versa
        self.verb_interact = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.noun_interact = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_n = nn.LayerNorm(embed_dim)
        self.verb_head = nn.Linear(embed_dim, num_verbs)
        self.noun_head = nn.Linear(embed_dim, num_nouns)

    def encode_tokens(self, tokens: torch.Tensor, return_attn: bool = False):
        """
        Extract disentangled (e_v, e_n) embeddings in R^{B x D}.
        Used directly by AR classifier, Member 3 (Decoupled AGA), and Member 4 (3-Way Gate).
        """
        if tokens.ndim != 3 or tokens.size(1) != 24 or tokens.size(2) != self.embed_dim:
            raise ValueError(f"Expected [B, 24, {self.embed_dim}] tokens, got {tuple(tokens.shape)}")

        e_temp = tokens[:, :8, :] + self.temp_pos
        e_spat = tokens[:, 8:, :] + self.spat_pos
        z_all = torch.cat([e_temp, e_spat], dim=1)

        attn_dict = {}
        if self.mode == "mean_pool":
            ev_raw = z_all.mean(dim=1)
            en_raw = z_all.mean(dim=1)
        elif self.mode == "full_attn":
            if return_attn:
                ev_raw, attn_dict["verb_attn"] = self.verb_pool(z_all, return_attn=True)
                en_raw, attn_dict["noun_attn"] = self.noun_pool(z_all, return_attn=True)
            else:
                ev_raw = self.verb_pool(z_all)
                en_raw = self.noun_pool(z_all)
        else:  # "factorized"
            # Verb queries attend to the 8 temporal tubelets + 1 global spatial summary token
            spat_summary = e_spat.mean(dim=1, keepdim=True)
            verb_kv = torch.cat([e_temp, spat_summary], dim=1)  # [B, 9, D]
            # Noun queries attend to the 16 spatial grid tokens + 1 global temporal summary token
            temp_summary = e_temp.mean(dim=1, keepdim=True)
            noun_kv = torch.cat([e_spat, temp_summary], dim=1)  # [B, 17, D]
            if return_attn:
                ev_raw, attn_dict["verb_attn"] = self.verb_pool(verb_kv, return_attn=True)
                en_raw, attn_dict["noun_attn"] = self.noun_pool(noun_kv, return_attn=True)
            else:
                ev_raw = self.verb_pool(verb_kv)
                en_raw = self.noun_pool(noun_kv)

        # Bidirectional Verb <-> Noun cross-interaction residual
        joint = torch.cat([ev_raw, en_raw], dim=-1)
        e_v = self.norm_v(ev_raw + self.verb_interact(joint))
        e_n = self.norm_n(en_raw + self.noun_interact(joint))

        if return_attn:
            return e_v, e_n, attn_dict
        return e_v, e_n

    def forward(self, tokens: torch.Tensor, return_embeddings: bool = False, return_attn: bool = False):
        if return_attn:
            e_v, e_n, attn_dict = self.encode_tokens(tokens, return_attn=True)
        else:
            e_v, e_n = self.encode_tokens(tokens, return_attn=False)

        verb_logits = self.verb_head(e_v)
        noun_logits = self.noun_head(e_n)
        out = {"verb_output": verb_logits, "noun_output": noun_logits}
        if return_embeddings:
            out["e_v"] = e_v
            out["e_n"] = e_n
        if return_attn:
            out.update(attn_dict)
        return out
