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
        self._sta_pred17 = None

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

    def sta_pred17_tokens(self, narration_id: str, sta_cutoff: float | None = None) -> tuple[torch.Tensor, bool]:
        if self._sta_pred17 is None:
            self._sta_pred17 = np.load(self.root / "sta_pred17_1664.npy", mmap_mode="r")
        r = self.row[narration_id]
        use_prior = self.use_prior_map[narration_id]
        if self.verify_leakage and sta_cutoff is not None and not use_prior:
            end_s = self.sta_clip_end[narration_id]
            if end_s > sta_cutoff + 1e-5:
                raise AssertionError(f"STA leakage detected for {narration_id}: {end_s} > {sta_cutoff}")
        return torch.from_numpy(np.array(self._sta_pred17[r], dtype=np.float32)), use_prior


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

    def forward(self, tokens: torch.Tensor, attn_bias: torch.Tensor | None = None, return_attn: bool = False):
        B = tokens.size(0)
        q = self.norm_q(self.queries.expand(B, -1, -1))
        kv = self.norm_kv(tokens)
        out, attn_weights = self.attn(
            q, kv, kv, attn_mask=attn_bias, need_weights=return_attn, average_attn_weights=True
        )
        out = out + self.proj(out)
        pooled = out.mean(dim=1)
        if return_attn:
            return pooled, attn_weights
        return pooled


class FactorizedAttentiveProbe(nn.Module):
    """
    Factorized Spatiotemporal-Motion Attentive Probe for EPIC-KITCHENS-100 AR & STA.
    Modes:
      - "mean_pool":   Global average pooling over all 24 tokens (Baseline 1).
      - "full_attn":   Single-stream attentive probe over all 24 tokens without stream specialization (Baseline 2).
      - "factorized":  Disentangled Verb (temporal-prioritized) & Noun (spatial-prioritized) multi-query routing
                       with learnable soft stream bias + bidirectional Verb<->Noun cross-interaction MLP.
      - "factorized_pred": (STA) Fuses observed tokens (e_t) + V-JEPA 2.1 predictor future tokens (f_t) via LayerNorm gate.
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
        valid_modes = {"factorized", "factorized_pred", "factorized_pred_1664", "pred_17_only", "full_attn", "mean_pool", "split_mean_mlp"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode: {mode}")
        self.embed_dim = embed_dim
        self.mode = mode
        self.num_queries = num_queries

        self.temp_pos = nn.Parameter(torch.randn(1, 8, embed_dim) * 0.02)
        self.spat_pos = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)

        if mode == "split_mean_mlp":
            # Dynamically match trainable parameter count of "factorized" within < 1%
            ref = FactorizedAttentiveProbe(
                embed_dim=embed_dim, num_queries=num_queries, num_heads=num_heads,
                num_verbs=num_verbs, num_nouns=num_nouns, dropout=dropout, mode="factorized",
            )
            target_params = sum(p.numel() for p in ref.parameters() if p.requires_grad)
            # Non-MLP params: temp_pos(8d) + spat_pos(16d) + 4*LN(d) + heads((d+1)*(V+N)) + out_bias(2d)
            non_mlp = (8 + 16) * embed_dim + 8 * embed_dim + (embed_dim + 1) * (num_verbs + num_nouns) + 2 * embed_dim
            per_hidden = 4 * embed_dim + 3  # Linear(2d, h) + LN(h) + Linear(h, 2d)
            d_hidden = max(64, round((target_params - non_mlp) / per_hidden))
            self.ln_temp_split = nn.LayerNorm(embed_dim)
            self.ln_spat_split = nn.LayerNorm(embed_dim)
            self.split_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, d_hidden),
                nn.GELU(),
                nn.LayerNorm(d_hidden),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        if mode in {"factorized", "factorized_pred", "factorized_pred_1664", "full_attn"}:
            self.verb_pool = MultiQueryCrossAttentionPool(embed_dim, num_queries, num_heads, dropout)
            self.noun_pool = MultiQueryCrossAttentionPool(embed_dim, num_queries, num_heads, dropout)

        if mode in {"factorized", "factorized_pred", "factorized_pred_1664"}:
            # Separate token-type embeddings + learnable attention logit biases for Verb (temporal) vs Noun (spatial)
            self.verb_type_emb = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)
            self.noun_type_emb = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)
            # Initialized so Verb favors E_temp (+1.0) but can still read active hand-object layout in E_spat (0.0),
            # and Noun favors E_spat (+1.0) while retaining temporal context (0.0)
            v_bias = torch.zeros(num_queries, 24)
            v_bias[:, :8] = 1.0
            n_bias = torch.zeros(num_queries, 24)
            n_bias[:, 8:] = 1.0
            self.verb_attn_bias = nn.Parameter(v_bias)
            self.noun_attn_bias = nn.Parameter(n_bias)

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

        if mode in {"factorized_pred", "factorized_pred_1664", "pred_17_only"}:
            if mode in {"factorized_pred_1664", "pred_17_only"}:
                self.pred_teacher_proj = nn.Sequential(
                    nn.LayerNorm(1664),
                    nn.Linear(1664, embed_dim),
                )
            if mode == "factorized_pred_1664":
                self.obs_ctx_adapter = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim),
                )
            if mode == "pred_17_only":
                self.pred17_temp_pos = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
                self.pred17_spat_pos = nn.Parameter(torch.randn(1, 16, embed_dim) * 0.02)
                v17_bias = torch.zeros(num_queries, 17)
                v17_bias[:, :1] = 1.0
                n17_bias = torch.zeros(num_queries, 17)
                n17_bias[:, 1:] = 1.0
                self.pred17_verb_bias = nn.Parameter(v17_bias)
                self.pred17_noun_bias = nn.Parameter(n17_bias)
            # Separate projection + LayerNorm gate for teacher-space predictor tokens (1664->1024 sliced)
            self.pred_verb_pool = MultiQueryCrossAttentionPool(embed_dim, num_queries, num_heads, dropout)
            self.pred_noun_pool = MultiQueryCrossAttentionPool(embed_dim, num_queries, num_heads, dropout)
            if mode in {"factorized_pred", "factorized_pred_1664"}:
                self.ln_obs_v = nn.LayerNorm(embed_dim)
                self.ln_pred_v = nn.LayerNorm(embed_dim)
                self.ln_obs_n = nn.LayerNorm(embed_dim)
                self.ln_pred_n = nn.LayerNorm(embed_dim)
                self.gate_v = nn.Linear(embed_dim * 2, 2)
                self.gate_n = nn.Linear(embed_dim * 2, 2)

        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_n = nn.LayerNorm(embed_dim)
        self.verb_head = nn.Linear(embed_dim, num_verbs)
        self.noun_head = nn.Linear(embed_dim, num_nouns)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        pred_tokens: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        if tokens.ndim != 3 or tokens.size(1) != 24 or tokens.size(2) != self.embed_dim:
            raise ValueError(f"Expected [B, 24, {self.embed_dim}] tokens, got {tuple(tokens.shape)}")

        e_temp = tokens[:, :8, :] + self.temp_pos
        e_spat = tokens[:, 8:, :] + self.spat_pos
        z_all = torch.cat([e_temp, e_spat], dim=1)

        attn_dict = {}
        if self.mode == "mean_pool":
            ev_raw = z_all.mean(dim=1)
            en_raw = z_all.mean(dim=1)
            e_v = self.norm_v(ev_raw)
            e_n = self.norm_n(en_raw)
        elif self.mode == "split_mean_mlp":
            z_temp = self.ln_temp_split(e_temp.mean(dim=1))
            z_spat = self.ln_spat_split(e_spat.mean(dim=1))
            z_cat = torch.cat([z_temp, z_spat], dim=-1)
            ev_raw, en_raw = (z_cat + self.split_mlp(z_cat)).chunk(2, dim=-1)
            e_v = self.norm_v(ev_raw)
            e_n = self.norm_n(en_raw)
        elif self.mode == "pred_17_only":
            if pred_tokens is None:
                raise ValueError("pred_17_only mode requires pred_tokens [B, 17, 1664]")
            p17 = self.pred_teacher_proj(pred_tokens)
            p_all = torch.cat(
                [p17[:, :1, :] + self.pred17_temp_pos, p17[:, 1:, :] + self.pred17_spat_pos], dim=1
            )
            fv_raw = self.pred_verb_pool(p_all, attn_bias=self.pred17_verb_bias)
            fn_raw = self.pred_noun_pool(p_all, attn_bias=self.pred17_noun_bias)
            e_v = self.norm_v(fv_raw)
            e_n = self.norm_n(fn_raw)
        elif self.mode == "full_attn":
            if return_attn:
                ev_raw, attn_dict["verb_attn"] = self.verb_pool(z_all, return_attn=True)
                en_raw, attn_dict["noun_attn"] = self.noun_pool(z_all, return_attn=True)
            else:
                ev_raw = self.verb_pool(z_all)
                en_raw = self.noun_pool(z_all)
            e_v = self.norm_v(ev_raw)
            e_n = self.norm_n(en_raw)
        else:  # "factorized" or "factorized_pred"
            verb_kv = torch.cat(
                [e_temp + self.verb_type_emb[:, 0:1, :], e_spat + self.verb_type_emb[:, 1:2, :]], dim=1
            )
            noun_kv = torch.cat(
                [e_temp + self.noun_type_emb[:, 0:1, :], e_spat + self.noun_type_emb[:, 1:2, :]], dim=1
            )
            if return_attn:
                ev_raw, attn_dict["verb_attn"] = self.verb_pool(
                    verb_kv, attn_bias=self.verb_attn_bias, return_attn=True
                )
                en_raw, attn_dict["noun_attn"] = self.noun_pool(
                    noun_kv, attn_bias=self.noun_attn_bias, return_attn=True
                )
            else:
                ev_raw = self.verb_pool(verb_kv, attn_bias=self.verb_attn_bias)
                en_raw = self.noun_pool(noun_kv, attn_bias=self.noun_attn_bias)

            joint = torch.cat([ev_raw, en_raw], dim=-1)
            e_v = self.norm_v(ev_raw + self.verb_interact(joint))
            e_n = self.norm_n(en_raw + self.noun_interact(joint))

            if self.mode in {"factorized_pred", "factorized_pred_1664"} and pred_tokens is not None:
                if self.mode == "factorized_pred_1664":
                    # Per-source adapters: 7 observed temporal tokens (1024) + 17 teacher tokens (1664 -> 1024)
                    obs_ctx_7 = self.obs_ctx_adapter(tokens[:, 1:8, :])
                    pred_17 = self.pred_teacher_proj(pred_tokens)
                    p_temp = torch.cat([obs_ctx_7, pred_17[:, :1, :]], dim=1) + self.temp_pos
                    p_spat = pred_17[:, 1:, :] + self.spat_pos
                else:
                    p_temp = pred_tokens[:, :8, :] + self.temp_pos
                    p_spat = pred_tokens[:, 8:, :] + self.spat_pos
                p_all = torch.cat([p_temp, p_spat], dim=1)
                fv_raw = self.pred_verb_pool(p_all, attn_bias=self.verb_attn_bias)
                fn_raw = self.pred_noun_pool(p_all, attn_bias=self.noun_attn_bias)

                ev_ln, fv_ln = self.ln_obs_v(e_v), self.ln_pred_v(fv_raw)
                en_ln, fn_ln = self.ln_obs_n(e_n), self.ln_pred_n(fn_raw)
                wv = F.softmax(self.gate_v(torch.cat([ev_ln, fv_ln], dim=-1)), dim=-1)
                wn = F.softmax(self.gate_n(torch.cat([en_ln, fn_ln], dim=-1)), dim=-1)
                e_v = wv[:, 0:1] * ev_ln + wv[:, 1:2] * fv_ln
                e_n = wn[:, 0:1] * en_ln + wn[:, 1:2] * fn_ln

        if return_attn:
            return e_v, e_n, attn_dict
        return e_v, e_n

    def forward(
        self,
        tokens: torch.Tensor,
        pred_tokens: torch.Tensor | None = None,
        return_embeddings: bool = False,
        return_attn: bool = False,
    ):
        if return_attn:
            e_v, e_n, attn_dict = self.encode_tokens(tokens, pred_tokens=pred_tokens, return_attn=True)
        else:
            e_v, e_n = self.encode_tokens(tokens, pred_tokens=pred_tokens, return_attn=False)

        verb_logits = self.verb_head(e_v)
        noun_logits = self.noun_head(e_n)
        out = {"verb_output": verb_logits, "noun_output": noun_logits}
        if return_embeddings:
            out["e_v"] = e_v
            out["e_n"] = e_n
        if return_attn:
            out.update(attn_dict)
        return out
