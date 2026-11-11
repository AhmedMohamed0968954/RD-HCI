"""Member 4: the future branch of the three-way gate, from cached V-JEPA 2.1 predictor tokens.

Member 2's extractor stores, per STA event, 24 predictor tokens [24, D] in sta_pred_tokens.npy
(scripts/extract_vjepa21_tokens.py, pool_predicted_24_tokens). Their layout:
    tokens 0..6   the last 7 OBSERVED temporal means (student encoder space, copied from the
                  observed clip, so they repeat what the present branch already sees)
    token  7      the PREDICTED temporal mean, 1 s after the cutoff
    tokens 8..23  the PREDICTED 4x4 spatial grid, 1 s after the cutoff
The predictor outputs the ViT-G teacher's space (sliced to 1024 channels), not the student's.

FutureBranch turns these tokens into one vector per field, f_v and f_n [B, d], the shape
GatedAGAModel(future=...) expects (Member 3, epic_starter/models/aga.py):
    encoder="mean"   mean of the 17 predictor-only tokens -> LayerNorm + Linear per field.
                     Uses nothing but the predictor's output, so the gate ablation measures
                     what the predictor adds on top of the observed clip.
    encoder="probe"  the predictor pathway of Member 2's STA 'factorized_pred' probe
                     (pred_verb_pool / pred_noun_pool + their LayerNorms, trained on all 24
                     tokens), loaded from its checkpoint, then LayerNorm + Linear per field.
Rows marked unavailable return zeros (the gate also gives them weight 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PREDICTED_TOKENS = (7,) + tuple(range(8, 24))      # token indices written by the predictor
ENCODERS = ("mean", "probe")


class FutureBranch(nn.Module):
    def __init__(self, feat_dim: int, d: int, encoder: str = "mean", freeze_probe: bool = False):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
        self.encoder, self.feat_dim, self.freeze_probe = encoder, feat_dim, freeze_probe
        self.register_buffer("index", torch.tensor(PREDICTED_TOKENS), persistent=False)
        if encoder == "probe":
            from epic_starter.models.factorized_probe import FactorizedAttentiveProbe   # Member 2
            self.probe = FactorizedAttentiveProbe(embed_dim=feat_dim, mode="factorized_pred")
            if freeze_probe:
                self.probe.requires_grad_(False)
        # same projection as the present branch (AGAMemoryModel.present_v / present_n)
        self.proj_v = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
        self.proj_n = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder == "probe" and self.freeze_probe:
            self.probe.eval()                                   # frozen probe: no dropout
        return self

    def load_probe(self, path, map_location="cpu"):
        """Initialise from Member 2's train_probe.py checkpoint for --task STA --mode factorized_pred.
        Returns (epoch, val action MT5R) stored in the checkpoint."""
        if self.encoder != "probe":
            raise ValueError("load_probe needs encoder='probe'")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if ckpt.get("mode") != "factorized_pred" or ckpt.get("task") not in (None, "STA"):
            raise ValueError(f"Expected an STA 'factorized_pred' probe checkpoint, got task={ckpt.get('task')} "
                             f"mode={ckpt.get('mode')}")
        self.probe.load_state_dict(ckpt["model"])               # strict: shapes and keys must match
        return ckpt.get("epoch"), ckpt.get("metrics", {}).get("all", {}).get("action_mt5r")

    def pooled(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 24, D] predictor tokens -> (verb, noun) features [B, D] before the projection."""
        if tokens.ndim != 3 or tokens.shape[1] != 24 or tokens.shape[2] != self.feat_dim:
            raise ValueError(f"Expected [B, 24, {self.feat_dim}] predictor tokens, got {tuple(tokens.shape)}")
        tokens = tokens.float()
        if self.encoder == "mean":
            x = tokens.index_select(1, self.index).mean(dim=1)
            return x, x
        # exactly the predictor pathway of FactorizedAttentiveProbe.encode_tokens (mode factorized_pred)
        p = self.probe
        x = torch.cat([tokens[:, :8] + p.temp_pos, tokens[:, 8:] + p.spat_pos], dim=1)
        fv = p.ln_pred_v(p.pred_verb_pool(x, attn_bias=p.verb_attn_bias))
        fn = p.ln_pred_n(p.pred_noun_pool(x, attn_bias=p.noun_attn_bias))
        return fv, fn

    def forward(self, tokens: torch.Tensor, available: torch.Tensor | None = None):
        """tokens [B, 24, D], available [B] bool (None = all) -> (f_v, f_n), each [B, d]."""
        if available is not None:
            keep = available.bool().view(-1, 1, 1)
            tokens = torch.where(keep, tokens, torch.zeros_like(tokens))   # NaN-safe for missing rows
        xv, xn = self.pooled(tokens)
        fv, fn = self.proj_v(xv), self.proj_n(xn)
        if available is not None:
            keep = available.bool().view(-1, 1)
            fv, fn = fv * keep, fn * keep
        return fv, fn
