"""Member 4: the future branch of the three-way gate, from cached V-JEPA 2.1 predictor tokens.

Member 2's extractor stores, per STA event, 24 predictor tokens [24, D] in sta_pred_tokens.npy
(scripts/extract_vjepa21_tokens.py, pool_predicted_24_tokens). Their layout:
    tokens 0..6   the last 7 OBSERVED temporal means (student encoder space, copied from the
                  observed clip, so they repeat what the present branch already sees)
    token  7      the PREDICTED temporal mean, 1 s after the cutoff
    tokens 8..23  the PREDICTED 4x4 spatial grid, 1 s after the cutoff
The predictor outputs the ViT-G teacher's space (sliced to 1024 channels), not the student's.
sta_pred17_1664.npy (scripts/extract_pred1664_and_test.py) holds only the predicted tokens in the
full teacher width: [17, 1664] = 1 predicted temporal mean + the predicted 4x4 grid.

FutureBranch turns these tokens into one vector per field, f_v and f_n [B, d], the shape
GatedAGAModel(future=...) expects (Member 3, epic_starter/models/aga.py):
    encoder="mean"   mean of the predictor-only tokens (7..23 of the 24-token file, all 17 of the
                     17-token file) -> LayerNorm + Linear per field. Uses nothing but the
                     predictor's output, so the gate ablation measures what the predictor adds.
    encoder="probe"  the predictor pathway of one of Member 2's STA probes, loaded from its
                     checkpoint, then LayerNorm + Linear per field (probe_mode, PROBE_MODES):
        factorized_pred       24 tokens [24, 1024]: pred pools + ln_pred_* on all 24 tokens
        pred_17_only          17 tokens [17, 1664]: teacher projection + pred pools + norm_*
                              (strict future: nothing observed)
        factorized_pred_1664  17 tokens [17, 1664] + 7 observed temporal tokens of the present
                              clip (obs_ctx_adapter), as Member 2's probe does: needs the present
                              clip's tokens (needs_obs; GatedAGAModel passes them)
Rows marked unavailable return zeros (the gate also gives them weight 0).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PREDICTED_TOKENS = (7,) + tuple(range(8, 24))      # token indices written by the predictor
ENCODERS = ("mean", "probe")
# Member 2's probe mode -> (tokens, width of its predictor tokens; None = the probe's embed_dim)
PROBE_MODES = {"factorized_pred": (24, None), "pred_17_only": (17, 1664), "factorized_pred_1664": (17, 1664)}
# the predictor file each probe mode was trained on
PROBE_TOKENS_FILE = {"factorized_pred": "sta_pred_tokens.npy", "pred_17_only": "sta_pred17_1664.npy",
                     "factorized_pred_1664": "sta_pred17_1664.npy"}


def probe_mode_of(path) -> str:
    """The 'mode' stored in a train_probe.py checkpoint (best.pt)."""
    return torch.load(path, map_location="cpu", weights_only=False).get("mode")


class FutureBranch(nn.Module):
    """feat_dim / n_tokens: width and count of the predictor tokens per event (24 x 1024 for
    sta_pred_tokens.npy, 17 x 1664 for sta_pred17_1664.npy). With encoder='mean' a 24-token file is
    read with the layout above (tokens 7..23); any other count is read as predictor-only tokens.
    With encoder='probe', probe_mode picks Member 2's pathway and embed_dim is the probe's width
    (that of the observed tokens, 1024; default feat_dim for factorized_pred, else 1024)."""

    def __init__(self, feat_dim: int, d: int, encoder: str = "mean", freeze_probe: bool = False,
                 n_tokens: int = 24, probe_mode: str = "factorized_pred", embed_dim: int | None = None):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
        self.encoder, self.feat_dim, self.freeze_probe, self.n_tokens = encoder, feat_dim, freeze_probe, n_tokens
        self.probe_mode = probe_mode if encoder == "probe" else None
        out_dim = feat_dim
        if encoder == "probe":
            if probe_mode not in PROBE_MODES:
                raise ValueError(f"probe_mode must be one of {tuple(PROBE_MODES)}, got {probe_mode!r}")
            embed_dim = embed_dim or (feat_dim if probe_mode == "factorized_pred" else 1024)
            want_n, want_d = PROBE_MODES[probe_mode]
            want_d = want_d or embed_dim
            if (n_tokens, feat_dim) != (want_n, want_d):
                raise ValueError(f"Member 2's '{probe_mode}' probe reads [{want_n}, {want_d}] predictor tokens "
                                 f"({PROBE_TOKENS_FILE[probe_mode]}), got [{n_tokens}, {feat_dim}]")
            from epic_starter.models.factorized_probe import FactorizedAttentiveProbe   # Member 2
            self.probe = FactorizedAttentiveProbe(embed_dim=embed_dim, mode=probe_mode)
            if freeze_probe:
                self.probe.requires_grad_(False)
            out_dim = embed_dim
        self.embed_dim = out_dim
        self.needs_obs = self.probe_mode == "factorized_pred_1664"
        index = PREDICTED_TOKENS if n_tokens == 24 else tuple(range(n_tokens))
        self.register_buffer("index", torch.tensor(index), persistent=False)
        # same projection as the present branch (AGAMemoryModel.present_v / present_n)
        self.proj_v = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, d))
        self.proj_n = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, d))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder == "probe" and self.freeze_probe:
            self.probe.eval()                                   # frozen probe: no dropout
        return self

    def load_probe(self, path, map_location="cpu"):
        """Initialise from Member 2's train_probe.py checkpoint (--task STA, --mode = probe_mode).
        Returns (epoch, val action MT5R) stored in the checkpoint."""
        if self.encoder != "probe":
            raise ValueError("load_probe needs encoder='probe'")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if ckpt.get("mode") != self.probe_mode or ckpt.get("task") not in (None, "STA"):
            raise ValueError(f"Expected an STA '{self.probe_mode}' probe checkpoint, got task={ckpt.get('task')} "
                             f"mode={ckpt.get('mode')}")
        self.probe.load_state_dict(ckpt["model"])               # strict: shapes and keys must match
        return ckpt.get("epoch"), ckpt.get("metrics", {}).get("all", {}).get("action_mt5r")

    def pooled(self, tokens: torch.Tensor, obs_tokens: torch.Tensor | None = None):
        """[B, n_tokens, feat_dim] predictor tokens (+ the present clip's [B, 24, embed_dim] tokens for
        factorized_pred_1664) -> (verb, noun) features [B, embed_dim] before the projection."""
        if tokens.ndim != 3 or tokens.shape[1] != self.n_tokens or tokens.shape[2] != self.feat_dim:
            raise ValueError(f"Expected [B, {self.n_tokens}, {self.feat_dim}] predictor tokens, "
                             f"got {tuple(tokens.shape)}")
        tokens = tokens.float()
        if self.encoder == "mean":
            x = tokens.index_select(1, self.index).mean(dim=1)
            return x, x
        # exactly the predictor pathways of FactorizedAttentiveProbe.encode_tokens
        p = self.probe
        if self.probe_mode == "pred_17_only":
            p17 = p.pred_teacher_proj(tokens)
            x = torch.cat([p17[:, :1] + p.pred17_temp_pos, p17[:, 1:] + p.pred17_spat_pos], dim=1)
            return (p.norm_v(p.pred_verb_pool(x, attn_bias=p.pred17_verb_bias)),
                    p.norm_n(p.pred_noun_pool(x, attn_bias=p.pred17_noun_bias)))
        if self.probe_mode == "factorized_pred_1664":
            if obs_tokens is None or obs_tokens.shape[1:] != (24, self.embed_dim):
                raise ValueError(f"factorized_pred_1664 also needs the present clip's [B, 24, {self.embed_dim}] "
                                 "tokens (obs_tokens)")
            p17 = p.pred_teacher_proj(tokens)
            temp = torch.cat([p.obs_ctx_adapter(obs_tokens.float()[:, 1:8]), p17[:, :1]], dim=1) + p.temp_pos
            x = torch.cat([temp, p17[:, 1:] + p.spat_pos], dim=1)
        else:                                                   # factorized_pred
            x = torch.cat([tokens[:, :8] + p.temp_pos, tokens[:, 8:] + p.spat_pos], dim=1)
        fv = p.ln_pred_v(p.pred_verb_pool(x, attn_bias=p.verb_attn_bias))
        fn = p.ln_pred_n(p.pred_noun_pool(x, attn_bias=p.noun_attn_bias))
        return fv, fn

    def forward(self, tokens: torch.Tensor, available: torch.Tensor | None = None,
                obs_tokens: torch.Tensor | None = None):
        """tokens [B, n_tokens, feat_dim], available [B] bool (None = all), obs_tokens [B, 24, embed_dim]
        (factorized_pred_1664 only) -> (f_v, f_n), each [B, d]."""
        if available is not None:
            keep = available.bool().view(-1, 1, 1)
            tokens = torch.where(keep, tokens, torch.zeros_like(tokens))   # NaN-safe for missing rows
            if obs_tokens is not None:
                obs_tokens = torch.where(keep, obs_tokens, torch.zeros_like(obs_tokens))
        xv, xn = self.pooled(tokens, obs_tokens)
        fv, fn = self.proj_v(xv), self.proj_n(xn)
        if available is not None:
            keep = available.bool().view(-1, 1)
            fv, fn = fv * keep, fn * keep
        return fv, fn
