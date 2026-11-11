"""Action-Guided Attention memory for STA (Tai et al., 2026, arXiv:2603.01743), adapted.

Input: per-step features [B,S,T,D] from HistoryDataset plus a validity mask [B,S].
Per step, an auxiliary head predicts verb/noun probabilities p_k (detached; the model's
own predictions are used in training and testing, never labels).
Query = EMA of p over valid steps (alpha), keys = p of earlier steps, values = their features.
mode="single": one memory over concatenated [p_verb, p_noun] (97+300 = 397 dims).
mode="decoupled": a verb memory and a noun memory with their own keys/values.
Per field (verb / noun) a sigmoid gate mixes the retrieved history h with the present evidence e
(AGA's 2-way gate). The present is either the memory's last step (token mean) or Member 2's
factorized probe on the full 24 tokens of the present clip (present_encoder="probe").
`encode()` exposes h and e per field so GatedAGAModel can use Member 4's ThreeWayGate.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

N_VERB, N_NOUN = 97, 300


def masked_attention(q, k, v, mask, heads):
    """q [B,d], k/v [B,L,d], mask [B,L] True = usable. Rows with no usable key return zeros."""
    B, L, d = k.shape
    dh = d // heads
    q = q.view(B, heads, 1, dh)
    k = k.view(B, L, heads, dh).transpose(1, 2)
    v = v.view(B, L, heads, dh).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / dh ** 0.5            # [B,H,1,L]
    scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    any_key = mask.any(dim=1)
    weights = torch.softmax(scores.masked_fill(~any_key[:, None, None, None], 0.0), dim=-1)
    weights = weights * any_key[:, None, None, None]
    return (weights @ v).reshape(B, d), weights.squeeze(2)


class Gate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, h, e):
        g = torch.sigmoid(self.mlp(torch.cat([h, e], dim=-1)))
        return g * h + (1 - g) * e, g


class Memory(nn.Module):
    """One AGA stream: EMA query over predictions, attention over past steps."""

    def __init__(self, p_dim, d, heads, alpha):
        super().__init__()
        self.alpha, self.heads = alpha, heads
        self.eq = nn.Sequential(nn.LayerNorm(p_dim), nn.Linear(p_dim, d))
        self.ek = nn.Sequential(nn.LayerNorm(p_dim), nn.Linear(p_dim, d))
        self.ev = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, p, e, valid):
        """p [B,S,P] detached probabilities, e [B,S,d], valid [B,S]. Returns history at last step."""
        q = torch.zeros_like(p[:, 0])
        for s in range(p.shape[1]):                               # EMA only over valid steps
            upd = self.alpha * p[:, s] + (1 - self.alpha) * q
            q = torch.where(valid[:, s:s + 1], upd, q)
        past = valid[:, :-1]                                      # keys/values: steps before the last
        h, w = masked_attention(self.eq(q), self.ek(p[:, :-1]), self.ev(e[:, :-1]), past, self.heads)
        return h + self.ffn(h), w


class AGAMemoryModel(nn.Module):
    """AGA memory with a present-clip encoder.

    present_encoder="mean"   the present is the memory's own last-step embedding (token mean).
    present_encoder="probe"  the present clip's 24 tokens go through Member 2's
                             FactorizedAttentiveProbe (separate verb / noun embeddings). Its own
                             verb/noun logits are the base prediction and the memory adds a
                             correction (zero-initialised), so a probe initialised from Member 2's
                             checkpoint starts exactly at the probe's score.
    """

    def __init__(self, feat_dim, d=256, heads=4, alpha=0.8, mode="decoupled", dropout=0.3,
                 present_encoder="mean", freeze_present=False):
        super().__init__()
        if mode not in {"single", "decoupled"}:
            raise ValueError("mode must be single or decoupled")
        if present_encoder not in {"mean", "probe"}:
            raise ValueError("present_encoder must be mean or probe")
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.mode, self.present_encoder, self.freeze_present = mode, present_encoder, freeze_present
        self.pool = lambda x: x.mean(dim=2)                       # [B,S,T,D] -> [B,S,D]
        self.fx = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, d))
        self.aux_verb, self.aux_noun = nn.Linear(d, N_VERB), nn.Linear(d, N_NOUN)
        if mode == "single":
            self.mem = Memory(N_VERB + N_NOUN, d, heads, alpha)
        else:
            self.mem_v, self.mem_n = Memory(N_VERB, d, heads, alpha), Memory(N_NOUN, d, heads, alpha)
        self.gate_v, self.gate_n = Gate(d), Gate(d)
        self.head_verb = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_VERB))
        self.head_noun = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_NOUN))
        if present_encoder == "probe":
            from epic_starter.models.factorized_probe import FactorizedAttentiveProbe   # Member 2
            self.present = FactorizedAttentiveProbe(embed_dim=feat_dim, mode="factorized")
            self.present_v = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
            self.present_n = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
            for head in (self.head_verb, self.head_noun):         # memory starts as a zero correction
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
            if freeze_present:
                self.present.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_present and self.present_encoder == "probe":
            self.present.eval()                                   # frozen probe: no dropout
        return self

    def load_present(self, path, map_location="cpu"):
        """Initialise the present probe from Member 2's train_probe.py checkpoint (best.pt)."""
        if self.present_encoder != "probe":
            raise ValueError("load_present needs present_encoder='probe'")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if ckpt.get("mode") not in (None, "factorized") or ckpt.get("task") not in (None, "STA"):
            raise ValueError(f"Expected an STA 'factorized' probe checkpoint, got task={ckpt.get('task')} "
                             f"mode={ckpt.get('mode')}")
        self.present.load_state_dict(ckpt["model"])               # strict: shapes and keys must match
        return ckpt.get("epoch"), ckpt.get("metrics", {}).get("all", {}).get("action_mt5r")

    def encode(self, feats, valid, present_tokens=None, present_available=None):
        """Memory outputs for any fusion head (e.g. Member 4's ThreeWayGate), all [B, d]:
        verb_history / noun_history  h_t per field (identical in single mode)
        verb_present / noun_present  e_t per field (identical with the mean present encoder)
        history_available            [B] bool, at least one valid step before the last
        present_available            [B] bool, a present clip exists
        plus the auxiliary per-step logits, attention weights and (probe) base logits.
        """
        valid = valid.bool()
        e = self.fx(self.pool(feats.float())) * valid[..., None]  # invalid steps contribute nothing
        aux_v, aux_n = self.aux_verb(e), self.aux_noun(e)
        pv, pn = aux_v.softmax(-1).detach(), aux_n.softmax(-1).detach()
        pv, pn = pv * valid[..., None], pn * valid[..., None]
        z = {"history_available": valid[:, :-1].any(dim=1), "aux_verb_logits": aux_v, "aux_noun_logits": aux_n}
        if self.present_encoder == "mean":
            z.update(verb_present=e[:, -1], noun_present=e[:, -1], present_available=valid[:, -1])
        else:
            if present_tokens is None or present_available is None:
                raise ValueError("present_encoder='probe' needs present_tokens and present_available")
            avail = present_available.bool()
            ev, en = self.present.encode_tokens(present_tokens.float())
            keep = avail[:, None].float()
            z.update(verb_present=self.present_v(ev) * keep, noun_present=self.present_n(en) * keep,
                     present_available=avail,
                     base_verb_logits=self.present.verb_head(ev) * keep,
                     base_noun_logits=self.present.noun_head(en) * keep)
        if self.mode == "single":
            h, w = self.mem(torch.cat([pv, pn], -1), e, valid)
            z.update(verb_history=h, noun_history=h, attn=w)
        else:
            hv, wv = self.mem_v(pv, e, valid)
            hn, wn = self.mem_n(pn, e, valid)
            z.update(verb_history=hv, noun_history=hn, attn_verb=wv, attn_noun=wn)
        return z

    def forward(self, feats, valid, present_tokens=None, present_available=None):
        z = self.encode(feats, valid, present_tokens, present_available)
        extra = {k: z[k] for k in ("attn", "attn_verb", "attn_noun") if k in z}
        ov, gv = self.gate_v(z["verb_history"], z["verb_present"])
        on, gn = self.gate_n(z["noun_history"], z["noun_present"])
        extra.update(gate_verb=gv, gate_noun=gn)
        verb, noun = self.head_verb(ov), self.head_noun(on)
        if "base_verb_logits" in z:
            verb, noun = verb + z["base_verb_logits"], noun + z["base_noun_logits"]
        return {"verb_logits": verb, "noun_logits": noun,
                "aux_verb_logits": z["aux_verb_logits"], "aux_noun_logits": z["aux_noun_logits"], **extra}


class GatedAGAModel(nn.Module):
    """AGA memory + an external verb/noun fusion head with the GatedVerbNounHead interface
    (Member 4, epic_starter/models/gate.py): head((h_v, e_v, f_v), (h_n, e_n, f_n), available).

    Without a future branch it is zeros and marked unavailable, so the head fuses history and
    present only. The future branch comes either as `future=(f_verb, f_noun)` ([B, d] each) or from
    `future_encoder` (Member 4's FutureBranch, epic_starter/models/future_branch.py) applied to
    `future_tokens` [B, 24, D]; `future_available` [B] marks the rows that have them. `branches`
    keeps only some of ("history", "present", "future") for ablations: the others get weight 0.
    With the probe present encoder, the probe's logits stay the base prediction and the gated head
    adds to them.
    """

    BRANCHES = ("history", "present", "future")

    def __init__(self, memory: AGAMemoryModel, head: nn.Module, future_encoder: nn.Module | None = None,
                 branches=BRANCHES):
        super().__init__()
        unknown = set(branches) - set(self.BRANCHES)
        if unknown or not branches:
            raise ValueError(f"branches must be a non-empty subset of {self.BRANCHES}, got {branches}")
        self.memory, self.head, self.future_encoder = memory, head, future_encoder
        self.branches = tuple(b for b in self.BRANCHES if b in branches)
        self.register_buffer("branch_mask", torch.tensor([b in branches for b in self.BRANCHES]),
                             persistent=False)
        if memory.present_encoder == "probe":
            for lin in (head.verb_head, head.noun_head):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def forward(self, feats, valid, present_tokens=None, present_available=None, future=None,
                future_tokens=None, future_available=None):
        z = self.memory.encode(feats, valid, present_tokens, present_available)
        ref = z["verb_present"]
        if future is None and future_tokens is not None and self.future_encoder is not None:
            future = self.future_encoder(future_tokens, future_available)
        if future is None:
            fv = fn = torch.zeros_like(ref)
            has_future = torch.zeros(ref.shape[0], dtype=torch.bool, device=ref.device)
        else:
            fv, fn = future
            has_future = (torch.ones(ref.shape[0], dtype=torch.bool, device=ref.device)
                          if future_available is None else future_available.bool())
        available = torch.stack([z["history_available"], z["present_available"], has_future], dim=1)
        available = available & self.branch_mask
        out = self.head((z["verb_history"], z["verb_present"], fv), (z["noun_history"], z["noun_present"], fn),
                        available=available)
        if "base_verb_logits" in z:
            out["verb_logits"] = out["verb_logits"] + z["base_verb_logits"]
            out["noun_logits"] = out["noun_logits"] + z["base_noun_logits"]
        out.update({k: z[k] for k in ("aux_verb_logits", "aux_noun_logits", "attn", "attn_verb", "attn_noun")
                    if k in z})
        return out


def aga_loss(out, batch, aux_weight=0.5, verb_weight=None, noun_weight=None, label_smoothing=0.0):
    """CE on the final verb/noun plus auxiliary CE on labelled valid steps (-1 is ignored).
    Events with no valid step (use_prior) are excluded from the main loss: they get the prior.
    """
    keep = ~batch["use_prior"] if torch.is_tensor(batch["use_prior"]) else torch.ones(len(out["verb_logits"]), dtype=torch.bool)
    yv, yn = batch["verb_label"].clone(), batch["noun_label"].clone()
    yv[~keep], yn[~keep] = -1, -1
    ls = label_smoothing
    main = F.cross_entropy(out["verb_logits"], yv, weight=verb_weight, ignore_index=-1, label_smoothing=ls) \
         + F.cross_entropy(out["noun_logits"], yn, weight=noun_weight, ignore_index=-1, label_smoothing=ls)
    av, an = batch["aux_verb"].reshape(-1), batch["aux_noun"].reshape(-1)
    aux = 0.0
    if (av >= 0).any():
        aux = aux + F.cross_entropy(out["aux_verb_logits"].reshape(-1, N_VERB), av, ignore_index=-1)
    if (an >= 0).any():
        aux = aux + F.cross_entropy(out["aux_noun_logits"].reshape(-1, N_NOUN), an, ignore_index=-1)
    if not (yv >= 0).any():
        main = out["verb_logits"].sum() * 0.0
    return main + aux_weight * aux
