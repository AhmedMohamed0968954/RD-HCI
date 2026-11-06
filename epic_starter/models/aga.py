"""Action-Guided Attention memory for STA (Tai et al., 2026, arXiv:2603.01743), adapted.

Input: per-step features [B,S,T,D] from HistoryDataset plus a validity mask [B,S].
Per step, an auxiliary head predicts verb/noun probabilities p_k (detached; the model's
own predictions are used in training and testing, never labels).
Query = EMA of p over valid steps (alpha), keys = p of earlier steps, values = their features.
mode="single": one memory over concatenated [p_verb, p_noun] (97+300 = 397 dims).
mode="decoupled": a verb memory and a noun memory with their own keys/values.
A sigmoid gate mixes the retrieved history h with the current-step evidence e (AGA's 2-way
gate). `encode()` exposes h and e per field so GatedAGAModel can use Member 4's ThreeWayGate.
Token pooling is a plain mean here; Member 2's factorized probe can replace `pool`.
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
    def __init__(self, feat_dim, d=256, heads=4, alpha=0.8, mode="decoupled", dropout=0.3):
        super().__init__()
        if mode not in {"single", "decoupled"}:
            raise ValueError("mode must be single or decoupled")
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.mode = mode
        self.pool = lambda x: x.mean(dim=2)                       # [B,S,T,D] -> [B,S,D]
        self.fx = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, d))
        self.aux_verb, self.aux_noun = nn.Linear(d, N_VERB), nn.Linear(d, N_NOUN)
        if mode == "single":
            self.mem = Memory(N_VERB + N_NOUN, d, heads, alpha)
            self.gate = Gate(d)
        else:
            self.mem_v, self.mem_n = Memory(N_VERB, d, heads, alpha), Memory(N_NOUN, d, heads, alpha)
            self.gate_v, self.gate_n = Gate(d), Gate(d)
        self.head_verb = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_VERB))
        self.head_noun = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, N_NOUN))

    def encode(self, feats, valid):
        """Memory outputs for any fusion head (e.g. Member 4's ThreeWayGate), all [B, d]:
        verb_history / noun_history  h_t per field (identical in single mode)
        present                      e_t of the last step (the clip ending at the cutoff)
        history_available            [B] bool, at least one valid step before the last
        present_available            [B] bool, the last step is valid
        plus the auxiliary per-step logits and attention weights.
        """
        valid = valid.bool()
        e = self.fx(self.pool(feats.float())) * valid[..., None]  # invalid steps contribute nothing
        aux_v, aux_n = self.aux_verb(e), self.aux_noun(e)
        pv, pn = aux_v.softmax(-1).detach(), aux_n.softmax(-1).detach()
        pv, pn = pv * valid[..., None], pn * valid[..., None]
        z = {"present": e[:, -1], "history_available": valid[:, :-1].any(dim=1),
             "present_available": valid[:, -1], "aux_verb_logits": aux_v, "aux_noun_logits": aux_n}
        if self.mode == "single":
            h, w = self.mem(torch.cat([pv, pn], -1), e, valid)
            z.update(verb_history=h, noun_history=h, attn=w)
        else:
            hv, wv = self.mem_v(pv, e, valid)
            hn, wn = self.mem_n(pn, e, valid)
            z.update(verb_history=hv, noun_history=hn, attn_verb=wv, attn_noun=wn)
        return z

    def forward(self, feats, valid):
        z = self.encode(feats, valid)
        extra = {k: z[k] for k in ("attn", "attn_verb", "attn_noun") if k in z}
        if self.mode == "single":
            o, g = self.gate(z["verb_history"], z["present"])
            ov = on = o
            extra["gate"] = g
        else:
            ov, gv = self.gate_v(z["verb_history"], z["present"])
            on, gn = self.gate_n(z["noun_history"], z["present"])
            extra.update(gate_verb=gv, gate_noun=gn)
        return {"verb_logits": self.head_verb(ov), "noun_logits": self.head_noun(on),
                "aux_verb_logits": z["aux_verb_logits"], "aux_noun_logits": z["aux_noun_logits"], **extra}


class GatedAGAModel(nn.Module):
    """AGA memory + an external verb/noun fusion head with the GatedVerbNounHead interface
    (Member 4, epic_starter/models/gate.py): head((h_v, e, f_v), (h_n, e, f_n), available).

    Until future latents exist the future branch is zeros and marked unavailable, so the head
    fuses history and present only. Pass `future=(f_verb, f_noun)` ([B, d] each) once Member 4's
    predictor features are wired in.
    """

    def __init__(self, memory: AGAMemoryModel, head: nn.Module):
        super().__init__()
        self.memory, self.head = memory, head

    def forward(self, feats, valid, future=None):
        z = self.memory.encode(feats, valid)
        B = z["present"].shape[0]
        if future is None:
            fv = fn = torch.zeros_like(z["present"])
            has_future = torch.zeros(B, dtype=torch.bool, device=z["present"].device)
        else:
            fv, fn = future
            has_future = torch.ones(B, dtype=torch.bool, device=z["present"].device)
        available = torch.stack([z["history_available"], z["present_available"], has_future], dim=1)
        out = self.head((z["verb_history"], z["present"], fv), (z["noun_history"], z["present"], fn),
                        available=available)
        out.update({k: z[k] for k in ("aux_verb_logits", "aux_noun_logits", "attn", "attn_verb", "attn_noun")
                    if k in z})
        return out


def aga_loss(out, batch, aux_weight=0.5, verb_weight=None, noun_weight=None):
    """CE on the final verb/noun plus auxiliary CE on labelled valid steps (-1 is ignored).
    Events with no valid step (use_prior) are excluded from the main loss: they get the prior.
    """
    keep = ~batch["use_prior"] if torch.is_tensor(batch["use_prior"]) else torch.ones(len(out["verb_logits"]), dtype=torch.bool)
    yv, yn = batch["verb_label"].clone(), batch["noun_label"].clone()
    yv[~keep], yn[~keep] = -1, -1
    main = F.cross_entropy(out["verb_logits"], yv, weight=verb_weight, ignore_index=-1) \
         + F.cross_entropy(out["noun_logits"], yn, weight=noun_weight, ignore_index=-1)
    av, an = batch["aux_verb"].reshape(-1), batch["aux_noun"].reshape(-1)
    aux = 0.0
    if (av >= 0).any():
        aux = aux + F.cross_entropy(out["aux_verb_logits"].reshape(-1, N_VERB), av, ignore_index=-1)
    if (an >= 0).any():
        aux = aux + F.cross_entropy(out["aux_noun_logits"].reshape(-1, N_NOUN), an, ignore_index=-1)
    if not (yv >= 0).any():
        main = out["verb_logits"].sum() * 0.0
    return main + aux_weight * aux
