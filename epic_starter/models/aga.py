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

Memory steps (step_encoder):
  "mean"   each past second is the mean of its V-JEPA tokens; an auxiliary per-step head predicts
           verb/noun probabilities, which serve as keys and query.
  "probe"  each past second is Member 2's frozen probe on that second's clip (precomputed,
           epic_starter/memory/probe_steps.py): the verb memory's values are the probe's verb
           embeddings and its keys the probe's verb probabilities, likewise for nouns. The
           auxiliary heads stay (same training signal), but no longer produce the keys.
           step_keys="features" (plain-attention baseline): keys and query come from the probe's
           embeddings instead of its predictions; everything else is unchanged.

Memory pre-training (two stages):
  1. present_encoder="none": no present branch and no gate. The STA label is predicted from the
     retrieved history h alone, so the memory has to learn to be useful on its own.
  2. A probe model loads the memory part of that checkpoint (`load_memory`: the per-step feature
     projection, the auxiliary heads and the AGA attention) and trains the gate and the correction
     head on top of it, with the memory fine-tuned or frozen (freeze_memory=True).
"""
from __future__ import annotations
import math
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
    present_encoder="none"   memory only (stage 1 of memory pre-training): no present branch and no
                             gate; the verb/noun heads read the retrieved history h directly.
    freeze_memory=True       keep the memory part (memory_parts()) fixed, e.g. after `load_memory`.
    """

    def __init__(self, feat_dim, d=256, heads=4, alpha=0.8, mode="decoupled", dropout=0.3,
                 present_encoder="mean", freeze_present=False, freeze_memory=False, step_encoder="mean",
                 step_keys="predictions"):
        super().__init__()
        if mode not in {"single", "decoupled"}:
            raise ValueError("mode must be single or decoupled")
        if present_encoder not in {"mean", "probe", "none"}:
            raise ValueError("present_encoder must be mean, probe or none")
        if step_encoder not in {"mean", "probe"}:
            raise ValueError("step_encoder must be mean or probe")
        if step_encoder == "probe" and mode != "decoupled":
            raise ValueError("step_encoder='probe' has separate verb/noun steps: use mode='decoupled'")
        if step_keys not in {"predictions", "features"} or (step_keys == "features" and step_encoder != "probe"):
            raise ValueError("step_keys must be predictions, or features with step_encoder='probe'")
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.mode, self.present_encoder, self.freeze_present = mode, present_encoder, freeze_present
        self.freeze_memory, self.step_encoder, self.step_keys = freeze_memory, step_encoder, step_keys
        # what a memory checkpoint must match to be loaded into this model (load_memory)
        self.memory_shape = {"feat_dim": feat_dim, "d": d, "heads": heads, "alpha": alpha, "mode": mode,
                             "step_encoder": step_encoder, "step_keys": step_keys}
        self.pool = lambda x: x.mean(dim=2)                       # [B,S,T,D] -> [B,S,D]
        if step_encoder == "mean":
            self.fx = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, d))
        else:                                                     # probe outputs per step
            from epic_starter.memory.probe_steps import layout, step_dim
            self.step_layout, self.step_dim = layout(feat_dim), step_dim(feat_dim)
            self.fv = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, d))
            self.fn = nn.Sequential(nn.LayerNorm(feat_dim), nn.Dropout(dropout), nn.Linear(feat_dim, d))
        self.aux_verb, self.aux_noun = nn.Linear(d, N_VERB), nn.Linear(d, N_NOUN)
        if mode == "single":
            self.mem = Memory(N_VERB + N_NOUN, d, heads, alpha)
        else:
            kv, kn = (feat_dim, feat_dim) if step_keys == "features" else (N_VERB, N_NOUN)
            self.mem_v, self.mem_n = Memory(kv, d, heads, alpha), Memory(kn, d, heads, alpha)
        if present_encoder != "none":
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
        if freeze_memory:
            for part in self.memory_parts().values():
                part.requires_grad_(False)

    def memory_parts(self) -> dict[str, nn.Module]:
        """The memory proper, shared by both pre-training stages: per-step feature projection,
        auxiliary per-step heads (they produce the keys and the query) and the AGA attention.
        Gates, heads and the present encoder are not part of it."""
        names = (["fx"] if self.step_encoder == "mean" else ["fv", "fn"]) + ["aux_verb", "aux_noun"]
        names += ["mem"] if self.mode == "single" else ["mem_v", "mem_n"]
        return {n: getattr(self, n) for n in names}

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_present and self.present_encoder == "probe":
            self.present.eval()                                   # frozen probe: no dropout
        if self.freeze_memory:
            for part in self.memory_parts().values():
                part.eval()                                       # frozen memory: no dropout
        return self

    def load_memory(self, path, map_location="cpu"):
        """Initialise the memory part from a train_sta_memory.py checkpoint (best.pt), normally a
        memory-only run (stage 1). Also accepts a GatedAGAModel checkpoint ('memory.' prefix).
        The feature size, d, heads, alpha and mode must match; S may differ (no weight depends on it).
        Returns the source run's epoch, val action MT5R and present encoder."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        ctor = {"step_encoder": "mean", "step_keys": "predictions", **(ckpt.get("ctor") or {})}   # older runs
        diff = {k: (ctor.get(k), v) for k, v in self.memory_shape.items() if ctor.get(k) != v}
        if diff:
            raise ValueError(f"Memory checkpoint {path} does not match this model "
                             f"(checkpoint, model): {diff}")
        state = ckpt["state_dict"]
        if any(k.startswith("memory.") for k in state):          # saved from GatedAGAModel
            state = {k[len("memory."):]: v for k, v in state.items() if k.startswith("memory.")}
        for name, part in self.memory_parts().items():
            sub = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
            part.load_state_dict(sub)                             # strict: every weight must be there
        return ckpt.get("epoch"), ckpt.get("val_action_mt5r"), ctor.get("present_encoder")

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
        if self.step_encoder == "mean":
            e = self.fx(self.pool(feats.float())) * valid[..., None]  # invalid steps contribute nothing
            sv = sn = e                                           # step embeddings per field
            aux_v, aux_n = self.aux_verb(e), self.aux_noun(e)
            pv, pn = aux_v.softmax(-1).detach(), aux_n.softmax(-1).detach()
        else:                                                     # Member 2's probe on each step's clip
            if feats.shape[2] != 1 or feats.shape[3] != self.step_dim:
                raise ValueError(f"probe steps must be [B,S,1,{self.step_dim}] (probe_steps.py), "
                                 f"got {tuple(feats.shape)}")
            x = feats[:, :, 0].float()
            part = lambda k: x[..., slice(*self.step_layout[k])]
            sv = self.fv(part("verb_emb")) * valid[..., None]
            sn = self.fn(part("noun_emb")) * valid[..., None]
            aux_v, aux_n = self.aux_verb(sv), self.aux_noun(sn)
            if self.step_keys == "predictions":
                pv, pn = part("verb_logits").softmax(-1), part("noun_logits").softmax(-1)   # frozen probe
            else:                                                 # plain attention: embeddings as keys
                pv, pn = part("verb_emb"), part("noun_emb")
        pv, pn = pv * valid[..., None], pn * valid[..., None]
        z = {"history_available": valid[:, :-1].any(dim=1), "aux_verb_logits": aux_v, "aux_noun_logits": aux_n}
        if self.present_encoder == "mean":
            z.update(verb_present=sv[:, -1], noun_present=sn[:, -1], present_available=valid[:, -1])
        elif self.present_encoder == "none":                       # memory only: no present branch
            zeros = torch.zeros_like(sv[:, -1])
            z.update(verb_present=zeros, noun_present=zeros, present_available=torch.zeros_like(valid[:, -1]))
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
            h, w = self.mem(torch.cat([pv, pn], -1), sv, valid)
            z.update(verb_history=h, noun_history=h, attn=w)
        else:
            hv, wv = self.mem_v(pv, sv, valid)
            hn, wn = self.mem_n(pn, sn, valid)
            z.update(verb_history=hv, noun_history=hn, attn_verb=wv, attn_noun=wn)
        return z

    def forward(self, feats, valid, present_tokens=None, present_available=None):
        z = self.encode(feats, valid, present_tokens, present_available)
        extra = {k: z[k] for k in ("attn", "attn_verb", "attn_noun") if k in z}
        if self.present_encoder == "none":                       # stage 1: predict from history alone
            return {"verb_logits": self.head_verb(z["verb_history"]),
                    "noun_logits": self.head_noun(z["noun_history"]),
                    "aux_verb_logits": z["aux_verb_logits"], "aux_noun_logits": z["aux_noun_logits"], **extra}
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

    Gate-collapse fixes (Member 4, all off by default):
    branch_dropout  p: in training only, a row (with history available) loses, with probability p,
                    the present branch, the future branch or both (one of the three at random), so
                    the gate must also learn to use the history; history is never dropped and the
                    probe's base logits are untouched. Evaluation always uses every branch.
    gate_uncertainty: the gate also sees the normalised entropy of the frozen probe's verb and noun
                    predictions [u_v, u_n] in [0, 1] (head built with extra_dim=2), so it can trust
                    history / future more when the present clip is ambiguous.
    """

    BRANCHES = ("history", "present", "future")

    def __init__(self, memory: AGAMemoryModel, head: nn.Module, future_encoder: nn.Module | None = None,
                 branches=BRANCHES, branch_dropout: float = 0.0, gate_uncertainty: bool = False):
        super().__init__()
        unknown = set(branches) - set(self.BRANCHES)
        if unknown or not branches:
            raise ValueError(f"branches must be a non-empty subset of {self.BRANCHES}, got {branches}")
        if memory.present_encoder == "none":
            raise ValueError("GatedAGAModel needs a present encoder (memory-only models have no present branch)")
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError(f"branch_dropout must be in [0, 1), got {branch_dropout}")
        if gate_uncertainty and memory.present_encoder != "probe":
            raise ValueError("gate_uncertainty needs the probe present encoder (its predictions)")
        if getattr(head, "extra_dim", 0) != (2 if gate_uncertainty else 0):
            raise ValueError("the head's extra_dim must be 2 with gate_uncertainty, else 0")
        if getattr(future_encoder, "needs_obs", False) and memory.present_encoder != "probe":
            raise ValueError("this future encoder reads the present clip's tokens: needs the probe present encoder")
        self.memory, self.head, self.future_encoder = memory, head, future_encoder
        self.branch_dropout, self.gate_uncertainty = branch_dropout, gate_uncertainty
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
            kw = {}
            if getattr(self.future_encoder, "needs_obs", False):     # factorized_pred_1664: + present clip
                kw["obs_tokens"] = present_tokens
            future = self.future_encoder(future_tokens, future_available, **kw)
        if future is None:
            fv = fn = torch.zeros_like(ref)
            has_future = torch.zeros(ref.shape[0], dtype=torch.bool, device=ref.device)
        else:
            fv, fn = future
            has_future = (torch.ones(ref.shape[0], dtype=torch.bool, device=ref.device)
                          if future_available is None else future_available.bool())
        available = torch.stack([z["history_available"], z["present_available"], has_future], dim=1)
        available = available & self.branch_mask
        if self.training and self.branch_dropout > 0:
            available = drop_branches(available, self.branch_dropout)
        extra = None
        if self.gate_uncertainty:
            extra = torch.stack([normalised_entropy(z["base_verb_logits"]),
                                 normalised_entropy(z["base_noun_logits"])], dim=1)
        kw = {} if extra is None else {"extra": extra}
        out = self.head((z["verb_history"], z["verb_present"], fv), (z["noun_history"], z["noun_present"], fn),
                        available=available, **kw)
        if "base_verb_logits" in z:
            out["verb_logits"] = out["verb_logits"] + z["base_verb_logits"]
            out["noun_logits"] = out["noun_logits"] + z["base_noun_logits"]
        out.update({k: z[k] for k in ("aux_verb_logits", "aux_noun_logits", "attn", "attn_verb", "attn_noun")
                    if k in z})
        return out


def drop_branches(available: torch.Tensor, p: float) -> torch.Tensor:
    """Gate branch dropout (training only): each row whose history is available loses, with
    probability p, its present branch, its future branch or both (uniformly one of the three).
    History is never dropped, so every row keeps at least one branch."""
    hit = (torch.rand(available.shape[0], device=available.device) < p) & available[:, 0]
    choice = torch.randint(0, 3, (available.shape[0],), device=available.device)   # 0 P, 1 F, 2 both
    out = available.clone()
    out[:, 1] &= ~(hit & (choice != 1))
    out[:, 2] &= ~(hit & (choice != 0))
    return out


def normalised_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits) divided by log(C): 0 = certain, 1 = uniform (also all-zero rows,
    i.e. events without a present clip). Detached: the gate reads it, it is not trained through it."""
    logp = torch.log_softmax(logits.detach().float(), dim=-1)
    return -(logp.exp() * logp).sum(-1) / math.log(logits.shape[-1])


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
