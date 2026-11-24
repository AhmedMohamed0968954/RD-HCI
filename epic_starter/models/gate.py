"""Normalized three-way gate (technical blueprint §5.3) and a verb/noun head built on it.

The gate fuses three same-shaped vectors for one field (verb or noun):
history h_t (temporal memory), present e_t (observed clip) and future f_t (predicted latent).
Each branch is projected and LayerNorm-ed so the branches have a comparable scale before mixing.
"""
from __future__ import annotations
import torch
from torch import nn

BRANCHES = ("history", "present", "future")


class ThreeWayGate(nn.Module):
    """Fuse history, present and future vectors with softmax gate weights.

    available: optional bool [B, 3] in BRANCHES order. An unavailable branch (no history yet,
    predictor not run, ...) gets weight 0 and its values are ignored, even if they are NaN.
    Rows with no available branch return a zero vector and zero weights; callers should fall
    back to the training prior for those rows (empty-history STA events).
    """

    def __init__(self, in_dim: int, dim: int | None = None, hidden: int = 256, extra_dim: int = 0):
        super().__init__()
        dim = dim or in_dim
        self.extra_dim = extra_dim
        self.proj = nn.ModuleList(nn.Linear(in_dim, dim) for _ in BRANCHES)
        self.norm = nn.ModuleList(nn.LayerNorm(dim) for _ in BRANCHES)
        # extra_dim > 0: extra gate inputs next to the branches (e.g. the present probe's uncertainty)
        self.score = nn.Sequential(nn.Linear(3 * dim + extra_dim, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, history, present, future, available=None, extra=None):
        inputs = (history, present, future)
        if (extra is None) != (self.extra_dim == 0) or (
                extra is not None and extra.shape != (history.shape[0], self.extra_dim)):
            raise ValueError(f"this gate takes extra inputs of shape [B, {self.extra_dim}]"
                             if self.extra_dim else "this gate takes no extra inputs")
        if any(x.shape != history.shape for x in inputs):
            raise ValueError("history, present and future must have the same shape")
        if available is None:
            available = torch.ones(history.shape[0], 3, dtype=torch.bool, device=history.device)
        if available.shape != (history.shape[0], 3) or available.dtype != torch.bool:
            raise ValueError("available must be a bool tensor of shape [B, 3]")
        branches = []
        for i, (x, proj, norm) in enumerate(zip(inputs, self.proj, self.norm)):
            keep = available[:, i:i + 1]
            # Zero missing inputs BEFORE the layers: NaN * 0 would still poison the weight gradients.
            x = torch.where(keep, x, torch.zeros_like(x))
            out = norm(proj(x))
            branches.append(torch.where(keep, out, torch.zeros_like(out)))
        score_in = torch.cat(branches if extra is None else [*branches, extra.to(branches[0].dtype)], dim=-1)
        logits = self.score(score_in).masked_fill(~available, float("-inf"))
        any_branch = available.any(dim=-1, keepdim=True)
        # Softmax of an all -inf row is NaN, so those rows get zero logits first and zero weights after.
        weights = torch.softmax(torch.where(any_branch, logits, torch.zeros_like(logits)), dim=-1)
        weights = torch.where(any_branch, weights, torch.zeros_like(weights))
        fused = sum(weights[:, i:i + 1] * branch for i, branch in enumerate(branches))
        return fused, weights


class GatedVerbNounHead(nn.Module):
    """Separate gates for verb and noun (the blueprint's c in {v, n}), then the two classifiers.

    Returns the agreed interface: verb scores [B, 97] and noun scores [B, 300] (logits),
    plus the gate weights [B, 3] per field for the interpretability plots.
    """

    def __init__(self, in_dim: int, dim: int | None = None, hidden: int = 256,
                 num_verbs: int = 97, num_nouns: int = 300, extra_dim: int = 0):
        super().__init__()
        dim = dim or in_dim
        self.extra_dim = extra_dim
        self.verb_gate = ThreeWayGate(in_dim, dim, hidden, extra_dim)
        self.noun_gate = ThreeWayGate(in_dim, dim, hidden, extra_dim)
        self.verb_head = nn.Linear(dim, num_verbs)
        self.noun_head = nn.Linear(dim, num_nouns)

    def forward(self, verb_inputs, noun_inputs, available=None, extra=None):
        """verb_inputs / noun_inputs: (history, present, future), each [B, in_dim]; extra: the
        same [B, extra_dim] extra gate inputs for both fields (extra_dim > 0 only)."""
        verb_fused, verb_gates = self.verb_gate(*verb_inputs, available=available, extra=extra)
        noun_fused, noun_gates = self.noun_gate(*noun_inputs, available=available, extra=extra)
        return {"verb_logits": self.verb_head(verb_fused), "noun_logits": self.noun_head(noun_fused),
                "verb_gates": verb_gates, "noun_gates": noun_gates}


def gate_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Entropy (nats) of each row of gate weights [B, 3]; unavailable branches (weight 0) add 0,
    so a row with one available branch has entropy 0. Maximum ln 3 for an even three-way split."""
    w = weights.clamp_min(0)
    return -(w * torch.log(w.clamp_min(1e-12))).sum(dim=-1)


def mean_gate_entropy(out: dict) -> torch.Tensor:
    """Mean gate entropy over verb and noun gates and over the rows that use the gate (rows with no
    available branch have all-zero weights and are left out). Training adds -lambda * this to the
    loss (--gate-entropy) so the gate keeps more than one branch instead of collapsing onto one."""
    total, fields = 0.0, 0
    for key in ("verb_gates", "noun_gates"):
        w = out[key]
        used = w.sum(dim=-1) > 0
        if used.any():
            total = total + gate_entropy(w[used]).mean()
            fields += 1
    return total / fields if fields else out["verb_gates"].sum() * 0.0
