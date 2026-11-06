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

    def __init__(self, in_dim: int, dim: int | None = None, hidden: int = 256):
        super().__init__()
        dim = dim or in_dim
        self.proj = nn.ModuleList(nn.Linear(in_dim, dim) for _ in BRANCHES)
        self.norm = nn.ModuleList(nn.LayerNorm(dim) for _ in BRANCHES)
        self.score = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, history, present, future, available=None):
        inputs = (history, present, future)
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
        logits = self.score(torch.cat(branches, dim=-1)).masked_fill(~available, float("-inf"))
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
                 num_verbs: int = 97, num_nouns: int = 300):
        super().__init__()
        dim = dim or in_dim
        self.verb_gate = ThreeWayGate(in_dim, dim, hidden)
        self.noun_gate = ThreeWayGate(in_dim, dim, hidden)
        self.verb_head = nn.Linear(dim, num_verbs)
        self.noun_head = nn.Linear(dim, num_nouns)

    def forward(self, verb_inputs, noun_inputs, available=None):
        """verb_inputs / noun_inputs: (history, present, future), each [B, in_dim]."""
        verb_fused, verb_gates = self.verb_gate(*verb_inputs, available=available)
        noun_fused, noun_gates = self.noun_gate(*noun_inputs, available=available)
        return {"verb_logits": self.verb_head(verb_fused), "noun_logits": self.noun_head(noun_fused),
                "verb_gates": verb_gates, "noun_gates": noun_gates}
