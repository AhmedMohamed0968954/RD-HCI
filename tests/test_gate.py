"""Three-way gate (Member 4): weights, missing branches, learning behaviour and the head interface."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from epic_starter.models.gate import GatedVerbNounHead, ThreeWayGate


def branches(batch=6, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, dim, generator=g) for _ in range(3)]


def test_weights_form_a_distribution():
    gate = ThreeWayGate(8, 12)
    fused, weights = gate(*branches())
    assert fused.shape == (6, 12) and weights.shape == (6, 3)
    assert (weights >= 0).all() and torch.allclose(weights.sum(dim=1), torch.ones(6))


def test_unavailable_branch_is_ignored_even_if_nan():
    gate = ThreeWayGate(8)
    h, e, f = branches()
    f = torch.full_like(f, float("nan"))  # e.g. predictor not run for these rows
    available = torch.tensor([[True, True, False]] * 6)
    fused, weights = gate(h, e, f, available)
    assert torch.isfinite(fused).all() and (weights[:, 2] == 0).all()
    fused.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in gate.parameters() if p.grad is not None)


def test_no_available_branch_gives_zero_output():
    gate = ThreeWayGate(8)
    available = torch.tensor([[False, False, False], [True, False, True]])
    fused, weights = gate(*branches(batch=2), available)
    assert (fused[0] == 0).all() and (weights[0] == 0).all()
    assert torch.allclose(weights[1].sum(), torch.tensor(1.0)) and weights[1, 1] == 0


def test_gate_learns_to_use_the_informative_branch():
    torch.manual_seed(0)
    n, dim, classes = 512, 16, 4
    labels = torch.randint(0, classes, (n,))
    prototypes = 3 * torch.randn(classes, dim)
    history, present = torch.randn(n, dim), torch.randn(n, dim)
    future = prototypes[labels] + 0.3 * torch.randn(n, dim)  # only the future branch carries the label
    gate, head = ThreeWayGate(dim, dim, hidden=32), torch.nn.Linear(dim, classes)
    opt = torch.optim.Adam(list(gate.parameters()) + list(head.parameters()), lr=1e-2)
    for _ in range(300):
        fused, weights = gate(history, present, future)
        loss = torch.nn.functional.cross_entropy(head(fused), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    accuracy = (head(fused).argmax(dim=1) == labels).float().mean()
    assert accuracy > 0.9
    assert weights[:, 2].mean() > 0.6, weights.mean(dim=0)


def test_head_returns_the_agreed_interface():
    head = GatedVerbNounHead(in_dim=8, dim=16)
    out = head(branches(batch=5), branches(batch=5, seed=1))
    assert out["verb_logits"].shape == (5, 97) and out["noun_logits"].shape == (5, 300)
    assert out["verb_gates"].shape == (5, 3) and out["noun_gates"].shape == (5, 3)
