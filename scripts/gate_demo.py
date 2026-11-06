"""Runnable three-way gate demo on synthetic vectors (Member 4, integration checkpoint).

Only one branch carries the label; the gate should learn to put its weight there. Rows marked
unavailable for that branch show the masking path. CPU only, a few seconds:
  python scripts/gate_demo.py --informative future
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from epic_starter.models.gate import BRANCHES, ThreeWayGate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--informative", choices=BRANCHES, default="future")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--samples", type=int, default=2048)
    p.add_argument("--steps", type=int, default=300)
    args = p.parse_args(argv)
    torch.manual_seed(0)
    labels = torch.randint(0, args.classes, (args.samples,))
    prototypes = 3 * torch.randn(args.classes, args.dim)
    inputs = {b: torch.randn(args.samples, args.dim) for b in BRANCHES}
    inputs[args.informative] = prototypes[labels] + 0.3 * torch.randn(args.samples, args.dim)
    available = torch.ones(args.samples, 3, dtype=torch.bool)
    target = BRANCHES.index(args.informative)
    available[: args.samples // 10, target] = False  # 10% of rows lack the informative branch
    gate, head = ThreeWayGate(args.dim, hidden=64), torch.nn.Linear(args.dim, args.classes)
    opt = torch.optim.Adam(list(gate.parameters()) + list(head.parameters()), lr=1e-2)
    for step in range(args.steps + 1):
        fused, weights = gate(*(inputs[b] for b in BRANCHES), available=available)
        logits = head(fused)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if step % 100 == 0:
            full = available[:, target]
            print(json.dumps({"step": step, "loss": round(float(loss), 4),
                              "accuracy": round(float((logits.argmax(1) == labels).float().mean()), 3),
                              "mean_gate_weights": dict(zip(BRANCHES, [round(float(w), 3) for w in weights[full].mean(0)])),
                              "weights_when_informative_missing":
                                  dict(zip(BRANCHES, [round(float(w), 3) for w in weights[~full].mean(0)]))}))
        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
