"""Run Member 2's frozen STA probe once over every bank clip and STA clip (Member 3's memory steps).

    python scripts/precompute_probe_steps.py --features-dir "$EPIC_FEATURES" \
        --probe-ckpt "$EPIC_PROBE_CKPT" --out-dir "$EPIC_PROBE_STEPS"

Output layout and row format: epic_starter/memory/probe_steps.py. Then train the memory with
`train_sta_memory.py --step-features "$EPIC_PROBE_STEPS"` (scripts/sta_memory_pretrain.slurm <seed> probe).
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from epic_starter.memory.probe_steps import precompute


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", required=True, help="consolidated V-JEPA feature folder (Member 2)")
    p.add_argument("--probe-ckpt", required=True, help="Member 2's STA factorized probe (best.pt)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    a = p.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads())))
    out = precompute(a.features_dir, a.probe_ckpt, a.out_dir, a.batch_size, log=lambda m: print(m, flush=True))
    print(f"probe steps in {out}")
    return out


if __name__ == "__main__":
    main()
