#!/usr/bin/env python3
"""
Warm-start vs cold-start benchmark on C2H2 (24Q STO-3G).

Hilbert = C(12,7)^2 = 627,264 — small enough to converge fast,
big enough to show Davidson iterations difference.
"""
import sys, time, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

H, info = get_molecule("C2H2", device="cuda")
n_orb = H.n_orbitals
print(f"C2H2 24Q: n_orb={n_orb}, Hilbert={info}", flush=True)

# Reasonable mid-size config
def make_cfg(warm_start):
    return HINQSSQDConfig(
        n_samples=30000,
        top_k=5000,
        max_basis_size=0,
        max_iterations=20,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
        warm_start=warm_start,
    )

print(f"\n{'='*70}", flush=True)
print(f"  WARM-START on C2H2 24Q", flush=True)
print(f"{'='*70}", flush=True)

np.random.seed(42)
torch.manual_seed(42)

t0 = time.time()
r = run_hi_nqs_sqd(H, info, config=make_cfg(warm_start=True))
elapsed = time.time() - t0

print(f"\n  WARM-start: E={r.energy:.10f}, basis={r.diag_dim}, "
      f"time={elapsed:.0f}s, converged={r.converged}", flush=True)

results = {
    "WARM-start": {
        "energy": float(r.energy),
        "basis": int(r.diag_dim),
        "time": float(elapsed),
        "converged": r.converged,
    }
}

with open("warmstart_c2h2_results.json", "w") as f:
    json.dump(results, f, indent=2)
