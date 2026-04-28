#!/usr/bin/env python3
"""
IncrementalSQDBackend benchmark on N2-CAS(10,20) 40Q.

Compares three backends on the main 40Q production system:
  1. legacy        — original solve_fermion (cold, RDM each iter)
  2. ws_only       — solve_fermion + SCIvector ci0 (warm-start works)
  3. incremental   — IncrementalSQDBackend (full optimization)

Uses Config E (top_k=10K, n_samples=30K) for reasonable runtime.
"""
import sys, time, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

def make_cfg(warm_start, use_incremental):
    return HINQSSQDConfig(
        n_samples=30000,
        top_k=10000,
        max_basis_size=0,
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
        warm_start=warm_start,
        use_incremental_sqd=use_incremental,
    )

VARIANTS = [
    ("incremental", True,  True),
]

print(f"\nN2-CAS(10,20) 40Q (cc-pVTZ)", flush=True)
print(f"  Config: top_k=10000, n_samples=30000", flush=True)
print(f"  Variant: incremental only", flush=True)

H, info = get_molecule("N2-CAS(10,20)", device="cuda")
print(f"  n_orb={H.n_orbitals}, hilbert={info}", flush=True)

results = {}

for vlabel, ws, incr in VARIANTS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {vlabel}  (warm_start={ws}, use_incremental={incr})", flush=True)
    print(f"{'='*70}", flush=True)

    np.random.seed(42); torch.manual_seed(42)

    t0 = time.time()
    r = run_hi_nqs_sqd(H, info, config=make_cfg(ws, incr))
    elapsed = time.time() - t0

    print(f"\n  {vlabel}: E={r.energy:.10f}, basis={r.diag_dim}, "
          f"time={elapsed:.0f}s ({elapsed/3600:.2f}h), converged={r.converged}", flush=True)

    results[vlabel] = {
        "energy": float(r.energy),
        "basis": int(r.diag_dim),
        "time": float(elapsed),
        "converged": r.converged,
    }

# Summary
print(f"\n{'='*70}", flush=True)
print(f"  N2-CAS(10,20) 40Q: incremental backend result", flush=True)
print(f"{'='*70}", flush=True)
r = results["incremental"]
print(f"  E={r['energy']:.10f}, basis={r['basis']}, "
      f"time={r['time']:.0f}s ({r['time']/3600:.2f}h)", flush=True)

# vs CIPSI baseline
cipsi_E = -109.2138472581
cipsi_t = 16809
print(f"\n  CIPSI baseline: E={cipsi_E:.10f}, basis=53998, "
      f"time={cipsi_t}s ({cipsi_t/3600:.2f}h, CPU)", flush=True)
err_mHa = (r["energy"] - cipsi_E) * 1000
t_ratio = cipsi_t / r["time"]
print(f"  HI-NQS incremental: ΔE={err_mHa:+.4f} mHa, "
      f"time-vs-CIPSI: {t_ratio:.2f}x", flush=True)

with open("incremental_sqd_40q_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to incremental_sqd_40q_results.json", flush=True)
