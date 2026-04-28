#!/usr/bin/env python3
"""
Test IncrementalSQDBackend on C2H2 (24Q) and NH3 (16Q).

Compares three backends:
  1. legacy solve_fermion (cold start, RDM computation each iter)
  2. solve_fermion + warm-start ci0 (SCIvector)
  3. IncrementalSQDBackend (persistent myci, no RDM, warm-start)
"""
import sys, time, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

def make_cfg(n_samples, top_k, max_iter, warm_start, use_incremental):
    return HINQSSQDConfig(
        n_samples=n_samples, top_k=top_k, max_basis_size=0,
        max_iterations=max_iter, convergence_threshold=1e-12, convergence_window=20,
        nqs_steps=5, nqs_lr=3e-4, entropy_weight=0.15,
        warm_start=warm_start,
        use_incremental_sqd=use_incremental,
    )

SYSTEMS = [
    ("NH3",  10000, 2000, 15),
    ("C2H2", 20000, 4000, 15),
]

VARIANTS = [
    ("legacy",        False, False),  # solve_fermion cold
    ("ws_only",       True,  False),  # solve_fermion + ci0 warm-start
    ("incremental",   True,  True),   # IncrementalSQDBackend
]

results = {}

for name, n_samp, top_k, max_iter in SYSTEMS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}  (n_samples={n_samp}, top_k={top_k}, max_iter={max_iter})", flush=True)
    print(f"{'='*70}", flush=True)

    H, info = get_molecule(name, device="cuda")
    results[name] = {}

    for vlabel, ws, incr in VARIANTS:
        print(f"\n  --- {vlabel} (warm_start={ws}, incremental={incr}) ---", flush=True)
        np.random.seed(42); torch.manual_seed(42)

        t0 = time.time()
        r = run_hi_nqs_sqd(H, info, config=make_cfg(n_samp, top_k, max_iter, ws, incr))
        elapsed = time.time() - t0

        print(f"  {vlabel}: E={r.energy:.10f}, basis={r.diag_dim}, t={elapsed:.1f}s", flush=True)

        results[name][vlabel] = {
            "energy": float(r.energy),
            "basis": int(r.diag_dim),
            "time": float(elapsed),
            "converged": r.converged,
        }

# Summary
print(f"\n{'='*70}", flush=True)
print(f"  SQD BACKEND COMPARISON", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'System':<8} {'Variant':<14} {'Energy':>16} {'basis':>7} {'Time':>8} {'Speedup':>8}", flush=True)
print(f"  {'-'*70}", flush=True)
for name in results:
    base_t = results[name]["legacy"]["time"]
    for vlabel in ["legacy", "ws_only", "incremental"]:
        r = results[name][vlabel]
        speedup = base_t / r["time"] if r["time"] > 0 else 0
        print(f"  {name:<8} {vlabel:<14} {r['energy']:>16.10f} {r['basis']:>7} "
              f"{r['time']:>7.1f}s {speedup:>7.2f}x", flush=True)

with open("incremental_sqd_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to incremental_sqd_results.json", flush=True)
