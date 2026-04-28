#!/usr/bin/env python3
"""
Warm-start validation: H2O 14Q + N2-CAS(10,12) 24Q + C2H2 24Q.
Compare cold vs warm to verify correctness and measure speedup.
"""
import sys, time, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

def make_cfg(warm_start, n_samples, top_k, max_iter):
    return HINQSSQDConfig(
        n_samples=n_samples, top_k=top_k, max_basis_size=0,
        max_iterations=max_iter, convergence_threshold=1e-12, convergence_window=20,
        nqs_steps=5, nqs_lr=3e-4, entropy_weight=0.15,
        warm_start=warm_start,
    )

SYSTEMS = [
    ("H2O",            2000,  500,  10),
    ("N2-CAS(10,12)",  10000, 2000, 15),
    ("C2H2",           20000, 4000, 15),
]

results = {}

for name, n_samp, top_k, max_iter in SYSTEMS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}  (n_samples={n_samp}, top_k={top_k}, max_iter={max_iter})", flush=True)
    print(f"{'='*70}", flush=True)

    H, info = get_molecule(name, device="cuda")

    # COLD
    print(f"\n  --- COLD ---", flush=True)
    np.random.seed(42); torch.manual_seed(42)
    t0 = time.time()
    r_cold = run_hi_nqs_sqd(H, info, config=make_cfg(False, n_samp, top_k, max_iter))
    t_cold = time.time() - t0
    print(f"  COLD: E={r_cold.energy:.10f}, basis={r_cold.diag_dim}, t={t_cold:.1f}s", flush=True)

    # WARM
    print(f"\n  --- WARM ---", flush=True)
    np.random.seed(42); torch.manual_seed(42)
    t0 = time.time()
    r_warm = run_hi_nqs_sqd(H, info, config=make_cfg(True, n_samp, top_k, max_iter))
    t_warm = time.time() - t0
    print(f"  WARM: E={r_warm.energy:.10f}, basis={r_warm.diag_dim}, t={t_warm:.1f}s", flush=True)

    results[name] = {
        "cold": {"E": float(r_cold.energy), "basis": int(r_cold.diag_dim), "t": float(t_cold)},
        "warm": {"E": float(r_warm.energy), "basis": int(r_warm.diag_dim), "t": float(t_warm)},
        "energy_diff_mHa": float(abs(r_cold.energy - r_warm.energy) * 1000),
        "speedup": float(t_cold / t_warm) if t_warm > 0 else None,
    }

# Summary
print(f"\n{'='*70}", flush=True)
print(f"  WARM-START VALIDATION SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'System':<20} {'COLD t':>8} {'WARM t':>8} {'Speedup':>8} {'ΔE (mHa)':>10}", flush=True)
print(f"  {'-'*60}", flush=True)
for name, r in results.items():
    print(f"  {name:<20} {r['cold']['t']:>7.1f}s {r['warm']['t']:>7.1f}s "
          f"{r['speedup']:>7.2f}x {r['energy_diff_mHa']:>9.6f}", flush=True)

with open("warmstart_validation.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to warmstart_validation.json", flush=True)
