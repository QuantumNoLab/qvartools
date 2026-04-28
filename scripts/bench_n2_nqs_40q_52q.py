#!/usr/bin/env python3
"""
N2-CAS(10,20) 40Q and N2-CAS(10,26) 52Q: HI-NQS v3 only.

SCI is running separately in job 157401.
This script runs HI-NQS v3 with parameters from bench_n2_large.py.
"""
import sys, time
import numpy as np
import torch
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# (mol_name, top_k, n_samples)
MOLECULES = [
    ("N2-CAS(10,20)", 2000, 10000),   # 40Q
    ("N2-CAS(10,26)", 4000, 20000),   # 52Q
]

results = []

for mol_name, top_k, n_samples in MOLECULES:
    print(f"\n{'='*70}", flush=True)
    print(f"  {mol_name}  NQS v3 (top_k={top_k}, n_samples={n_samples})", flush=True)
    print(f"{'='*70}", flush=True)

    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    nq = info["n_qubits"]
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f"  {nq}Q, {n_orb} orb, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}", flush=True)

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=0,
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=10,
    )

    t0 = time.time()
    r = run_hi_nqs_sqd(H, info, config=cfg)
    elapsed = time.time() - t0

    print(f"\n  {mol_name} NQS v3: E={r.energy:.10f} Ha, "
          f"basis={r.diag_dim}, time={elapsed:.0f}s ({elapsed/3600:.2f}h), "
          f"converged={r.converged}", flush=True)

    results.append(dict(mol=mol_name, nq=nq, E=r.energy,
                        basis=r.diag_dim, time=elapsed, converged=r.converged))

print(f"\n{'='*70}", flush=True)
print(f"  N2 40Q/52Q HI-NQS v3 SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Mol':<20} {'Q':>3}  {'Energy':>18}  {'basis':>8}  {'time':>8}  {'conv':>5}", flush=True)
for r in results:
    print(f"  {r['mol']:<20} {r['nq']:>3}  {r['E']:>18.10f}  {r['basis']:>8,}  "
          f"{r['time']:>7.0f}s  {'Y' if r['converged'] else 'N':>5}", flush=True)
