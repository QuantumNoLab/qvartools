#!/usr/bin/env python3
"""
N2-CAS(10,20) 40Q and N2-CAS(10,26) 52Q: HI-NQS v3 parameter sweep.

top_k: 10000, 15000
n_samples: 20000, 30000
→ 4 configs × 2 systems = 8 runs
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

CONFIGS = [
    ("A", 10000, 20000),
    ("B", 10000, 30000),
    ("C", 15000, 20000),
    ("D", 15000, 30000),
]

SYSTEMS = [
    ("N2-CAS(10,20)", "40Q"),
    ("N2-CAS(10,26)", "52Q"),
]

# References from previous runs
REFS = {
    "N2-CAS(10,20)": -109.2138472581,  # CIPSI basis=53998
    "N2-CAS(10,26)": None,              # no reference yet
}

all_results = []

for mol_name, qlabel in SYSTEMS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {mol_name} ({qlabel})", flush=True)
    print(f"{'='*70}", flush=True)

    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    nq = info["n_qubits"]
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f"  {nq}Q, {n_orb} orb, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}", flush=True)

    ref = REFS.get(mol_name)

    for tag, top_k, n_samples in CONFIGS:
        print(f"\n  Config {tag}: top_k={top_k}, n_samples={n_samples}", flush=True)

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

        err_str = ""
        if ref and r.energy:
            err = (r.energy - ref) * 1000
            err_str = f", err vs CIPSI={err:+.4f} mHa"

        print(f"\n  Config {tag}: E={r.energy:.10f} Ha, basis={r.diag_dim}, "
              f"time={elapsed:.0f}s ({elapsed/3600:.2f}h), converged={r.converged}"
              f"{err_str}", flush=True)

        all_results.append(dict(
            mol=mol_name, qlabel=qlabel, tag=tag,
            top_k=top_k, n_samples=n_samples,
            E=r.energy, basis=r.diag_dim,
            time=elapsed, converged=r.converged,
        ))

# Summary
print(f"\n{'='*70}", flush=True)
print(f"  PARAMETER SWEEP SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Mol':<18} {'Cfg':>3} {'top_k':>6} {'samp':>6} {'Energy':>18} {'basis':>7} {'time':>7} {'conv':>5}", flush=True)
print(f"  {'-'*80}", flush=True)
for r in all_results:
    e_str = f"{r['E']:.10f}" if r['E'] else "FAIL"
    print(f"  {r['mol']:<18} {r['tag']:>3} {r['top_k']:>6} {r['n_samples']:>6} "
          f"{e_str:>18} {r['basis']:>7} {r['time']:>6.0f}s {'Y' if r['converged'] else 'N':>5}", flush=True)
