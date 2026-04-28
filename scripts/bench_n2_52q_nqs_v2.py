#!/usr/bin/env python3
"""
N2-CAS(10,26) 52Q ONLY: HI-NQS v3 with GPU PT2 + tuned hyperparams.

Configs:
  E: n_samples=30K,  top_k=10K  (ratio=3x)
  F: n_samples=45K,  top_k=15K  (ratio=3x)
  G: n_samples=100K, top_k=10K  (ratio=10x)
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
    ("E", 10000,  30000),   # ratio=3x
    ("F", 15000,  45000),   # ratio=3x
    ("G", 10000, 100000),   # ratio=10x
]

REFS = {
    "N2-CAS(10,26)": None,
}

all_results = []

mol_name = "N2-CAS(10,26)"
qlabel = "52Q"

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
    print(f"\n  Config {tag}: top_k={top_k}, n_samples={n_samples} (ratio={n_samples//top_k}x)", flush=True)

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=0,
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
    )

    t0 = time.time()
    r = run_hi_nqs_sqd(H, info, config=cfg)
    elapsed = time.time() - t0

    err_str = ""
    if ref and r.energy:
        err = (r.energy - ref) * 1000
        err_str = f", err vs ref={err:+.4f} mHa"

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
print(f"  52Q PARAMETER SWEEP SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Cfg':>3} {'top_k':>6} {'samp':>6} {'ratio':>5} {'Energy':>18} {'basis':>7} {'time':>7} {'conv':>5}", flush=True)
print(f"  {'-'*75}", flush=True)
for r in all_results:
    e_str = f"{r['E']:.10f}" if r['E'] else "FAIL"
    ratio = r['n_samples'] // r['top_k']
    print(f"  {r['tag']:>3} {r['top_k']:>6} {r['n_samples']:>6} {ratio:>4}x "
          f"{e_str:>18} {r['basis']:>7} {r['time']:>6.0f}s {'Y' if r['converged'] else 'N':>5}", flush=True)
