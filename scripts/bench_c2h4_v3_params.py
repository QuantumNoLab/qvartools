#!/usr/bin/env python3
"""
C2H4 v3 parameter sweep: find the right top_k / n_samples for 28Q.

Analysis showed v3 with top_k=2000 stagnates at basis~2400 → 2.3 mHa error.
Old IBM version with basis=10000, samples=20000 got 0.06 mHa.
Goal: find minimum top_k/samples that achieves chemical accuracy (<1.6 mHa).

Configs tested:
  A. top_k=2000,  n_samples=5000   (baseline v3, for reference)
  B. top_k=5000,  n_samples=10000  (moderate increase)
  C. top_k=10000, n_samples=20000  (match old version basis size)
  D. top_k=5000,  n_samples=20000  (more samples, moderate basis)
"""
import sys, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

H, info = get_molecule("C2H4")
print(f"C2H4: 28Q, 14 orb, ({H.n_alpha},{H.n_beta})e, Hilbert=9,018,009", flush=True)

# SCI reference from previous runs
SCI_REF = -77.2351408123  # basis=10000 from tier6 log

CONFIGS = [
    ("A-baseline",  2000,  5000),
    ("B-moderate",  5000, 10000),
    ("C-full",     10000, 20000),
    ("D-balanced",  5000, 20000),
]

results = []

for label, top_k, n_samples in CONFIGS:
    print(f"\n{'='*70}", flush=True)
    print(f"  Config {label}: top_k={top_k}, n_samples={n_samples}", flush=True)
    print(f"{'='*70}", flush=True)

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=0,
        max_iterations=30,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=10,
    )

    t0 = time.time()
    r = run_hi_nqs_sqd(H, info, config=cfg)
    elapsed = time.time() - t0

    err = (r.energy - SCI_REF) * 1000 if r.energy else None
    chem_acc = abs(err) < 1.6 if err is not None else False

    print(f"\n  Config {label}: E={r.energy:.10f} Ha, "
          f"err={err:+.4f} mHa, basis={r.diag_dim}, "
          f"time={elapsed:.0f}s, chem_acc={'YES' if chem_acc else 'NO'}", flush=True)

    results.append(dict(
        label=label, top_k=top_k, n_samples=n_samples,
        energy=r.energy, err=err, basis=r.diag_dim,
        time=elapsed, chem_acc=chem_acc, converged=r.converged,
    ))

print(f"\n{'='*70}", flush=True)
print(f"  C2H4 v3 PARAMETER SWEEP SUMMARY", flush=True)
print(f"  SCI reference: {SCI_REF:.10f} Ha (basis=10000)", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Config':<12} {'top_k':>7} {'samples':>8} {'err(mHa)':>10} "
      f"{'basis':>7} {'time':>7} {'<1.6mHa':>8}", flush=True)
print(f"  {'-'*65}", flush=True)
for res in results:
    e = f"{res['err']:+.4f}" if res['err'] is not None else "FAIL"
    acc = "YES ✓" if res['chem_acc'] else "NO"
    print(f"  {res['label']:<12} {res['top_k']:>7} {res['n_samples']:>8} "
          f"{e:>10} {res['basis']:>7} {res['time']:>6.0f}s {acc:>8}", flush=True)
