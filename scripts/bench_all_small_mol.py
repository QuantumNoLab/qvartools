#!/usr/bin/env python3
"""
HI-NQS rerun across all small-to-medium molecules (≤36Q).

Skips N2-CAS(10,20) 40Q (already running in job 164762).

Config rule: n_samples / top_k scale with system size.
Single seed (42). Stream-saves to results/rerun_small_mol.json after each mol.
"""
import sys, time, json
from pathlib import Path
from math import comb

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

# ----------------------------------------------------------------------------
# Per-molecule configs. Ordered from fastest to slowest.
# ----------------------------------------------------------------------------
CONFIGS = [
    # (molecule_name, n_samples, top_k, max_iter)
    ("LiH",            5000,  1000,  30),
    ("H2O",            5000,  1000,  30),
    ("BeH2",           5000,  1500,  30),
    ("NH3",           10000,  2000,  30),
    ("CH4",           10000,  3000,  30),
    ("N2",            10000,  3000,  30),
    ("HCN",           15000,  4000,  30),
    ("C2H2",          20000,  4000,  30),
    ("N2-CAS(10,12)", 20000,  4000,  30),
    ("Cr2",           20000,  4000,  30),
    ("H2S",           15000,  3000,  30),
    ("C2H4",          30000,  6000,  40),
    ("Benzene",       30000,  6000,  40),
    ("Cr2-CAS(12,18)", 40000, 8000,  40),
]

RESULTS_PATH = Path(__file__).parent.parent / "results" / "rerun_small_mol.json"

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
      flush=True)
print(f"Total molecules to run: {len(CONFIGS)}", flush=True)
print(f"Stream-saving to: {RESULTS_PATH}", flush=True)

all_results = []

# Load existing (allows resumption if job is re-submitted)
if RESULTS_PATH.exists():
    try:
        with open(RESULTS_PATH) as f:
            all_results = json.load(f).get("runs", [])
        done_mols = {r["molecule"] for r in all_results}
        print(f"\n[Resume] Found {len(done_mols)} already-completed molecules: "
              f"{sorted(done_mols)}", flush=True)
    except Exception:
        all_results = []
        done_mols = set()
else:
    done_mols = set()

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
for mol_name, n_samples, top_k, max_iter in CONFIGS:
    if mol_name in done_mols:
        print(f"\n[Skip] {mol_name} (already in results)", flush=True)
        continue

    print(f"\n{'='*70}", flush=True)
    print(f"  {mol_name}  (n_samples={n_samples:,}, top_k={top_k:,})", flush=True)
    print(f"{'='*70}", flush=True)

    try:
        H, info = get_molecule(mol_name)
    except Exception as ex:
        print(f"  [ERROR] Failed to load {mol_name}: {ex}", flush=True)
        all_results.append({
            "molecule": mol_name, "error": f"load: {ex}",
        })
        continue

    n_orb = H.n_orbitals
    nq = info.get("n_qubits", 2 * n_orb)
    try:
        hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    except Exception:
        hilbert = None

    print(f"  {nq}Q, {n_orb} orb, ({H.n_alpha},{H.n_beta})e, "
          f"Hilbert={hilbert:,}" if hilbert else f"  {nq}Q, {n_orb} orb",
          flush=True)

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=0,
        max_iterations=max_iter,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
        warm_start=True,
        use_incremental_sqd=True,
    )

    t0 = time.time()
    try:
        r = run_hi_nqs_sqd(H, info, config=cfg)
        elapsed = time.time() - t0
        print(f"\n  {mol_name}: E={r.energy:.10f} Ha, basis={r.diag_dim:,}, "
              f"time={elapsed:.1f}s, converged={r.converged}", flush=True)
        all_results.append({
            "molecule": mol_name,
            "n_qubits": nq,
            "n_orbitals": n_orb,
            "n_alpha": H.n_alpha,
            "n_beta": H.n_beta,
            "hilbert": hilbert,
            "basis_set": getattr(H, "basis", None) or "unknown",
            "n_samples": n_samples,
            "top_k": top_k,
            "energy": float(r.energy) if r.energy is not None else None,
            "diag_dim": int(r.diag_dim),
            "wall_time_s": float(elapsed),
            "converged": bool(r.converged),
        })
    except Exception as ex:
        elapsed = time.time() - t0
        print(f"  [ERROR] {mol_name} failed after {elapsed:.0f}s: {ex}", flush=True)
        all_results.append({
            "molecule": mol_name,
            "n_qubits": nq if "nq" in dir() else None,
            "error": str(ex),
            "wall_time_s": float(elapsed),
        })

    # Stream-save after every molecule
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "method": "HI+NQS+SQD (incremental backend, warm-start)",
            "seed": 42,
            "runs": all_results,
        }, f, indent=2)

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print(f"  ALL SMALL-MOLECULE HI-NQS RERUN — SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Molecule':<18} {'Q':>3} {'Energy (Ha)':>18} {'Basis':>10} "
      f"{'Time (s)':>10} {'Conv':>6}", flush=True)
print(f"  {'-'*72}", flush=True)

for r in all_results:
    if "error" in r:
        print(f"  {r['molecule']:<18} {'?':>3} {'[ERROR]':>18} "
              f"{'-':>10} {r.get('wall_time_s', 0):>9.1f}s {'N':>6}", flush=True)
    else:
        conv = "Y" if r.get("converged") else "N"
        print(f"  {r['molecule']:<18} {r['n_qubits']:>3} "
              f"{r['energy']:>18.10f} {r['diag_dim']:>10,} "
              f"{r['wall_time_s']:>9.1f}s {conv:>6}", flush=True)

print(f"\n  Saved to {RESULTS_PATH}", flush=True)
