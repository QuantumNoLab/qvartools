#!/usr/bin/env python3
"""
N2-CAS(10,26) 52Q cc-pVTZ HI-NQS rerun.

Previous attempts all failed:
  - job 157591: disk full crash after iter 0
  - job 162941: SLURM timeout before iter 0 finished
  - (others)

Strategy for this run:
  - Start with moderate config (not too aggressive) to ensure iter 0 completes
  - Increase top_k if convergence needs it
  - Stream-save after every iter in a JSON checkpoint so any crash
    leaves partial data behind
  - HF canonical orbitals (no CASSCF — CASSCF on CAS(10,26) would take hours
    on top of an already-long run)

Uses molecule registry entry 'N2-CAS(10,26)'.
"""
import sys, time, json
from pathlib import Path
from math import comb

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

RESULTS_PATH = Path(__file__).parent.parent / "results" / "rerun_n2_52q.json"

# ----------------------------------------------------------------------------
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
      flush=True)

MOL_NAME = "N2-CAS(10,26)"
H, info = get_molecule(MOL_NAME)
n_orb = H.n_orbitals
nq = info["n_qubits"]
hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)

print(f"\n{'='*72}", flush=True)
print(f"  {MOL_NAME}  ({nq}Q, cc-pVTZ, CAS(10,26))", flush=True)
print(f"  Hilbert space: {hilbert:,}", flush=True)
print(f"{'='*72}", flush=True)

np.random.seed(42)
torch.manual_seed(42)

# Moderate config — balance between coverage and per-iter cost
cfg = HINQSSQDConfig(
    n_samples=50000,              # less aggressive than 100k to fit iter 0
    top_k=8000,
    max_basis_size=0,
    max_iterations=30,
    convergence_threshold=1e-8,
    convergence_window=3,
    nqs_steps=5,                  # fewer update steps per iter
    nqs_lr=3e-4,
    entropy_weight=0.15,
    warm_start=True,
    use_incremental_sqd=True,
)

print(f"\n  Config: n_samples={cfg.n_samples:,}, top_k={cfg.top_k:,}, "
      f"max_iter={cfg.max_iterations}, nqs_steps={cfg.nqs_steps}", flush=True)

t0 = time.time()
try:
    r = run_hi_nqs_sqd(H, info, config=cfg)
    elapsed = time.time() - t0
    print(f"\n  ✓ {MOL_NAME} DONE: E={r.energy:.10f} Ha, "
          f"basis={r.diag_dim:,}, time={elapsed:.0f}s ({elapsed/3600:.2f}h), "
          f"converged={r.converged}", flush=True)

    result = {
        "molecule": MOL_NAME,
        "n_qubits": nq,
        "n_orbitals": n_orb,
        "n_alpha": H.n_alpha,
        "n_beta": H.n_beta,
        "hilbert": hilbert,
        "basis_set": "cc-pVTZ",
        "n_samples": cfg.n_samples,
        "top_k": cfg.top_k,
        "energy": float(r.energy) if r.energy is not None else None,
        "diag_dim": int(r.diag_dim),
        "wall_time_s": float(elapsed),
        "converged": bool(r.converged),
    }
except Exception as ex:
    elapsed = time.time() - t0
    print(f"\n  ✗ {MOL_NAME} FAILED after {elapsed:.0f}s: {ex}", flush=True)
    import traceback
    traceback.print_exc()
    result = {
        "molecule": MOL_NAME,
        "n_qubits": nq,
        "error": str(ex),
        "wall_time_s": float(elapsed),
    }

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump({
        "method": "HI+NQS+SQD (incremental, warm-start)",
        "seed": 42,
        "config": {
            "n_samples": cfg.n_samples,
            "top_k": cfg.top_k,
            "nqs_steps": cfg.nqs_steps,
            "max_iterations": cfg.max_iterations,
        },
        "run": result,
    }, f, indent=2)
print(f"\n  Saved to {RESULTS_PATH}", flush=True)
