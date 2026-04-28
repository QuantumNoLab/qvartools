#!/usr/bin/env python3
"""
Compare two HI-NQS basis-management strategies on identical hyperparameters:

  1. EVICT  (current default): Iter 1 rescores everything and may drop existing
            configs; max_basis_size enforces a hard cap; Davidson warm-start is
            partially defeated by string churn.
  2. MONO   (monotonic_basis=True): never evict, never rescore. Iter 0 uses a
            smaller cold-start (top_k // 4). Davidson warm-start is preserved
            because new ci_strs ⊇ old ci_strs.

Run on three molecules of increasing difficulty:
  - NH3 (16Q):  baseline sanity check
  - C2H2 (24Q): real per-iter SQD cost shows up
  - N2-CAS(10,12) (24Q cc-pVTZ): another 24Q point with different structure

Same seed (42), same n_samples / top_k / nqs_steps so the only variable is the
basis management strategy.
"""
import sys, time, json
from pathlib import Path
from math import comb

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

RESULTS_PATH = Path(__file__).parent.parent / "results" / "compare_monotonic.json"

# ----------------------------------------------------------------------------
# Test matrix
# ----------------------------------------------------------------------------
TESTS = [
    # (molecule_name, n_samples, top_k, max_iter)
    ("NH3",            10000, 2000, 25),
    ("C2H2",           20000, 4000, 25),
    ("N2-CAS(10,12)",  20000, 4000, 25),
]

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
      flush=True)
print(f"\n{'='*80}", flush=True)
print(f"  Comparing EVICT vs MONOTONIC basis-management on 3 molecules", flush=True)
print(f"{'='*80}", flush=True)

all_results = []

for mol_name, n_samples, top_k, max_iter in TESTS:
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    nq = info["n_qubits"]
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)

    print(f"\n{'#'*80}", flush=True)
    print(f"  {mol_name}  ({nq}Q, Hilbert={hilbert:,})", flush=True)
    print(f"  Config: n_samples={n_samples:,}, top_k={top_k:,}, max_iter={max_iter}",
          flush=True)
    print(f"{'#'*80}", flush=True)

    pair_result = {
        "molecule": mol_name,
        "n_qubits": nq,
        "n_orbitals": n_orb,
        "hilbert": hilbert,
        "config": {"n_samples": n_samples, "top_k": top_k, "max_iter": max_iter},
        "modes": {},
    }

    for mode_name, mono_flag in [("evict", False), ("monotonic", True)]:
        print(f"\n--- {mol_name}  mode={mode_name.upper()} ---", flush=True)

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
            monotonic_basis=mono_flag,
        )

        t0 = time.time()
        try:
            r = run_hi_nqs_sqd(H, info, config=cfg)
            elapsed = time.time() - t0
            print(f"  → E={r.energy:.10f}  basis={r.diag_dim:,}  "
                  f"time={elapsed:.1f}s  converged={r.converged}", flush=True)
            pair_result["modes"][mode_name] = {
                "energy": float(r.energy) if r.energy is not None else None,
                "diag_dim": int(r.diag_dim),
                "wall_time_s": float(elapsed),
                "converged": bool(r.converged),
            }
        except Exception as ex:
            elapsed = time.time() - t0
            print(f"  [ERROR] {mode_name} failed after {elapsed:.0f}s: {ex}",
                  flush=True)
            import traceback; traceback.print_exc()
            pair_result["modes"][mode_name] = {
                "error": str(ex),
                "wall_time_s": float(elapsed),
            }

    all_results.append(pair_result)

    # Stream-save after each molecule
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"comparisons": all_results}, f, indent=2)

# ----------------------------------------------------------------------------
# Final comparison table
# ----------------------------------------------------------------------------
print(f"\n{'='*80}", flush=True)
print(f"  EVICT vs MONOTONIC SUMMARY", flush=True)
print(f"{'='*80}", flush=True)
print(f"\n  {'Molecule':<18} {'Mode':<10} {'Energy':>18} {'Basis':>10} "
      f"{'Time (s)':>10} {'Conv':>5}", flush=True)
print(f"  {'-'*78}", flush=True)

for pr in all_results:
    for mode in ("evict", "monotonic"):
        m = pr["modes"].get(mode, {})
        if "error" in m:
            print(f"  {pr['molecule']:<18} {mode:<10} {'[ERROR]':>18} "
                  f"{'-':>10} {m.get('wall_time_s', 0):>9.1f}s {'N':>5}", flush=True)
        else:
            conv = "Y" if m.get("converged") else "N"
            e = m.get("energy", float("nan"))
            print(f"  {pr['molecule']:<18} {mode:<10} {e:>18.10f} "
                  f"{m.get('diag_dim', 0):>10,} {m.get('wall_time_s', 0):>9.1f}s "
                  f"{conv:>5}", flush=True)
    # Per-molecule delta
    em = pr["modes"].get("evict", {})
    mm = pr["modes"].get("monotonic", {})
    if "energy" in em and "energy" in mm and em["energy"] and mm["energy"]:
        de_mha = (mm["energy"] - em["energy"]) * 1000
        speedup = em["wall_time_s"] / mm["wall_time_s"] if mm["wall_time_s"] > 0 else 0
        basis_ratio = mm["diag_dim"] / em["diag_dim"] if em["diag_dim"] > 0 else 0
        print(f"    Δ:     mono−evict   ΔE={de_mha:+.4e} mHa   "
              f"speedup(evict/mono)={speedup:.2f}x   "
              f"basis ratio(mono/evict)={basis_ratio:.2f}x", flush=True)

print(f"\n  Saved to {RESULTS_PATH}", flush=True)
