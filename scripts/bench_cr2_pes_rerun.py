#!/usr/bin/env python3
"""
Cr2 PES scan with HI-NQS on CAS(12,20) cc-pVDZ (40Q).

Cr2 has a formal sextuple bond and is the archetype strong-correlation
molecule. CCSD/CCSD(T) fails, CASSCF is essential, DMRG is the gold standard.

PES scan at 7 bond lengths:
  1.50 (compressed)
  1.68 (equilibrium)
  1.80
  2.00
  2.40
  2.80
  3.50 (dissociated)

CCSD/CCSD(T) PES already exists in bench_cr2_pes_162981.log for R ∈
{1.5, 1.68, 1.8, 2.0}. This run fills in HI-NQS for all 7 points.
"""
import sys, time, json
from pathlib import Path
from math import comb

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hamiltonians.molecular import create_cr2_hamiltonian
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

RESULTS_PATH = Path(__file__).parent.parent / "results" / "rerun_cr2_pes.json"

BOND_LENGTHS = [1.50, 1.68, 1.80, 2.00, 2.40, 2.80, 3.50]
CAS = (12, 20)      # 10 α + 10 β electrons... wait, CAS(12,20) = 12 electrons, 20 orbitals
BASIS = "cc-pvdz"

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
      flush=True)
print(f"\nCr2 PES scan — HI-NQS on CAS{CAS} {BASIS} ({2*CAS[1]}Q)", flush=True)
print(f"Bond lengths: {BOND_LENGTHS}", flush=True)

# Resume support
all_results = []
done_R = set()
if RESULTS_PATH.exists():
    try:
        with open(RESULTS_PATH) as f:
            all_results = json.load(f).get("runs", [])
        done_R = {r["bond_length"] for r in all_results if "energy" in r}
        print(f"\n[Resume] Already done: {sorted(done_R)}", flush=True)
    except Exception:
        pass

# ----------------------------------------------------------------------------
for R in BOND_LENGTHS:
    if R in done_R:
        print(f"\n[Skip] R={R:.2f} Å (already in results)", flush=True)
        continue

    print(f"\n{'='*70}", flush=True)
    print(f"  Cr2 at R={R:.2f} Å,  CAS{CAS}  {BASIS}  40Q", flush=True)
    print(f"{'='*70}", flush=True)

    try:
        t0 = time.time()
        H = create_cr2_hamiltonian(bond_length=R, basis=BASIS, cas=CAS, device="cuda")
        t_build = time.time() - t0
        hilbert = comb(CAS[1], CAS[0]//2) ** 2
        print(f"  Hamiltonian built in {t_build:.0f}s, Hilbert={hilbert:,}", flush=True)
    except Exception as ex:
        print(f"  [ERROR] Hamiltonian build failed: {ex}", flush=True)
        import traceback; traceback.print_exc()
        all_results.append({
            "bond_length": R, "error": f"build: {ex}",
        })
        continue

    info = {"n_qubits": 2 * H.n_orbitals}

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=100000,
        top_k=10000,
        max_basis_size=0,
        max_iterations=40,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
        warm_start=True,
        use_incremental_sqd=True,
    )

    t1 = time.time()
    try:
        r = run_hi_nqs_sqd(H, info, config=cfg)
        elapsed = time.time() - t1
        total_time = time.time() - t0
        print(f"\n  R={R:.2f}: E={r.energy:.10f} Ha, basis={r.diag_dim:,}, "
              f"HINQS={elapsed:.0f}s, total={total_time:.0f}s "
              f"({total_time/3600:.2f}h), converged={r.converged}", flush=True)
        all_results.append({
            "bond_length": R,
            "n_qubits": 2 * H.n_orbitals,
            "n_orbitals": H.n_orbitals,
            "n_alpha": H.n_alpha,
            "n_beta": H.n_beta,
            "cas": list(CAS),
            "basis_set": BASIS,
            "hilbert": hilbert,
            "n_samples": cfg.n_samples,
            "top_k": cfg.top_k,
            "energy": float(r.energy) if r.energy is not None else None,
            "diag_dim": int(r.diag_dim),
            "hinqs_time_s": float(elapsed),
            "total_time_s": float(total_time),
            "converged": bool(r.converged),
        })
    except Exception as ex:
        elapsed = time.time() - t1
        print(f"  [ERROR] R={R:.2f} HI-NQS failed after {elapsed:.0f}s: {ex}",
              flush=True)
        import traceback; traceback.print_exc()
        all_results.append({
            "bond_length": R,
            "error": str(ex),
            "hinqs_time_s": float(elapsed),
        })

    # Stream-save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "system": "Cr2",
            "method": "HI+NQS+SQD (incremental, warm-start)",
            "cas": list(CAS),
            "basis_set": BASIS,
            "seed": 42,
            "bond_lengths_plan": BOND_LENGTHS,
            "runs": all_results,
        }, f, indent=2)

# ----------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print(f"  Cr2 PES HI-NQS SUMMARY  —  CAS{CAS} {BASIS} 40Q", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'R (Å)':>6} {'Energy (Ha)':>18} {'Basis':>10} {'Time (s)':>10} {'Conv':>6}",
      flush=True)
print(f"  {'-'*62}", flush=True)
for r in all_results:
    if "error" in r:
        print(f"  {r['bond_length']:>6.2f} {'[ERROR]':>18} {'-':>10} "
              f"{r.get('hinqs_time_s', 0):>9.0f}s {'N':>6}", flush=True)
    else:
        conv = "Y" if r.get("converged") else "N"
        print(f"  {r['bond_length']:>6.2f} {r['energy']:>18.10f} "
              f"{r['diag_dim']:>10,} {r['hinqs_time_s']:>9.0f}s {conv:>6}",
              flush=True)

print(f"\n  Saved to {RESULTS_PATH}", flush=True)
