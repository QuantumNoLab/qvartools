#!/usr/bin/env python3
"""
Cr₂ Potential Energy Surface scan — the ultimate strong correlation benchmark.

Cr₂ has a formal sextuple bond (3d-3d), one of the hardest molecules for
classical quantum chemistry. CCSD fails completely.

Bond lengths: 1.5, 1.68 (eq), 1.8, 2.0, 2.4, 2.8, 3.5 Å
Active spaces:
  - CAS(12,20) = 40 qubits (main)
  - CAS(12,26) = 52 qubits (scaling test)
Methods: HI-NQS, CIPSI, CCSD, CCSD(T)
"""
import sys, time, json
import numpy as np
import torch
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hamiltonians.molecular import create_cr2_hamiltonian
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

# Cr₂ bond lengths
BOND_LENGTHS = [1.5, 1.68, 1.8, 2.0, 2.4, 2.8, 3.5]

# Active spaces to test
CAS_CONFIGS = [
    ((12, 20), "cc-pvdz", 40),   # CAS(12,20) = 40Q
]

NQS_CFG = HINQSSQDConfig(
    n_samples=100000,
    top_k=10000,
    max_basis_size=0,
    max_iterations=50,
    convergence_threshold=1e-8,
    convergence_window=3,
    nqs_steps=7,
    nqs_lr=3e-4,
    entropy_weight=0.15,
)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
print(f"Cr₂ PES scan: {len(BOND_LENGTHS)} points", flush=True)

# ========== CCSD / CCSD(T) on full space ==========
print(f"\n{'='*70}", flush=True)
print(f"  Phase 1: CCSD / CCSD(T) — Cr₂ full space", flush=True)
print(f"{'='*70}", flush=True)

from pyscf import gto, scf, cc, dft

ccsd_results = {}
for R in BOND_LENGTHS:
    print(f"\n  R={R:.2f} Å:", flush=True)
    try:
        mol = gto.M(atom=f"Cr 0 0 0; Cr 0 0 {R}", basis="cc-pvdz",
                     spin=0, verbose=0, symmetry=False)
        mf = scf.RHF(mol)
        mf.max_cycle = 200
        mf.kernel()

        if not mf.converged:
            mf = scf.UHF(mol)
            mf.max_cycle = 200
            mf.kernel()

        try:
            mycc = cc.CCSD(mf)
            mycc.max_cycle = 200
            mycc.kernel()
            e_ccsd = mycc.e_tot if mycc.converged else None
            if mycc.converged:
                et = mycc.ccsd_t()
                e_ccsdt = e_ccsd + et
            else:
                e_ccsdt = None
            print(f"    CCSD={e_ccsd}, CCSD(T)={e_ccsdt}", flush=True)
        except Exception as ex:
            print(f"    CCSD FAILED: {ex}", flush=True)
            e_ccsd, e_ccsdt = None, None

        ccsd_results[R] = {"ccsd": e_ccsd, "ccsdt": e_ccsdt, "hf": mf.e_tot}
    except Exception as ex:
        print(f"    SCF FAILED: {ex}", flush=True)
        ccsd_results[R] = {"ccsd": None, "ccsdt": None, "hf": None}

# ========== CIPSI in CAS space ==========
print(f"\n{'='*70}", flush=True)
print(f"  Phase 2: CIPSI — Cr₂ CAS(12,20) cc-pVDZ 40Q", flush=True)
print(f"{'='*70}", flush=True)

from src.solvers.sci import run_sci

cipsi_results = {}
for R in BOND_LENGTHS:
    print(f"\n  R={R:.2f} Å:", flush=True)
    try:
        t0 = time.time()
        H = create_cr2_hamiltonian(bond_length=R, basis="cc-pvdz",
                                   cas=(12, 20), device="cpu")
        info = {"n_qubits": 2 * H.n_orbitals}
        r = run_sci(H, info, expansion_size=1000, max_basis=0, max_iterations=200)
        elapsed = time.time() - t0
        print(f"    CIPSI: E={r.energy:.10f}, basis={r.diag_dim}, time={elapsed:.0f}s", flush=True)
        cipsi_results[R] = {"energy": r.energy, "basis": r.diag_dim, "time": elapsed}
    except Exception as ex:
        print(f"    CIPSI FAILED: {ex}", flush=True)
        cipsi_results[R] = {"energy": None, "basis": None, "time": None}

# ========== HI-NQS ==========
print(f"\n{'='*70}", flush=True)
print(f"  Phase 3: HI-NQS — Cr₂ CAS(12,20) cc-pVDZ 40Q", flush=True)
print(f"{'='*70}", flush=True)

nqs_results = {}
for R in BOND_LENGTHS:
    print(f"\n  R={R:.2f} Å:", flush=True)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        t0 = time.time()
        H = create_cr2_hamiltonian(bond_length=R, basis="cc-pvdz",
                                   cas=(12, 20), device="cuda")
        info = {"n_qubits": 2 * H.n_orbitals}
        r = run_hi_nqs_sqd(H, info, config=NQS_CFG)
        elapsed = time.time() - t0
        print(f"    HI-NQS: E={r.energy:.10f}, basis={r.diag_dim}, time={elapsed:.0f}s, "
              f"converged={r.converged}", flush=True)
        nqs_results[R] = {"energy": r.energy, "basis": r.diag_dim, "time": elapsed,
                          "converged": r.converged}
    except Exception as ex:
        print(f"    HI-NQS FAILED: {ex}", flush=True)
        nqs_results[R] = {"energy": None, "basis": None, "time": None, "converged": False}

# ========== SUMMARY ==========
print(f"\n{'='*70}", flush=True)
print(f"  Cr₂ PES SUMMARY — CAS(12,20) cc-pVDZ 40Q", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'R(Å)':>6} | {'HF':>14} | {'CCSD':>14} | {'CCSD(T)':>14} | "
      f"{'CIPSI':>14} | {'HI-NQS':>14}", flush=True)
print(f"  {'-'*90}", flush=True)

for R in BOND_LENGTHS:
    hf_e = ccsd_results.get(R, {}).get("hf", None)
    cc_e = ccsd_results.get(R, {}).get("ccsd", None)
    cct_e = ccsd_results.get(R, {}).get("ccsdt", None)
    ci_e = cipsi_results.get(R, {}).get("energy", None)
    nqs_e = nqs_results.get(R, {}).get("energy", None)

    def fmt(e):
        return f"{e:.10f}" if e is not None else "FAIL"

    print(f"  {R:>6.2f} | {fmt(hf_e):>14} | {fmt(cc_e):>14} | {fmt(cct_e):>14} | "
          f"{fmt(ci_e):>14} | {fmt(nqs_e):>14}", flush=True)

with open("cr2_pes_results.json", "w") as f:
    json.dump({"bond_lengths": BOND_LENGTHS, "ccsd": ccsd_results,
               "cipsi": cipsi_results, "nqs": nqs_results}, f, indent=2, default=str)
print(f"\nResults saved to cr2_pes_results.json", flush=True)
