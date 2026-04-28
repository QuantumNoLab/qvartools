#!/usr/bin/env python3
"""
N₂ Potential Energy Surface scan for HI-NQS paper.

Bond lengths: 0.8, 1.0, 1.098 (eq), 1.2, 1.5, 2.0, 2.5, 3.0 Å
Methods: HI-NQS, CIPSI, CCSD, CCSD(T), HCI (reference)
Active space: CAS(10,20) cc-pVTZ = 40 qubits

This is the CORE FIGURE of the paper: CCSD(T) fails at stretched bonds,
HI-NQS captures strong correlation across all geometries.
"""
import sys, time, json
import numpy as np
import torch
from math import comb
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hamiltonians.molecular import create_n2_cas_hamiltonian, compute_molecular_integrals
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

# N₂ bond lengths in Angstroms
BOND_LENGTHS = [0.8, 1.0, 1.098, 1.2, 1.5, 2.0, 2.5, 3.0]

# Best HI-NQS config from 40Q benchmark
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
print(f"N₂ PES scan: {len(BOND_LENGTHS)} points, CAS(10,20) cc-pVTZ 40Q", flush=True)
print(f"HI-NQS config: top_k={NQS_CFG.top_k}, n_samples={NQS_CFG.n_samples}, "
      f"entropy={NQS_CFG.entropy_weight}, lr={NQS_CFG.nqs_lr}", flush=True)

# ========== CCSD / CCSD(T) ==========
print(f"\n{'='*70}", flush=True)
print(f"  Phase 1: CCSD / CCSD(T) (all bond lengths)", flush=True)
print(f"{'='*70}", flush=True)

from pyscf import gto, scf, cc

ccsd_results = {}
for R in BOND_LENGTHS:
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvtz", spin=0, verbose=0)
    mf = scf.RHF(mol).run()

    if not mf.converged:
        # Try UHF for stretched bonds
        mf = scf.UHF(mol).run()

    try:
        mycc = cc.CCSD(mf).run()
        e_ccsd = mycc.e_tot
        et = mycc.ccsd_t()
        e_ccsdt = e_ccsd + et
        print(f"  R={R:.3f} Å: CCSD={e_ccsd:.10f}, CCSD(T)={e_ccsdt:.10f}", flush=True)
        ccsd_results[R] = {"ccsd": e_ccsd, "ccsdt": e_ccsdt, "hf": mf.e_tot}
    except Exception as ex:
        print(f"  R={R:.3f} Å: CCSD FAILED — {ex}", flush=True)
        ccsd_results[R] = {"ccsd": None, "ccsdt": None, "hf": mf.e_tot}

# ========== CIPSI ==========
print(f"\n{'='*70}", flush=True)
print(f"  Phase 2: CIPSI (all bond lengths)", flush=True)
print(f"{'='*70}", flush=True)

from src.solvers.sci import run_sci

cipsi_results = {}
for R in BOND_LENGTHS:
    print(f"\n  R={R:.3f} Å:", flush=True)
    t0 = time.time()
    H = create_n2_cas_hamiltonian(bond_length=R, basis="cc-pvtz", cas=(10, 20), device="cpu")
    info = {"n_qubits": 2 * H.n_orbitals}

    r = run_sci(H, info, expansion_size=1000, max_basis=0, max_iterations=200)
    elapsed = time.time() - t0
    print(f"    CIPSI: E={r.energy:.10f}, basis={r.diag_dim}, time={elapsed:.0f}s", flush=True)
    cipsi_results[R] = {"energy": r.energy, "basis": r.diag_dim, "time": elapsed}

# ========== HCI reference ==========
print(f"\n{'='*70}", flush=True)
print(f"  Phase 3: HCI reference (all bond lengths)", flush=True)
print(f"{'='*70}", flush=True)

from pyscf import mcscf
from pyscf.hci import hci as pyscf_hci

hci_results = {}
for R in BOND_LENGTHS:
    print(f"\n  R={R:.3f} Å:", flush=True)
    t0 = time.time()
    mol = gto.M(atom=f"N 0 0 0; N 0 0 {R}", basis="cc-pvtz", spin=0, verbose=0)
    mf = scf.RHF(mol).run()

    ncas, nelecas = 20, 10
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver = pyscf_hci.SCI(mol)
    mc.fcisolver.select_cutoff = 1e-4
    mc.fcisolver.ci_coeff_cutoff = 1e-4
    mc.kernel()

    elapsed = time.time() - t0
    e_hci = mc.e_tot
    print(f"    HCI: E={e_hci:.10f}, time={elapsed:.0f}s", flush=True)
    hci_results[R] = {"energy": e_hci, "time": elapsed}

# ========== HI-NQS ==========
print(f"\n{'='*70}", flush=True)
print(f"  Phase 4: HI-NQS (all bond lengths)", flush=True)
print(f"{'='*70}", flush=True)

nqs_results = {}
for R in BOND_LENGTHS:
    print(f"\n  R={R:.3f} Å:", flush=True)
    np.random.seed(42)
    torch.manual_seed(42)

    t0 = time.time()
    H = create_n2_cas_hamiltonian(bond_length=R, basis="cc-pvtz", cas=(10, 20), device="cuda")
    info = {"n_qubits": 2 * H.n_orbitals}

    r = run_hi_nqs_sqd(H, info, config=NQS_CFG)
    elapsed = time.time() - t0
    print(f"    HI-NQS: E={r.energy:.10f}, basis={r.diag_dim}, time={elapsed:.0f}s, "
          f"converged={r.converged}", flush=True)
    nqs_results[R] = {"energy": r.energy, "basis": r.diag_dim, "time": elapsed,
                      "converged": r.converged}

# ========== SUMMARY ==========
print(f"\n{'='*70}", flush=True)
print(f"  N₂ PES SUMMARY — CAS(10,20) cc-pVTZ 40Q", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'R(Å)':>6} | {'HF':>14} | {'CCSD':>14} | {'CCSD(T)':>14} | "
      f"{'CIPSI':>14} | {'HI-NQS':>14} | {'HCI':>14}", flush=True)
print(f"  {'-'*100}", flush=True)

for R in BOND_LENGTHS:
    hf_e = ccsd_results.get(R, {}).get("hf", None)
    cc_e = ccsd_results.get(R, {}).get("ccsd", None)
    cct_e = ccsd_results.get(R, {}).get("ccsdt", None)
    ci_e = cipsi_results.get(R, {}).get("energy", None)
    nqs_e = nqs_results.get(R, {}).get("energy", None)
    hci_e = hci_results.get(R, {}).get("energy", None)

    def fmt(e):
        return f"{e:.10f}" if e is not None else "FAIL"

    print(f"  {R:>6.3f} | {fmt(hf_e):>14} | {fmt(cc_e):>14} | {fmt(cct_e):>14} | "
          f"{fmt(ci_e):>14} | {fmt(nqs_e):>14} | {fmt(hci_e):>14}", flush=True)

# Save JSON for plotting
results_json = {
    "system": "N2-CAS(10,20)",
    "basis": "cc-pVTZ",
    "qubits": 40,
    "bond_lengths": BOND_LENGTHS,
    "ccsd": ccsd_results,
    "cipsi": cipsi_results,
    "hci": hci_results,
    "nqs": nqs_results,
}
with open("n2_pes_results.json", "w") as f:
    json.dump(results_json, f, indent=2, default=str)
print(f"\nResults saved to n2_pes_results.json", flush=True)
