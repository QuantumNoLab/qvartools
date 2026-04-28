#!/usr/bin/env python3
"""
N2-CAS(10,20) 40Q, CASSCF orbital basis: HI-NQS top_k scan.

Goal: close the 10.6 mHa gap vs HCI in CASSCF orbitals (job 163794 showed
top_k=10000, n_samples=100000 converged to E=-109.2954938 Ha while HCI in
the same orbital basis reached E=-109.3060890 Ha, ndets=25,452,025).

Scan: n_samples=100000 fixed, top_k in {10000, 15000, 20000, 30000}.
Seed 42, same NQS hyperparameters as bench_orbital_fair.py so results are
directly comparable to the previous point.

CASSCF integrals are cached to a local .npz file after the first build so
that retries avoid the 2 h orbital-optimization step.
"""
import sys, time, json, os
import numpy as np
import torch
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import selected_ci

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
      flush=True)

BOND_LENGTH = 1.10
BASIS = "cc-pvtz"
NELECAS = 10
NCAS = 20
HILBERT = comb(NCAS, NELECAS // 2) ** 2

# HCI reference from job 163794 (bench_orbital_fair)
HCI_CASSCF_E = -109.3060890042
HCI_CASSCF_NDETS = 25_452_025
HCI_CASSCF_TIME = 466.0

CACHE_NPZ = Path(__file__).parent.parent / "results" / "n2_40q_casscf_integrals.npz"

print(f"\nN2-CAS(10,20) 40Q  cc-pVTZ", flush=True)
print(f"  Hilbert space: {HILBERT:,}", flush=True)
print(f"  HCI CASSCF reference: E={HCI_CASSCF_E:.10f} Ha, "
      f"ndets={HCI_CASSCF_NDETS:,}", flush=True)

# -----------------------------------------------------------------------------
# 1. Build or load CASSCF-optimized Hamiltonian
# -----------------------------------------------------------------------------
if CACHE_NPZ.exists():
    print(f"\n[Load] Using cached CASSCF integrals: {CACHE_NPZ}", flush=True)
    data = np.load(CACHE_NPZ)
    integrals = MolecularIntegrals(
        h1e=data["h1e"].astype(np.float64),
        h2e=data["h2e"].astype(np.float64),
        nuclear_repulsion=float(data["e_core"]),
        n_electrons=NELECAS,
        n_orbitals=NCAS,
        n_alpha=NELECAS // 2,
        n_beta=NELECAS // 2,
    )
    casscf_e_tot = float(data["e_casscf"])
    casscf_time = float(data["t_casscf"])
    print(f"  e_core={integrals.nuclear_repulsion:.10f}, "
          f"E_CASSCF={casscf_e_tot:.10f}", flush=True)
else:
    print(f"\n[Build] CASSCF integrals cache miss → running CASSCF+SCI...",
          flush=True)
    mol = gto.Mole()
    mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, BOND_LENGTH))]
    mol.basis = BASIS
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.symmetry = False
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"  RHF: E={mf.e_tot:.10f}", flush=True)

    mc_casscf = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc_casscf.max_cycle_macro = 100
    myci = selected_ci.SCI(mol)
    myci.select_cutoff = 1e-4
    myci.ci_coeff_cutoff = 1e-4
    mc_casscf.fcisolver = myci

    t0 = time.time()
    mc_casscf.kernel()
    casscf_time = time.time() - t0
    casscf_e_tot = float(mc_casscf.e_tot)
    print(f"  CASSCF+SCI converged={mc_casscf.converged}, "
          f"E={casscf_e_tot:.10f}, time={casscf_time:.0f}s", flush=True)

    h1e_cas, e_core = mc_casscf.h1e_for_cas()
    active_mo = mc_casscf.mo_coeff[:, mc_casscf.ncore : mc_casscf.ncore + NCAS]
    h2e_cas = ao2mo.full(mol, active_mo)
    h2e_cas = ao2mo.restore(1, h2e_cas, NCAS)

    integrals = MolecularIntegrals(
        h1e=np.asarray(h1e_cas, dtype=np.float64),
        h2e=np.asarray(h2e_cas, dtype=np.float64),
        nuclear_repulsion=float(e_core),
        n_electrons=NELECAS,
        n_orbitals=NCAS,
        n_alpha=NELECAS // 2,
        n_beta=NELECAS // 2,
    )

    CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_NPZ,
        h1e=integrals.h1e,
        h2e=integrals.h2e,
        e_core=np.float64(integrals.nuclear_repulsion),
        e_casscf=np.float64(casscf_e_tot),
        t_casscf=np.float64(casscf_time),
    )
    print(f"  Saved CASSCF integrals to {CACHE_NPZ}", flush=True)

H_gpu = MolecularHamiltonian(integrals, device="cuda")

# -----------------------------------------------------------------------------
# 2. Top-k scan with n_samples=100000 fixed
# -----------------------------------------------------------------------------
TOP_K_VALUES = [10_000, 15_000, 20_000, 30_000]
N_SAMPLES = 100_000

results = []
info = {"n_qubits": 2 * NCAS}

for top_k in TOP_K_VALUES:
    print(f"\n{'='*78}", flush=True)
    print(f"  [Run] top_k={top_k:,}  n_samples={N_SAMPLES:,}  (CASSCF orbitals)",
          flush=True)
    print(f"{'='*78}", flush=True)

    np.random.seed(42)
    torch.manual_seed(42)

    cfg = HINQSSQDConfig(
        n_samples=N_SAMPLES,
        top_k=top_k,
        max_basis_size=0,
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
        nqs_steps=7,
        nqs_lr=3e-4,
        entropy_weight=0.15,
        warm_start=True,
        use_incremental_sqd=True,
    )

    t0 = time.time()
    r = run_hi_nqs_sqd(H_gpu, info, config=cfg)
    elapsed = time.time() - t0

    de_mha = (r.energy - HCI_CASSCF_E) * 1000
    print(f"\n  → top_k={top_k:,}: E={r.energy:.10f} Ha, "
          f"basis={r.diag_dim:,}, time={elapsed:.0f}s "
          f"({elapsed/3600:.2f}h), converged={r.converged}", flush=True)
    print(f"    ΔE vs HCI(CASSCF) = {de_mha:+.4f} mHa  "
          f"(target < 1.6 mHa)", flush=True)

    results.append({
        "top_k": top_k,
        "n_samples": N_SAMPLES,
        "E": float(r.energy),
        "basis": int(r.diag_dim),
        "time_s": float(elapsed),
        "converged": bool(r.converged),
        "de_vs_hci_mHa": float(de_mha),
    })

    # Stream-save after every run
    out = {
        "system": "N2-CAS(10,20)",
        "qubits": 40,
        "basis_set": BASIS,
        "hilbert": HILBERT,
        "orbital_basis": "CASSCF",
        "casscf_energy": casscf_e_tot,
        "casscf_time_s": casscf_time,
        "hci_reference": {
            "E": HCI_CASSCF_E,
            "ndets": HCI_CASSCF_NDETS,
            "time_s": HCI_CASSCF_TIME,
        },
        "n_samples": N_SAMPLES,
        "runs": results,
    }
    json_path = Path(__file__).parent.parent / "results" / "n2_40q_casscf_topk_scan.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

# -----------------------------------------------------------------------------
# 3. Summary table
# -----------------------------------------------------------------------------
print(f"\n{'='*78}", flush=True)
print(f"  TOP-K SCAN SUMMARY  N2-CAS(10,20) 40Q  CASSCF basis  n_samples=100K",
      flush=True)
print(f"{'='*78}", flush=True)
print(f"\n  HCI reference: E={HCI_CASSCF_E:.10f} Ha, ndets={HCI_CASSCF_NDETS:,}, "
      f"time={HCI_CASSCF_TIME:.0f}s", flush=True)
print(f"\n  {'top_k':>8} {'E (Ha)':>18} {'ΔE (mHa)':>12} {'basis':>10} "
      f"{'time':>10} {'conv':>5}", flush=True)
print(f"  {'-'*68}", flush=True)
for r in results:
    conv = "Y" if r["converged"] else "N"
    print(f"  {r['top_k']:>8,} {r['E']:>18.10f} {r['de_vs_hci_mHa']:>+12.4f} "
          f"{r['basis']:>10,} {r['time_s']:>9.0f}s {conv:>5}", flush=True)

print(f"\n  Results saved to: results/n2_40q_casscf_topk_scan.json", flush=True)
