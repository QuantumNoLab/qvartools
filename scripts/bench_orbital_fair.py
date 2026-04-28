#!/usr/bin/env python3
"""
Fair HCI vs HI-NQS comparison: SAME orbital basis, different sampling.

For N2-CAS(10,20) 40Q, run both methods in two orbital bases:

  Orbital basis      | HI-NQS (NQS+PT2)      | HCI (heat-bath SCI)
  ───────────────────┼───────────────────────┼─────────────────────
  HF canonical       | run_hi_nqs_sqd        | CASCI + selected_ci
  CASSCF optimized   | run_hi_nqs_sqd        | CASSCF + selected_ci

→ 4 cells. Cells in the same row use the SAME Hamiltonian, so any energy
  or basis-size difference is purely due to the sampling strategy.

Goal: prove the claim "HI-NQS reaches the same energy with ~600× fewer
determinants than HCI" on a controlled orbital basis.
"""
import sys, time, json
import numpy as np
import torch
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import selected_ci

# -----------------------------------------------------------------------------
# 0. System setup
# -----------------------------------------------------------------------------
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

BOND_LENGTH = 1.10
BASIS = "cc-pvtz"
NELECAS = 10
NCAS = 20
HILBERT = comb(NCAS, NELECAS // 2) ** 2

print(f"\nN2-CAS(10,20) 40Q  cc-pVTZ", flush=True)
print(f"  Hilbert space: {HILBERT:,}", flush=True)

# Build PySCF molecule
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

# -----------------------------------------------------------------------------
# 1. Build TWO Hamiltonians: HF canonical & CASSCF-optimized
# -----------------------------------------------------------------------------

def build_h_from_mc(mc, label):
    """Extract h1e, h2e in active orbital basis from an mcscf solver."""
    h1e_cas, e_core = mc.h1e_for_cas()
    active_mo = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
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
    print(f"  [{label}] integrals built, e_core={e_core:.10f}", flush=True)
    return integrals

# Path A: HF canonical orbitals (CASCI mode)
print(f"\n[Step A] Building Hamiltonian with HF canonical orbitals...", flush=True)
mc_hf = mcscf.CASCI(mf, NCAS, NELECAS)
integrals_hf = build_h_from_mc(mc_hf, "HF orbitals")
H_hf_gpu = MolecularHamiltonian(integrals_hf, device="cuda")
H_hf_cpu = MolecularHamiltonian(integrals_hf, device="cpu")

# Path B: CASSCF-optimized orbitals (run CASSCF+SCI to get optimized orbitals)
print(f"\n[Step B] Running CASSCF+SCI to optimize orbitals...", flush=True)
mc_casscf = mcscf.CASSCF(mf, NCAS, NELECAS)
mc_casscf.max_cycle_macro = 100
myci_for_casscf = selected_ci.SCI(mol)
myci_for_casscf.select_cutoff = 1e-4
myci_for_casscf.ci_coeff_cutoff = 1e-4
mc_casscf.fcisolver = myci_for_casscf

t0 = time.time()
mc_casscf.kernel()
t_casscf = time.time() - t0
print(f"  CASSCF+SCI converged: {mc_casscf.converged}, "
      f"E_CASSCF={mc_casscf.e_tot:.10f}, time={t_casscf:.0f}s", flush=True)

integrals_cs = build_h_from_mc(mc_casscf, "CASSCF orbitals")
H_cs_gpu = MolecularHamiltonian(integrals_cs, device="cuda")
H_cs_cpu = MolecularHamiltonian(integrals_cs, device="cpu")

# -----------------------------------------------------------------------------
# 2. HCI in BOTH orbital bases (with the same selected_ci tolerance)
# -----------------------------------------------------------------------------

def run_hci(integrals, mol_for_solver, label):
    """Run selected_ci.kernel_fixed_space-style SCI sweep on given integrals."""
    print(f"\n[HCI] {label}", flush=True)
    myci = selected_ci.SCI(mol_for_solver)
    myci.select_cutoff = 1e-4
    myci.ci_coeff_cutoff = 1e-4

    # Use a CASCI shell to host the solver with these integrals
    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    mc.fcisolver = myci

    # Inject the integrals (CASCI calls h1e_for_cas and ao2mo internally,
    # so we monkey-patch them to return our pre-computed integrals)
    h1e_arr = integrals.h1e
    h2e_arr = integrals.h2e
    e_core = integrals.nuclear_repulsion
    mc.get_h2eff = lambda mo=None: ao2mo.restore(8, h2e_arr, NCAS)
    mc.h1e_for_cas = lambda mo=None, ncas=None, ncore=None: (h1e_arr, e_core)

    t0 = time.time()
    mc.kernel()
    t_hci = time.time() - t0
    e_hci = mc.e_tot

    ci_vec = mc.ci
    if ci_vec is not None and hasattr(ci_vec, 'shape'):
        ndets = ci_vec.size
        shape = ci_vec.shape
    else:
        ndets = None
        shape = "?"

    print(f"  HCI: E={e_hci:.10f}, shape={shape}, "
          f"ndets={ndets:,}" if isinstance(ndets, int) else f"  HCI: E={e_hci:.10f}", flush=True)
    print(f"  Time: {t_hci:.1f}s", flush=True)
    return {"E": float(e_hci), "ndets": int(ndets) if ndets else None,
            "time": float(t_hci)}

hci_hf = run_hci(integrals_hf, mol, "HF orbital basis (CASCI)")
hci_cs = run_hci(integrals_cs, mol, "CASSCF orbital basis")

# -----------------------------------------------------------------------------
# 3. HI-NQS in BOTH orbital bases (Config G + incremental backend)
# -----------------------------------------------------------------------------

def run_hinqs(H_gpu, label):
    print(f"\n[HI-NQS] {label}", flush=True)
    np.random.seed(42); torch.manual_seed(42)
    cfg = HINQSSQDConfig(
        n_samples=100000,
        top_k=10000,
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
    info = {"n_qubits": 2 * NCAS}
    t0 = time.time()
    r = run_hi_nqs_sqd(H_gpu, info, config=cfg)
    elapsed = time.time() - t0
    print(f"  HI-NQS: E={r.energy:.10f}, basis={r.diag_dim}, "
          f"time={elapsed:.0f}s ({elapsed/3600:.2f}h), converged={r.converged}", flush=True)
    return {"E": float(r.energy), "basis": int(r.diag_dim),
            "time": float(elapsed), "converged": r.converged}

hi_hf = run_hinqs(H_hf_gpu, "HF orbital basis")
hi_cs = run_hinqs(H_cs_gpu, "CASSCF orbital basis")

# -----------------------------------------------------------------------------
# 4. Final comparison table
# -----------------------------------------------------------------------------
print(f"\n{'='*78}", flush=True)
print(f"  FAIR ORBITAL COMPARISON  N2-CAS(10,20) 40Q", flush=True)
print(f"{'='*78}", flush=True)
print(f"\n  {'Orbital basis':<22} {'Method':<14} {'Energy (Ha)':<18} {'basis/ndets':>12} {'time':>10}", flush=True)
print(f"  {'-'*78}", flush=True)
print(f"  {'HF canonical':<22} {'HCI':<14} {hci_hf['E']:<18.10f} "
      f"{hci_hf['ndets'] or 0:>12,} {hci_hf['time']:>9.0f}s", flush=True)
print(f"  {'HF canonical':<22} {'HI-NQS':<14} {hi_hf['E']:<18.10f} "
      f"{hi_hf['basis']:>12,} {hi_hf['time']:>9.0f}s", flush=True)
print(f"  {'CASSCF optimized':<22} {'HCI':<14} {hci_cs['E']:<18.10f} "
      f"{hci_cs['ndets'] or 0:>12,} {hci_cs['time']:>9.0f}s", flush=True)
print(f"  {'CASSCF optimized':<22} {'HI-NQS':<14} {hi_cs['E']:<18.10f} "
      f"{hi_cs['basis']:>12,} {hi_cs['time']:>9.0f}s", flush=True)

# Per-row comparison (same orbital basis)
print(f"\n  Same-orbital-basis efficiency (smaller basis at same energy = better):", flush=True)
for label, hci_r, hi_r in [
    ("HF canonical",     hci_hf, hi_hf),
    ("CASSCF optimized", hci_cs, hi_cs),
]:
    de_mHa = (hi_r["E"] - hci_r["E"]) * 1000
    if hci_r['ndets']:
        ratio = hci_r['ndets'] / hi_r['basis']
        print(f"    {label}: ΔE(HI-NQS - HCI) = {de_mHa:+.4f} mHa, "
              f"basis ratio HCI/HI-NQS = {ratio:.1f}x", flush=True)

results = {
    "system": "N2-CAS(10,20)", "qubits": 40, "basis_set": BASIS,
    "hilbert": HILBERT,
    "rhf_energy": float(mf.e_tot),
    "casscf_energy": float(mc_casscf.e_tot),
    "casscf_time": float(t_casscf),
    "hci_hf_orbitals":     hci_hf,
    "hci_casscf_orbitals": hci_cs,
    "hinqs_hf_orbitals":     hi_hf,
    "hinqs_casscf_orbitals": hi_cs,
}
with open("orbital_fair_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to orbital_fair_results.json", flush=True)
