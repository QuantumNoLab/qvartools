#!/usr/bin/env python3
"""
CASSCF-orbital HI-NQS benchmark.

The original HI-NQS / CIPSI uses CASCI (HF canonical MOs), while HCI uses
CASSCF (optimized orbitals via CASSCF+SCI solver). This creates an ~82 mHa
artificial gap between the methods.

This script uses the SAME CASSCF+SCI orbital-optimization approach as the
HCI benchmark, then runs HI-NQS (Config G, the best from previous sweep)
in the CASSCF-optimized active space.

Systems:
  - N2-CAS(10,14) 28Q: smaller test, CASSCF converges easily
  - N2-CAS(10,20) 40Q: main target (comparison with HCI -109.29642)

HI-NQS Config G: top_k=10K, n_samples=100K, entropy=0.15, lr=3e-4
"""
import sys, time, json
import numpy as np
import torch
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hamiltonians.molecular import MolecularHamiltonian, MolecularIntegrals
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig

from pyscf import gto, scf, mcscf, ao2mo, fci
from pyscf.fci import selected_ci


def build_casscf_sci_hamiltonian(bond_length, basis, cas, device="cuda"):
    """
    Build molecular Hamiltonian in the CASSCF-optimized active space,
    using Selected CI as the internal fcisolver (same as HCI benchmark).

    Returns:
        (MolecularHamiltonian, casscf_energy, casscf_info_dict)
    """
    nelecas, ncas = cas

    mol = gto.Mole()
    mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, bond_length))]
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 3
    mol.output = "/dev/null"
    mol.symmetry = False  # avoid PointGroupSymmetryError on linear molecules
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    print(f"    RHF: E={mf.e_tot:.10f}", flush=True)

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.max_cycle_macro = 100

    # Use Selected CI as fcisolver — same approach as HCI benchmark.
    # This allows CASSCF to handle large active spaces like CAS(10,20).
    myci = selected_ci.SCI(mol)
    myci.select_cutoff = 1e-4
    myci.ci_coeff_cutoff = 1e-4
    mc.fcisolver = myci

    print(f"    Running CASSCF+SCI for CAS({nelecas},{ncas})...", flush=True)
    t0 = time.time()
    mc.kernel()
    t_casscf = time.time() - t0
    print(f"    CASSCF converged: {mc.converged}, "
          f"E_CASSCF={mc.e_tot:.10f}, time={t_casscf:.0f}s", flush=True)

    # Extract CAS integrals in the optimized orbital basis
    h1e_cas, e_core = mc.h1e_for_cas()
    active_mo = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    h2e_cas = ao2mo.full(mol, active_mo)
    h2e_cas = ao2mo.restore(1, h2e_cas, ncas)

    h1e_cas = np.asarray(h1e_cas, dtype=np.float64)
    h2e_cas = np.asarray(h2e_cas, dtype=np.float64)

    n_alpha = nelecas // 2
    n_beta = nelecas // 2

    integrals = MolecularIntegrals(
        h1e=h1e_cas,
        h2e=h2e_cas,
        nuclear_repulsion=float(e_core),
        n_electrons=nelecas,
        n_orbitals=ncas,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    H = MolecularHamiltonian(integrals, device=device)

    return H, mc.e_tot, {
        "t_casscf": t_casscf,
        "converged": mc.converged,
        "casscf_energy": mc.e_tot,
        "cas": cas,
        "basis": basis,
    }


print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# Best config from previous 40Q sweep
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
print(f"HI-NQS Config G: top_k={NQS_CFG.top_k}, n_samples={NQS_CFG.n_samples}, "
      f"entropy={NQS_CFG.entropy_weight}, lr={NQS_CFG.nqs_lr}", flush=True)

SYSTEMS = [
    ("N2-CAS(10,14)",  1.10, "cc-pvtz", (10, 14), "28Q"),
    ("N2-CAS(10,20)",  1.10, "cc-pvtz", (10, 20), "40Q"),
]

all_results = []

for name, R, basis, cas, qlabel in SYSTEMS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {name} ({qlabel}) — R={R} Å, basis={basis}", flush=True)
    print(f"{'='*70}", flush=True)

    nelecas, ncas = cas
    hilbert = comb(ncas, nelecas // 2) ** 2
    print(f"  Hilbert space: {hilbert:,}", flush=True)

    # ---------- Step 1: CASSCF+SCI orbital optimization ----------
    print(f"\n  [Step 1] CASSCF+SCI orbital optimization", flush=True)
    try:
        H, e_casscf, info_casscf = build_casscf_sci_hamiltonian(
            R, basis, cas, device="cuda"
        )
    except Exception as ex:
        print(f"    FAILED: {ex}", flush=True)
        all_results.append({"name": name, "error": str(ex)})
        continue

    # ---------- Step 2: HI-NQS in CASSCF orbital basis ----------
    print(f"\n  [Step 2] HI-NQS (Config G) on CASSCF orbitals", flush=True)
    np.random.seed(42)
    torch.manual_seed(42)

    info = {"n_qubits": 2 * ncas}
    t0 = time.time()
    r_nqs = run_hi_nqs_sqd(H, info, config=NQS_CFG)
    t_nqs = time.time() - t0

    print(f"\n    HI-NQS: E={r_nqs.energy:.10f} Ha, basis={r_nqs.diag_dim}, "
          f"time={t_nqs:.0f}s ({t_nqs/3600:.2f}h), converged={r_nqs.converged}", flush=True)

    # ---------- Step 3: CIPSI in same CASSCF orbital basis ----------
    print(f"\n  [Step 3] CIPSI (same CASSCF orbitals, for fair comparison)", flush=True)
    try:
        from src.solvers.sci import run_sci
        H_cpu = MolecularHamiltonian(H.integrals, device="cpu")
        info_cpu = {"n_qubits": 2 * ncas}
        t0 = time.time()
        r_sci = run_sci(H_cpu, info_cpu, expansion_size=1000, max_basis=0,
                        max_iterations=200)
        t_sci = time.time() - t0
        print(f"    CIPSI: E={r_sci.energy:.10f} Ha, basis={r_sci.diag_dim}, "
              f"time={t_sci:.0f}s", flush=True)
    except Exception as ex:
        print(f"    CIPSI FAILED: {ex}", flush=True)
        r_sci = None
        t_sci = None

    # ---------- Record ----------
    gap_nqs_vs_casscf = (r_nqs.energy - e_casscf) * 1000 if r_nqs.energy else None
    print(f"\n  Summary for {name}:", flush=True)
    print(f"    CASSCF+SCI:  E={e_casscf:.10f}", flush=True)
    print(f"    HI-NQS:      E={r_nqs.energy:.10f} "
          f"(vs CASSCF: {gap_nqs_vs_casscf:+.4f} mHa)" if gap_nqs_vs_casscf else "", flush=True)
    if r_sci:
        gap_cipsi_vs_casscf = (r_sci.energy - e_casscf) * 1000
        gap_nqs_vs_cipsi = (r_nqs.energy - r_sci.energy) * 1000 if r_nqs.energy else None
        print(f"    CIPSI:       E={r_sci.energy:.10f} "
              f"(vs CASSCF: {gap_cipsi_vs_casscf:+.4f} mHa)", flush=True)
        print(f"    HI-NQS vs CIPSI: {gap_nqs_vs_cipsi:+.4f} mHa", flush=True)

    all_results.append({
        "name": name, "qlabel": qlabel, "R": R, "basis": basis, "cas": list(cas),
        "hilbert": hilbert,
        "casscf": {"E": float(e_casscf), "t": info_casscf["t_casscf"],
                   "converged": info_casscf["converged"]},
        "nqs": {"E": float(r_nqs.energy) if r_nqs.energy else None,
                "basis": int(r_nqs.diag_dim), "t": float(t_nqs),
                "converged": r_nqs.converged},
        "cipsi": {"E": float(r_sci.energy) if r_sci else None,
                  "basis": int(r_sci.diag_dim) if r_sci else None,
                  "t": float(t_sci) if t_sci else None} if r_sci else None,
    })

# ---------- Summary ----------
print(f"\n{'='*70}", flush=True)
print(f"  CASSCF-ORBITAL HI-NQS SUMMARY", flush=True)
print(f"{'='*70}", flush=True)

for r in all_results:
    if "error" in r:
        print(f"  {r['name']}: FAILED — {r['error']}", flush=True)
        continue
    print(f"\n  {r['name']} ({r['qlabel']}):", flush=True)
    print(f"    Hilbert = {r['hilbert']:,}", flush=True)
    print(f"    CASSCF+SCI: E={r['casscf']['E']:.10f}, t={r['casscf']['t']:.0f}s", flush=True)
    if r['nqs']['E']:
        print(f"    HI-NQS:     E={r['nqs']['E']:.10f}, basis={r['nqs']['basis']}, "
              f"t={r['nqs']['t']:.0f}s ({r['nqs']['t']/3600:.2f}h)", flush=True)
    if r.get('cipsi') and r['cipsi']['E']:
        print(f"    CIPSI:      E={r['cipsi']['E']:.10f}, basis={r['cipsi']['basis']}, "
              f"t={r['cipsi']['t']:.0f}s", flush=True)

with open("casscf_nqs_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to casscf_nqs_results.json", flush=True)
