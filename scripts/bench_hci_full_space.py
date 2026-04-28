#!/usr/bin/env python3
"""
HCI in FULL space (no artificial CAS) for fair comparison with HI-NQS.

Previous bench_pyscf_methods.py limited HCI to CAS(10,10) for C2H2,
which is unfair vs HI-NQS running in the full 12-orbital space.

This script runs HCI in the full molecular orbital space for all
small/medium molecules where it's feasible.

Systems:
  H2O   14Q  STO-3G  (5α,5β) full → C(7,5)² = 441
  NH3   16Q  STO-3G  (5α,5β) full → C(8,5)² = 3,136
  N2    20Q  STO-3G  (7α,7β) full → C(10,7)² = 14,400
  C2H2  24Q  STO-3G  (7α,7β) full → C(12,7)² = 627,264
  C2H4  28Q  STO-3G  (8α,8β) full → C(14,8)² = 9,018,009
"""
import sys, time, json
import numpy as np
from math import comb
from pyscf import gto, scf, mcscf
from pyscf.fci import selected_ci

SYSTEMS = [
    ("H2O", [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.757, 0.586, 0.0)),
        ("H", (-0.757, 0.586, 0.0)),
    ], "sto-3g"),
    ("NH3", [
        ("N", (0.0, 0.0, 0.0)),
        ("H", (0.0, -0.9377, -0.3816)),
        ("H", (0.8121, 0.4689, -0.3816)),
        ("H", (-0.8121, 0.4689, -0.3816)),
    ], "sto-3g"),
    ("N2", [
        ("N", (0, 0, 0)),
        ("N", (0, 0, 1.10)),
    ], "sto-3g"),
    ("C2H2", [
        ("H", (0, 0, 0)),
        ("C", (0, 0, 1.06)),
        ("C", (0, 0, 2.26)),
        ("H", (0, 0, 3.32)),
    ], "sto-3g"),
]

results = {}

for name, geom, basis in SYSTEMS:
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}  basis={basis}  (FULL SPACE)", flush=True)
    print(f"{'='*60}", flush=True)

    mol = gto.M(atom=geom, basis=basis, spin=0, charge=0, verbose=0)
    mol.build()

    nocc = mol.nelec[0]  # alpha electrons
    norb = mol.nao       # total orbitals (= active space)
    hilbert = comb(norb, nocc) ** 2
    print(f"  orbitals={norb}, electrons=({mol.nelec[0]},{mol.nelec[1]}), "
          f"Hilbert={hilbert:,}", flush=True)

    # RHF
    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.kernel()
    print(f"  HF: E={mf.e_tot:.10f}", flush=True)

    # HCI in full space (CASCI with all orbitals = no CAS truncation)
    nelec_total = mol.nelec[0] + mol.nelec[1]
    mc = mcscf.CASCI(mf, norb, nelec_total)

    myci = selected_ci.SCI(mol)
    myci.select_cutoff = 1e-4
    myci.ci_coeff_cutoff = 1e-4
    mc.fcisolver = myci

    t0 = time.time()
    mc.kernel()
    t_hci = time.time() - t0

    E_hci = mc.e_tot

    # Get number of determinants
    ci_vec = mc.ci
    if ci_vec is not None and hasattr(ci_vec, 'shape'):
        ndets = ci_vec.size
        shape = ci_vec.shape
    else:
        ndets = "?"
        shape = "?"

    print(f"\n  HCI (full): E={E_hci:.10f}, t={t_hci:.1f}s", flush=True)
    print(f"  CI shape={shape}, ndets={ndets:,}" if isinstance(ndets, int) else f"  CI shape={shape}", flush=True)
    if isinstance(ndets, int):
        print(f"  Compression: {ndets}/{hilbert:,} = {ndets/hilbert*100:.4f}%", flush=True)

    results[name] = {
        "energy": float(E_hci),
        "time": float(t_hci),
        "ndets": int(ndets) if isinstance(ndets, int) else None,
        "hilbert": hilbert,
        "norb": norb,
        "nelec": list(mol.nelec),
    }

# Summary
print(f"\n{'='*60}", flush=True)
print(f"  HCI FULL-SPACE SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  {'System':<8} {'norb':>5} {'Hilbert':>15} {'Energy':>16} {'ndets':>10} {'time':>8}", flush=True)
print(f"  {'-'*70}", flush=True)
for name, r in results.items():
    nd = f"{r['ndets']:,}" if r['ndets'] else "?"
    print(f"  {name:<8} {r['norb']:>5} {r['hilbert']:>15,} {r['energy']:>16.10f} "
          f"{nd:>10} {r['time']:>7.1f}s", flush=True)

with open("hci_full_space_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to hci_full_space_results.json", flush=True)
