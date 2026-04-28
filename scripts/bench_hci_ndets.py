#!/usr/bin/env python3
"""
Get exact number of determinants from PySCF HCI (Selected CI)
for N2-CAS(10,20) 40Q and N2-CAS(10,26) 52Q.
"""
import sys, time
import numpy as np
from math import comb
from pyscf import gto, scf, mcscf
from pyscf.fci import selected_ci

SYSTEMS = [
    ("N2-CAS(10,20)", "cc-pvtz", 10, 20),
    ("N2-CAS(10,26)", "cc-pvtz", 10, 26),
]

for name, basis, nelec, norb in SYSTEMS:
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}: CAS({nelec},{norb}), basis={basis}", flush=True)
    print(f"  Hilbert = {comb(norb, nelec//2)**2:,}", flush=True)
    print(f"{'='*60}", flush=True)

    mol = gto.Mole()
    mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, 1.10))]
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 3
    mol.output = "/dev/null"
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    mc = mcscf.CASSCF(mf, norb, nelec)
    mc.max_cycle_macro = 100

    myci = selected_ci.SCI(mol)
    myci.select_cutoff = 1e-4
    myci.ci_coeff_cutoff = 1e-4
    mc.fcisolver = myci

    t0 = time.time()
    mc.kernel()
    elapsed = time.time() - t0

    E = mc.e_tot

    # Get number of determinants
    ci_vec = mc.ci
    if ci_vec is not None:
        if hasattr(ci_vec, 'shape'):
            ndets = ci_vec.size
            shape = ci_vec.shape
        elif isinstance(ci_vec, (list, tuple)):
            ndets = ci_vec[0].size if len(ci_vec) > 0 else "?"
            shape = ci_vec[0].shape if len(ci_vec) > 0 else "?"
        else:
            ndets = "?"
            shape = "?"
    else:
        ndets = "?"
        shape = "?"

    # Also try to get from fcisolver
    try:
        ci_strs = myci._strs
        if ci_strs is not None:
            na_strs = len(ci_strs[0]) if isinstance(ci_strs, (list, tuple)) else "?"
            nb_strs = len(ci_strs[1]) if isinstance(ci_strs, (list, tuple)) else "?"
        else:
            na_strs = nb_strs = "?"
    except:
        na_strs = nb_strs = "?"

    print(f"\n  Result: E = {E:.10f} Ha", flush=True)
    print(f"  CI vector shape: {shape}", flush=True)
    print(f"  Total elements: {ndets:,}" if isinstance(ndets, int) else f"  Total elements: {ndets}", flush=True)
    print(f"  Alpha strings: {na_strs}", flush=True)
    print(f"  Beta strings:  {nb_strs}", flush=True)
    if isinstance(ndets, int):
        hilbert = comb(norb, nelec//2)**2
        print(f"  Compression: {ndets}/{hilbert:,} = {ndets/hilbert*100:.4f}%", flush=True)
    print(f"  Time: {elapsed:.1f}s", flush=True)
