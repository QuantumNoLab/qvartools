"""N2-40Q basis budget scan — CIPSI / HCI / NQS head-to-head.

For a given (backend, budget) pair, run the corresponding solver on
N2-CAS(10,20) cc-pVTZ HF canonical orbitals and record (N_det, E, wall).

Backends:
  - hci   : PySCF selected_ci.SCI with select_cutoff tuned to hit budget
  - cipsi : our CIPSISolver (Huron-Malrieu PT2 selection), max_basis cap
  - nqs   : already collected from v3/v4; pulled from existing results

The 40Q HF reference is HCI ε=1e-4 = -109.21473333 (17.4M dets).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from pyscf import gto, scf, mcscf
from pyscf.fci import selected_ci

from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver


REF_HCI_HF = -109.21473332712169


def run_hci_scan_single(cutoff: float):
    """PySCF HCI at given cutoff. Returns {'E', 'N_det', 'n_a', 'n_b', 'wall'}."""
    mol = gto.Mole()
    mol.atom = [("N", (0, 0, 0)), ("N", (0, 0, 1.10))]
    mol.basis = "cc-pvtz"
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    mc = mcscf.CASCI(mf, 20, 10)
    myci = selected_ci.SCI(mol)
    myci.select_cutoff = cutoff
    myci.ci_coeff_cutoff = cutoff
    mc.fcisolver = myci

    t0 = time.time()
    mc.kernel()
    wall = time.time() - t0

    strs = getattr(myci, "_strs", None)
    if strs is not None:
        n_a = len(strs[0]); n_b = len(strs[1])
    else:
        n_a = n_b = None
    n_det = n_a * n_b if n_a and n_b else int(np.asarray(mc.ci).size)

    return {
        "E": float(mc.e_tot), "N_det": int(n_det),
        "n_a": n_a, "n_b": n_b,
        "cutoff": float(cutoff),
        "wall_s": float(wall),
    }


def run_cipsi_single(max_basis: int):
    """Our CIPSISolver at given budget. Returns (E, N_det, wall)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, info = get_molecule("N2-CAS(10,20)", device=device)

    solver = CIPSISolver(
        max_iterations=50, max_basis_size=max_basis,
        convergence_threshold=1e-7,
        expansion_size=max(100, max_basis // 20),
    )
    t0 = time.time()
    r = solver.solve(H, info)
    wall = time.time() - t0
    return {
        "E": float(r.energy), "N_det": int(r.diag_dim),
        "budget": int(max_basis),
        "wall_s": float(wall), "converged": bool(r.converged),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["hci", "cipsi"])
    parser.add_argument("--param", required=True, type=float,
                        help="cutoff for hci (e.g. 1e-4), or max_basis for cipsi (e.g. 100000)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"backend={args.backend}  param={args.param}", flush=True)

    if args.backend == "hci":
        result = run_hci_scan_single(args.param)
    else:
        result = run_cipsi_single(int(args.param))

    err = (result["E"] - REF_HCI_HF) * 1000
    print(f"\nE = {result['E']:.10f}", flush=True)
    print(f"N_det = {result['N_det']:,}", flush=True)
    print(f"err vs HCI ref = {err:+.3f} mHa", flush=True)
    print(f"wall = {result['wall_s']:.1f}s", flush=True)

    result["backend"] = args.backend
    result["ref_hci_hf"] = REF_HCI_HF
    result["err_mha"] = float(err)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
