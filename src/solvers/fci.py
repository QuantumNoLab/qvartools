"""Full Configuration Interaction (FCI) solver.

Uses PySCF's FCI solver for consistency with solve_fermion (which also
uses PySCF internally). This ensures HI-NQS energies are always ≥ FCI
energy — no numerical inconsistency between different Hamiltonian
representations.
"""

import time
from math import comb

import numpy as np

from .base import Solver, SolverResult

MAX_FCI_CONFIGS = 500_000


class FCISolver(Solver):
    """Exact diagonalisation via PySCF FCI."""

    def solve(self, hamiltonian, mol_info: dict) -> SolverResult:
        """Run FCI and return the ground-state energy.

        Uses PySCF's FCI solver with the molecular integrals (hcore, eri),
        which is the same backend that solve_fermion uses. This guarantees
        energy consistency: HI-NQS (via solve_fermion) will never produce
        an energy below FCI.
        """
        integrals = hamiltonian.integrals
        n_orb = integrals.n_orbitals
        n_alpha = integrals.n_alpha
        n_beta = integrals.n_beta

        diag_dim = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

        if diag_dim > MAX_FCI_CONFIGS:
            return SolverResult(
                energy=None,
                diag_dim=diag_dim,
                wall_time=0.0,
                method="FCI",
                converged=False,
                metadata={
                    "skipped": True,
                    "reason": (
                        f"FCI determinant space ({diag_dim:,}) exceeds "
                        f"MAX_FCI_CONFIGS ({MAX_FCI_CONFIGS:,})"
                    ),
                },
            )

        t0 = time.perf_counter()

        try:
            energy = _pyscf_fci(integrals, n_orb, n_alpha, n_beta)
        except ImportError:
            # Fallback to internal FCI if PySCF not available
            energy = hamiltonian.fci_energy()

        wall_time = time.perf_counter() - t0

        return SolverResult(
            energy=energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="FCI",
            converged=energy is not None,
        )


def _pyscf_fci(integrals, n_orb, n_alpha, n_beta):
    """Compute FCI energy using PySCF's direct_spin1 FCI solver.

    This is the same computational backend that solve_fermion uses
    (pyscf.fci), ensuring numerical consistency.
    """
    from pyscf import fci

    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)
    nuclear_repulsion = float(integrals.nuclear_repulsion)

    cisolver = fci.direct_spin1.FCI()
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-12

    e_fci, _ = cisolver.kernel(hcore, eri, n_orb, (n_alpha, n_beta))

    return e_fci + nuclear_repulsion
