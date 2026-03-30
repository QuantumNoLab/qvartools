"""
fci --- Full Configuration Interaction solver
==============================================

Implements :class:`FCISolver`, which computes the exact ground-state
energy via full configuration interaction.  Uses PySCF's FCI module
when available; falls back to dense diagonalisation of the Hamiltonian
matrix for small systems.

For CAS (Complete Active Space) molecules where the Hamiltonian is
defined only in an active-space subblock, the solver uses PySCF's
``fci.direct_spin1.FCI`` on the active-space integrals directly,
avoiding a full-molecule rebuild that would give the wrong energy or
hang on large systems.
"""

from __future__ import annotations

import logging
import time
from math import comb
from typing import Any

from qvartools.hamiltonians.hamiltonian import Hamiltonian
from qvartools.solvers.solver import Solver, SolverResult

__all__ = [
    "FCISolver",
]

logger = logging.getLogger(__name__)

#: Maximum Hilbert-space dimension for CAS FCI via PySCF before the
#: solver gives up and returns ``energy=None``.
_CAS_FCI_MAX_CONFIGS: int = 50_000_000


class FCISolver(Solver):
    """Full configuration interaction solver.

    Attempts to use PySCF's FCI module for molecular Hamiltonians.
    When PySCF is unavailable or the Hamiltonian is not molecular,
    falls back to dense exact diagonalisation via
    :meth:`~qvartools.hamiltonians.hamiltonian.Hamiltonian.exact_ground_state`.

    For CAS molecules (``mol_info["is_cas"] = True``), the solver uses
    the active-space integrals directly instead of rebuilding the full
    molecule, and returns ``energy=None`` when the CAS Hilbert space
    exceeds 50 million configurations.

    Parameters
    ----------
    max_configs : int, optional
        Maximum number of configurations (Hilbert-space dimension) for
        which dense diagonalisation is attempted (default ``500_000``).
        Systems exceeding this limit return ``energy=None`` when PySCF
        is unavailable.

    Attributes
    ----------
    max_configs : int
        Configuration limit for dense fallback.

    Examples
    --------
    >>> solver = FCISolver()
    >>> result = solver.solve(hamiltonian, mol_info)
    >>> result.energy
    -1.1373060356
    """

    def __init__(self, max_configs: int = 500_000) -> None:
        if max_configs < 1:
            raise ValueError(f"max_configs must be >= 1, got {max_configs}")
        self.max_configs: int = max_configs

    def solve(self, hamiltonian: Hamiltonian, mol_info: dict[str, Any]) -> SolverResult:
        """Compute the FCI ground-state energy.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The molecular Hamiltonian.
        mol_info : dict
            Molecular metadata (must contain ``"name"``).  If
            ``"is_cas"`` is ``True``, uses active-space integrals
            directly instead of rebuilding the full molecule.

        Returns
        -------
        SolverResult
            FCI energy result.  ``energy`` is ``None`` when the
            computation was skipped (e.g. CAS too large, dense fallback
            exceeds ``max_configs``).
        """
        t_start = time.perf_counter()

        energy, diag_dim, converged, metadata = self._try_pyscf_fci(
            hamiltonian, mol_info
        )

        if energy is None and not metadata.get("_skip_dense_fallback", False):
            energy, diag_dim, converged, metadata = self._dense_fallback(hamiltonian)

        wall_time = time.perf_counter() - t_start

        if energy is not None:
            logger.info(
                "FCISolver [%s]: energy=%.10f, dim=%d, time=%.2fs",
                mol_info.get("name", "unknown"),
                energy,
                diag_dim,
                wall_time,
            )
        else:
            logger.info(
                "FCISolver [%s]: energy unavailable (reason=%s), time=%.2fs",
                mol_info.get("name", "unknown"),
                metadata.get("reason", "unknown"),
                wall_time,
            )

        return SolverResult(
            energy=energy,
            diag_dim=diag_dim,
            wall_time=wall_time,
            method="FCI",
            converged=converged,
            metadata=metadata,
        )

    def _try_pyscf_fci(
        self, hamiltonian: Hamiltonian, mol_info: dict[str, Any]
    ) -> tuple:
        """Attempt FCI via PySCF.

        For CAS molecules (``mol_info["is_cas"] = True``), uses the
        active-space integrals directly with ``pyscf.fci.direct_spin1``
        instead of rebuilding the full molecule.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            Must have an ``integrals`` attribute for PySCF FCI.
        mol_info : dict
            Molecular metadata.

        Returns
        -------
        tuple
            ``(energy, diag_dim, converged, metadata)`` or
            ``(None, 0, False, {})`` if PySCF is unavailable or the
            Hamiltonian is not molecular.  For CAS molecules that are
            too large, the metadata includes
            ``"_skip_dense_fallback": True`` to prevent the dense
            fallback from being attempted.
        """
        if not hasattr(hamiltonian, "integrals"):
            return None, 0, False, {}

        integrals = hamiltonian.integrals
        n_orb = integrals.n_orbitals
        n_alpha = integrals.n_alpha
        n_beta = integrals.n_beta
        diag_dim = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

        # --- CAS molecule: use active-space integrals directly ---
        if mol_info.get("is_cas", False):
            return self._try_cas_fci(integrals, n_orb, n_alpha, n_beta, diag_dim)

        # --- Standard molecule: rebuild from geometry ---
        try:
            from pyscf import fci, gto, scf
        except ImportError:
            logger.info("PySCF not available; falling back to dense FCI.")
            return None, 0, False, {}

        geometry = mol_info.get("geometry", [])
        basis = mol_info.get("basis", "sto-3g")
        charge = mol_info.get("charge", 0)
        spin = mol_info.get("spin", 0)

        mol = gto.Mole()
        mol.atom = [(atom, coord) for atom, coord in geometry]
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = "Angstrom"
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        cisolver = fci.FCI(mf)
        e_fci, ci_vec = cisolver.kernel()

        metadata: dict[str, Any] = {
            "pyscf_converged": mf.converged,
            "n_orbitals": n_orb,
            "n_alpha": n_alpha,
            "n_beta": n_beta,
        }

        return float(e_fci), diag_dim, True, metadata

    def _try_cas_fci(
        self,
        integrals: Any,
        n_orb: int,
        n_alpha: int,
        n_beta: int,
        diag_dim: int,
    ) -> tuple:
        """Run FCI on CAS (active-space) integrals directly.

        Parameters
        ----------
        integrals : MolecularIntegrals
            Active-space integrals (h1e, h2e, nuclear_repulsion).
        n_orb : int
            Number of active-space spatial orbitals.
        n_alpha : int
            Number of alpha electrons in the active space.
        n_beta : int
            Number of beta electrons in the active space.
        diag_dim : int
            Hilbert-space dimension ``C(n_orb, n_alpha) * C(n_orb, n_beta)``.

        Returns
        -------
        tuple
            ``(energy, diag_dim, converged, metadata)``.  Returns
            ``energy=None`` when the CAS space is too large or PySCF
            is unavailable.
        """
        # Guard: CAS Hilbert space too large for FCI
        if diag_dim > _CAS_FCI_MAX_CONFIGS:
            logger.info(
                "CAS Hilbert space (%d configs) exceeds %d; skipping FCI.",
                diag_dim,
                _CAS_FCI_MAX_CONFIGS,
            )
            return (
                None,
                0,
                False,
                {
                    "reason": "CAS too large for FCI",
                    "hilbert_dim": diag_dim,
                    "_skip_dense_fallback": True,
                },
            )

        try:
            from pyscf import fci as pyscf_fci
        except ImportError:
            logger.info("PySCF not available; cannot run CAS FCI.")
            return (
                None,
                0,
                False,
                {
                    "reason": "pyscf_unavailable",
                    "_skip_dense_fallback": True,
                },
            )

        h1e = integrals.h1e
        h2e = integrals.h2e
        e_core = integrals.nuclear_repulsion

        cisolver = pyscf_fci.direct_spin1.FCI()
        n_electrons = n_alpha + n_beta
        e_fci, ci_vec = cisolver.kernel(h1e, h2e, n_orb, n_electrons)
        e_fci += e_core

        metadata: dict[str, Any] = {
            "cas_fci": True,
            "n_orbitals": n_orb,
            "n_alpha": n_alpha,
            "n_beta": n_beta,
        }

        logger.info(
            "CAS FCI: n_orb=%d, n_el=%d, dim=%d, energy=%.10f",
            n_orb,
            n_electrons,
            diag_dim,
            e_fci,
        )

        return float(e_fci), diag_dim, True, metadata

    def _dense_fallback(self, hamiltonian: Hamiltonian) -> tuple:
        """Fall back to dense exact diagonalisation.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            The Hamiltonian to diagonalise.

        Returns
        -------
        tuple
            ``(energy, diag_dim, converged, metadata)``.  Returns
            ``(None, 0, False, ...)`` when the Hilbert dimension
            exceeds ``max_configs``.
        """
        diag_dim = hamiltonian.hilbert_dim
        if diag_dim > self.max_configs:
            logger.info(
                "Dense FCI requires %d configs (max_configs=%d); skipping.",
                diag_dim,
                self.max_configs,
            )
            return (
                None,
                0,
                False,
                {
                    "reason": "hilbert_dim_exceeded",
                    "hilbert_dim": diag_dim,
                    "max_configs": self.max_configs,
                },
            )

        logger.info("Using dense diagonalisation (dim=%d).", diag_dim)
        energy, _ = hamiltonian.exact_ground_state()

        metadata: dict[str, Any] = {"fallback": "dense_diag"}
        return energy, diag_dim, True, metadata
