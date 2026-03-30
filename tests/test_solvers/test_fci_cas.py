"""Tests for CAS-aware FCISolver and graceful FCI-unavailable handling.

Verifies that:
- CAS molecules (``is_cas=True``) use active-space integrals directly
  instead of rebuilding the full molecule.
- Huge Hilbert spaces return ``SolverResult(energy=None, converged=False)``
  instead of hanging or crashing.
- ``_dense_fallback`` returns ``(None, 0, False, ...)`` instead of raising
  ``RuntimeError`` when the Hilbert dim exceeds ``max_configs``.
"""

from __future__ import annotations

from math import comb
from unittest.mock import MagicMock

import numpy as np
import pytest

from qvartools.solvers.reference.fci import FCISolver
from qvartools.solvers.solver import SolverResult

# ---------------------------------------------------------------------------
# Helpers: lightweight mock Hamiltonian with integrals
# ---------------------------------------------------------------------------


def _make_mock_hamiltonian(n_orbitals: int, n_alpha: int, n_beta: int) -> MagicMock:
    """Create a mock Hamiltonian with ``integrals`` attribute."""
    integrals = MagicMock()
    integrals.n_orbitals = n_orbitals
    integrals.n_alpha = n_alpha
    integrals.n_beta = n_beta
    integrals.n_electrons = n_alpha + n_beta
    integrals.nuclear_repulsion = 0.0
    integrals.h1e = np.zeros((n_orbitals, n_orbitals))
    integrals.h2e = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))

    ham = MagicMock()
    ham.integrals = integrals
    ham.num_sites = 2 * n_orbitals
    ham.hilbert_dim = comb(n_orbitals, n_alpha) * comb(n_orbitals, n_beta)
    return ham


# ---------------------------------------------------------------------------
# Test 1: CAS molecule detection
# ---------------------------------------------------------------------------


class TestFCISolverDetectsCASMolecule:
    """FCISolver should use CAS integrals directly when ``is_cas=True``."""

    def test_fci_solver_detects_cas_molecule(self) -> None:
        """When ``mol_info["is_cas"]`` is ``True``, the solver must NOT
        try to rebuild the molecule from geometry.  It should either use
        the active-space integrals directly or return gracefully.

        Currently (before the fix), this crashes because there is no
        geometry to rebuild from.
        """
        ham = _make_mock_hamiltonian(n_orbitals=20, n_alpha=5, n_beta=5)
        mol_info = {"is_cas": True, "name": "test_cas"}

        solver = FCISolver()
        result = solver.solve(ham, mol_info)

        # The solver must return a SolverResult (not crash)
        assert isinstance(result, SolverResult)
        # With is_cas=True and zero integrals, energy should be
        # the nuclear repulsion (0.0) or some finite value — NOT a crash
        if result.energy is not None:
            assert isinstance(result.energy, float)
        # If it chose to skip, converged must be False
        if result.energy is None:
            assert result.converged is False


# ---------------------------------------------------------------------------
# Test 2: Huge Hilbert space returns None instead of hanging
# ---------------------------------------------------------------------------


class TestFCISolverHugeHilbert:
    """FCISolver should return ``energy=None`` for astronomically large systems."""

    def test_fci_solver_returns_none_for_huge_hilbert(self) -> None:
        """A CAS molecule with >50M configs should NOT attempt full FCI.
        The solver should return ``SolverResult(energy=None, converged=False)``.
        """
        # C(30, 8) * C(30, 8) = 5_852_925 * 5_852_925 ~ 34 trillion
        ham = _make_mock_hamiltonian(n_orbitals=30, n_alpha=8, n_beta=8)
        mol_info = {"is_cas": True, "name": "huge_cas"}

        solver = FCISolver()
        result = solver.solve(ham, mol_info)

        assert isinstance(result, SolverResult)
        assert result.energy is None
        assert result.converged is False
        assert "reason" in result.metadata


# ---------------------------------------------------------------------------
# Test 3: _dense_fallback returns None instead of RuntimeError
# ---------------------------------------------------------------------------


class TestDenseFallbackGraceful:
    """``_dense_fallback`` should return ``(None, 0, False, ...)`` for
    Hilbert dimensions exceeding ``max_configs``, instead of raising
    ``RuntimeError``.
    """

    def test_dense_fallback_returns_none_instead_of_error(self) -> None:
        """When ``hilbert_dim > max_configs`` AND PySCF is unavailable,
        the solver should return ``energy=None`` rather than raising.
        """
        ham = _make_mock_hamiltonian(n_orbitals=10, n_alpha=5, n_beta=5)
        # C(10,5)^2 = 63504 > 100
        solver = FCISolver(max_configs=100)

        # _dense_fallback currently raises RuntimeError — after the fix
        # it should return (None, 0, False, {...})
        energy, diag_dim, converged, metadata = solver._dense_fallback(ham)

        assert energy is None
        assert converged is False
        assert "reason" in metadata


# ---------------------------------------------------------------------------
# Test 4: Small CAS system exercises actual FCI kernel
# ---------------------------------------------------------------------------


class TestCASFCIActualExecution:
    """Exercise the actual CAS FCI kernel with small parameters."""

    def test_cas_fci_with_small_system(self) -> None:
        """CAS FCI should compute energy for a small active space.

        Uses n_orbitals=6, n_alpha=3, n_beta=3 so that
        C(6,3)^2 = 400 < 50M limit, which exercises the actual
        ``_try_cas_fci`` kernel instead of hitting the size guard.
        """
        pytest.importorskip("pyscf")
        ham = _make_mock_hamiltonian(n_orbitals=6, n_alpha=3, n_beta=3)
        mol_info = {"is_cas": True, "name": "small_cas_test"}

        solver = FCISolver()
        result = solver.solve(ham, mol_info)

        assert isinstance(result, SolverResult)
        # With zero integrals, energy should be 0.0 (nuclear_repulsion)
        if result.energy is not None:
            assert isinstance(result.energy, float)
