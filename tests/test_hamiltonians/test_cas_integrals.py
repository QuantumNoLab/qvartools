"""Tests for CAS (Complete Active Space) integral computation.

TDD Red phase for Issue #21 Phase 1 (ADR-004).
Tests verify that compute_molecular_integrals supports CAS/CASCI
and returns correct active-space integrals.
"""

from __future__ import annotations

import importlib.util
import inspect
from math import comb

import numpy as np
import pytest

from qvartools.hamiltonians.integrals import (
    MolecularIntegrals,
    cached_compute_molecular_integrals,
    compute_molecular_integrals,
)

_HAS_PYSCF = importlib.util.find_spec("pyscf") is not None
pyscf_required = pytest.mark.skipif(not _HAS_PYSCF, reason="PySCF not installed")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N2_GEOMETRY = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]
CR2_GEOMETRY = [("Cr", (0.0, 0.0, 0.0)), ("Cr", (0.0, 0.0, 1.68))]
BENZENE_CC = 1.40
BENZENE_CH = 1.08


def _benzene_geometry() -> list[tuple[str, tuple[float, float, float]]]:
    """Regular hexagon benzene geometry."""
    geom = []
    for i in range(6):
        angle = i * np.pi / 3.0
        cx = BENZENE_CC * np.cos(angle)
        cy = BENZENE_CC * np.sin(angle)
        geom.append(("C", (cx, cy, 0.0)))
        hx = (BENZENE_CC + BENZENE_CH) * np.cos(angle)
        hy = (BENZENE_CC + BENZENE_CH) * np.sin(angle)
        geom.append(("H", (hx, hy, 0.0)))
    return geom


# ---------------------------------------------------------------------------
# Tests: compute_molecular_integrals CAS parameter signature
# ---------------------------------------------------------------------------


class TestCASParameterSignature:
    """Verify that cas and casci parameters exist on compute_molecular_integrals."""

    def test_cas_parameter_exists(self):
        sig = inspect.signature(compute_molecular_integrals)
        assert "cas" in sig.parameters, "compute_molecular_integrals must accept 'cas'"

    def test_casci_parameter_exists(self):
        sig = inspect.signature(compute_molecular_integrals)
        assert "casci" in sig.parameters, (
            "compute_molecular_integrals must accept 'casci'"
        )

    def test_cas_default_is_none(self):
        sig = inspect.signature(compute_molecular_integrals)
        assert sig.parameters["cas"].default is None

    def test_casci_default_is_false(self):
        sig = inspect.signature(compute_molecular_integrals)
        assert sig.parameters["casci"].default is False


# ---------------------------------------------------------------------------
# Tests: N₂ CAS(10,8) integral shapes and values
# ---------------------------------------------------------------------------


@pyscf_required
@pytest.mark.pyscf
@pytest.mark.slow
class TestN2CAS10_8:
    """N₂ CAS(10,8) on cc-pVDZ — small enough for full CASSCF."""

    @pytest.fixture(scope="class")
    def integrals(self) -> MolecularIntegrals:
        return compute_molecular_integrals(N2_GEOMETRY, basis="cc-pvdz", cas=(10, 8))

    def test_h1e_shape(self, integrals: MolecularIntegrals):
        assert integrals.h1e.shape == (8, 8)

    def test_h2e_shape(self, integrals: MolecularIntegrals):
        assert integrals.h2e.shape == (8, 8, 8, 8)

    def test_n_orbitals(self, integrals: MolecularIntegrals):
        assert integrals.n_orbitals == 8

    def test_n_alpha(self, integrals: MolecularIntegrals):
        assert integrals.n_alpha == 5

    def test_n_beta(self, integrals: MolecularIntegrals):
        assert integrals.n_beta == 5

    def test_n_electrons(self, integrals: MolecularIntegrals):
        assert integrals.n_electrons == 10

    def test_nuclear_repulsion_is_ecore(self, integrals: MolecularIntegrals):
        """nuclear_repulsion must be e_core (frozen-core), NOT mol.energy_nuc().

        For N₂ cc-pVDZ: e_core ≈ -77.4, mol.energy_nuc() ≈ 23.6.
        The difference is ~100 Ha (frozen electron energy).
        """
        # e_core is negative (includes frozen electron energy)
        assert integrals.nuclear_repulsion < 0.0, (
            f"nuclear_repulsion should be e_core (negative), "
            f"got {integrals.nuclear_repulsion:.4f}"
        )
        # True nuclear repulsion for N₂ at 1.10 Å is ~23.6 Ha (positive)
        assert integrals.nuclear_repulsion < -50.0, (
            f"nuclear_repulsion={integrals.nuclear_repulsion:.4f} is too high; "
            f"expected e_core ≈ -77.4 for N₂ CAS(10,8)"
        )

    def test_valid_config_count(self, integrals: MolecularIntegrals):
        """Hilbert space dim = C(ncas, n_alpha) * C(ncas, n_beta)."""
        expected = comb(integrals.n_orbitals, integrals.n_alpha) * comb(
            integrals.n_orbitals, integrals.n_beta
        )
        assert expected == 3136


# ---------------------------------------------------------------------------
# Tests: N₂ CAS(10,12) — 24 qubits
# ---------------------------------------------------------------------------


@pyscf_required
@pytest.mark.pyscf
@pytest.mark.slow
class TestN2CAS10_12:
    """N₂ CAS(10,12) on cc-pVDZ — 24 qubits."""

    @pytest.fixture(scope="class")
    def integrals(self) -> MolecularIntegrals:
        return compute_molecular_integrals(N2_GEOMETRY, basis="cc-pvdz", cas=(10, 12))

    def test_n_orbitals_is_ncas(self, integrals: MolecularIntegrals):
        assert integrals.n_orbitals == 12

    def test_electron_counts(self, integrals: MolecularIntegrals):
        assert integrals.n_alpha == 5
        assert integrals.n_beta == 5
        assert integrals.n_electrons == 10

    def test_h1e_h2e_shapes(self, integrals: MolecularIntegrals):
        assert integrals.h1e.shape == (12, 12)
        assert integrals.h2e.shape == (12, 12, 12, 12)


# ---------------------------------------------------------------------------
# Tests: CASCI fallback for large active spaces
# ---------------------------------------------------------------------------


@pyscf_required
@pytest.mark.pyscf
@pytest.mark.slow
class TestCASCIFallback:
    """Verify auto-CASCI when ncas >= 15 or explicit casci=True."""

    def test_explicit_casci_produces_integrals(self):
        """casci=True should produce valid integrals without orbital optimization."""
        mi = compute_molecular_integrals(
            N2_GEOMETRY, basis="cc-pvdz", cas=(10, 8), casci=True
        )
        assert isinstance(mi, MolecularIntegrals)
        assert mi.n_orbitals == 8

    def test_large_ncas_auto_casci(self):
        """CAS(10,15) should auto-use CASCI (ncas >= 15)."""
        mi = compute_molecular_integrals(N2_GEOMETRY, basis="cc-pvdz", cas=(10, 15))
        assert isinstance(mi, MolecularIntegrals)
        assert mi.n_orbitals == 15
        assert mi.n_alpha == 5
        assert mi.n_beta == 5


# ---------------------------------------------------------------------------
# Tests: Cache bypass for CAS
# ---------------------------------------------------------------------------


@pyscf_required
@pytest.mark.pyscf
@pytest.mark.slow
class TestCacheBypassForCAS:
    """joblib cache must NOT cache CAS results (CASSCF is non-deterministic)."""

    def test_cached_fn_accepts_cas_param(self):
        """cached_compute_molecular_integrals must accept cas kwarg."""
        sig = inspect.signature(cached_compute_molecular_integrals)
        assert "cas" in sig.parameters

    def test_cached_fn_returns_cas_integrals(self):
        """cached version should still return correct CAS integrals."""
        mi = cached_compute_molecular_integrals(
            N2_GEOMETRY, basis="cc-pvdz", cas=(10, 8)
        )
        assert isinstance(mi, MolecularIntegrals)
        assert mi.n_orbitals == 8

    def test_cache_actually_bypassed_for_cas(self):
        """Verify joblib cached function is NOT called when cas is set."""
        from unittest.mock import MagicMock, patch

        mock_cached_fn = MagicMock()
        with patch(
            "qvartools.hamiltonians.integrals._default_cached_fn",
            mock_cached_fn,
        ):
            cached_compute_molecular_integrals(
                N2_GEOMETRY, basis="cc-pvdz", cas=(10, 8)
            )
            mock_cached_fn.assert_not_called(), (
                "joblib cached function was invoked for CAS integrals — "
                "CAS results must bypass the cache"
            )
