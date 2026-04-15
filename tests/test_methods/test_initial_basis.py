"""Tests for initial_basis warm-start in HI+NQS+SQD and HI+NQS+SKQD.

Verifies that the ``initial_basis`` keyword argument correctly seeds the
cumulative basis in iterative NQS pipelines (Issue #10).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from qvartools.methods.nqs.hi_nqs_skqd import HINQSSKQDConfig, run_hi_nqs_skqd
from qvartools.methods.nqs.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from qvartools.solvers.solver import SolverResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ORB = 2
N_ALPHA = 1
N_BETA = 1
N_QUBITS = 2 * N_ORB


@pytest.fixture()
def mol_info():
    return {
        "n_orbitals": N_ORB,
        "n_alpha": N_ALPHA,
        "n_beta": N_BETA,
        "n_qubits": N_QUBITS,
    }


@pytest.fixture()
def minimal_config_sqd():
    return HINQSSQDConfig(
        n_iterations=1,
        n_samples_per_iter=10,
        n_batches=1,
        max_configs_per_batch=100,
        nqs_train_epochs=1,
        embed_dim=16,
        n_heads=2,
        n_layers=1,
    )


@pytest.fixture()
def minimal_config_skqd():
    return HINQSSKQDConfig(
        n_iterations=1,
        n_samples_per_iter=10,
        n_batches=1,
        max_configs_per_batch=100,
        nqs_train_epochs=1,
        embed_dim=16,
        n_heads=2,
        n_layers=1,
        krylov_max_new=5,
        krylov_n_ref=2,
    )


def _make_fake_hamiltonian():
    ham = MagicMock()
    ham.integrals = None
    return ham


def _make_initial_basis(n_configs=3):
    """Create a valid initial basis: binary configs with correct particle number.

    With N_ORB=2, N_ALPHA=1, N_BETA=1 there are only 4 unique configs.
    Default n_configs=3 avoids guaranteed duplicates.
    """
    rng = np.random.default_rng(0)
    configs = []
    for _ in range(n_configs):
        alpha = np.zeros(N_ORB, dtype=np.int64)
        beta = np.zeros(N_ORB, dtype=np.int64)
        alpha[rng.choice(N_ORB, N_ALPHA, replace=False)] = 1
        beta[rng.choice(N_ORB, N_BETA, replace=False)] = 1
        configs.append(np.concatenate([alpha, beta]))
    return torch.tensor(np.array(configs), dtype=torch.long)


# ---------------------------------------------------------------------------
# Tests: run_hi_nqs_sqd
# ---------------------------------------------------------------------------


class TestRunHiNqsSqdInitialBasis:
    """Test initial_basis kwarg for run_hi_nqs_sqd."""

    @patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False)
    @patch("qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion")
    def test_accepts_initial_basis_none(
        self, mock_solver, mol_info, minimal_config_sqd
    ):
        """Default (None) should start from empty basis."""
        mock_solver.return_value = (
            -1.0,
            np.array([1.0]),
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        )
        result = run_hi_nqs_sqd(
            _make_fake_hamiltonian(),
            mol_info,
            config=minimal_config_sqd,
            initial_basis=None,
        )
        assert isinstance(result, SolverResult)
        assert result.method == "HI+NQS+SQD"

    @patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False)
    @patch("qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion")
    def test_accepts_initial_basis_tensor(
        self, mock_solver, mol_info, minimal_config_sqd
    ):
        """Providing a tensor should seed the cumulative basis."""
        mock_solver.return_value = (
            -1.5,
            np.array([1.0]),
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        )
        basis = _make_initial_basis(3)
        result = run_hi_nqs_sqd(
            _make_fake_hamiltonian(),
            mol_info,
            config=minimal_config_sqd,
            initial_basis=basis,
        )
        assert isinstance(result, SolverResult)
        # With warm-start, final basis should be non-empty
        assert result.metadata["final_basis_size"] > 0

    @patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False)
    @patch("qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion")
    def test_initial_basis_deduplicates(
        self, mock_solver, mol_info, minimal_config_sqd
    ):
        """Duplicate rows in initial_basis should be deduplicated."""
        mock_solver.return_value = (
            -1.0,
            np.array([1.0]),
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        )
        single = _make_initial_basis(1)
        duped = single.repeat(10, 1)  # 10 identical rows
        result = run_hi_nqs_sqd(
            _make_fake_hamiltonian(),
            mol_info,
            config=minimal_config_sqd,
            initial_basis=duped,
        )
        assert isinstance(result, SolverResult)


# ---------------------------------------------------------------------------
# Tests: run_hi_nqs_skqd
# ---------------------------------------------------------------------------


class TestRunHiNqsSkqdInitialBasis:
    """Test initial_basis kwarg for run_hi_nqs_skqd."""

    @patch("qvartools.methods.nqs.hi_nqs_skqd.expand_basis_via_connections")
    @patch("qvartools.methods.nqs.hi_nqs_skqd.gpu_solve_fermion")
    def test_accepts_initial_basis_none(
        self, mock_solver, mock_expand, mol_info, minimal_config_skqd
    ):
        """Default (None) should start from empty basis."""
        mock_solver.return_value = (
            -1.0,
            np.array([1.0]),
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        )
        # expand_basis_via_connections should return the input unchanged
        mock_expand.side_effect = lambda basis, *a, **kw: basis
        result = run_hi_nqs_skqd(
            _make_fake_hamiltonian(),
            mol_info,
            config=minimal_config_skqd,
            initial_basis=None,
        )
        assert isinstance(result, SolverResult)
        assert result.method == "HI+NQS+SKQD"

    @patch("qvartools.methods.nqs.hi_nqs_skqd.expand_basis_via_connections")
    @patch("qvartools.methods.nqs.hi_nqs_skqd.gpu_solve_fermion")
    def test_accepts_initial_basis_tensor(
        self, mock_solver, mock_expand, mol_info, minimal_config_skqd
    ):
        """Providing a tensor should seed the cumulative basis."""
        mock_solver.return_value = (
            -1.5,
            np.array([1.0]),
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        )
        mock_expand.side_effect = lambda basis, *a, **kw: basis
        basis = _make_initial_basis(3)
        result = run_hi_nqs_skqd(
            _make_fake_hamiltonian(),
            mol_info,
            config=minimal_config_skqd,
            initial_basis=basis,
        )
        assert isinstance(result, SolverResult)
        assert result.metadata["final_basis_size"] > 0

    @patch("qvartools.methods.nqs.hi_nqs_skqd.expand_basis_via_connections")
    @patch("qvartools.methods.nqs.hi_nqs_skqd.gpu_solve_fermion")
    def test_initial_basis_deduplicates(
        self, mock_solver, mock_expand, mol_info, minimal_config_skqd
    ):
        """Duplicate rows in initial_basis should be deduplicated."""
        mock_solver.return_value = (
            -1.0,
            np.array([1.0]),
            (np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        )
        mock_expand.side_effect = lambda basis, *a, **kw: basis
        single = _make_initial_basis(1)
        duped = single.repeat(10, 1)
        result = run_hi_nqs_skqd(
            _make_fake_hamiltonian(),
            mol_info,
            config=minimal_config_skqd,
            initial_basis=duped,
        )
        assert isinstance(result, SolverResult)


# ---------------------------------------------------------------------------
# Tests: Signature contract
# ---------------------------------------------------------------------------


class TestSignatureContract:
    """Verify that initial_basis is keyword-only."""

    def test_sqd_keyword_only(self):
        """run_hi_nqs_sqd should reject initial_basis as positional."""
        import inspect

        sig = inspect.signature(run_hi_nqs_sqd)
        param = sig.parameters["initial_basis"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_skqd_keyword_only(self):
        """run_hi_nqs_skqd should reject initial_basis as positional."""
        import inspect

        sig = inspect.signature(run_hi_nqs_skqd)
        param = sig.parameters["initial_basis"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_sqd_default_is_none(self):
        """initial_basis should default to None."""
        import inspect

        sig = inspect.signature(run_hi_nqs_sqd)
        assert sig.parameters["initial_basis"].default is None

    def test_skqd_default_is_none(self):
        """initial_basis should default to None."""
        import inspect

        sig = inspect.signature(run_hi_nqs_skqd)
        assert sig.parameters["initial_basis"].default is None


# ---------------------------------------------------------------------------
# Tests: Shape validation
# ---------------------------------------------------------------------------


class TestShapeValidation:
    """Verify that invalid initial_basis shapes raise ValueError."""

    def test_sqd_rejects_1d_tensor(self, mol_info, minimal_config_sqd):
        """1D tensor should raise ValueError."""
        bad_basis = torch.ones(4, dtype=torch.long)
        with pytest.raises(ValueError, match="initial_basis must have shape"):
            run_hi_nqs_sqd(
                _make_fake_hamiltonian(),
                mol_info,
                config=minimal_config_sqd,
                initial_basis=bad_basis,
            )

    def test_sqd_rejects_wrong_columns(self, mol_info, minimal_config_sqd):
        """Wrong number of columns should raise ValueError."""
        bad_basis = torch.ones(3, N_QUBITS + 1, dtype=torch.long)
        with pytest.raises(ValueError, match="initial_basis must have shape"):
            run_hi_nqs_sqd(
                _make_fake_hamiltonian(),
                mol_info,
                config=minimal_config_sqd,
                initial_basis=bad_basis,
            )

    def test_skqd_rejects_1d_tensor(self, mol_info, minimal_config_skqd):
        """1D tensor should raise ValueError."""
        bad_basis = torch.ones(4, dtype=torch.long)
        with pytest.raises(ValueError, match="initial_basis must have shape"):
            run_hi_nqs_skqd(
                _make_fake_hamiltonian(),
                mol_info,
                config=minimal_config_skqd,
                initial_basis=bad_basis,
            )

    def test_skqd_rejects_wrong_columns(self, mol_info, minimal_config_skqd):
        """Wrong number of columns should raise ValueError."""
        bad_basis = torch.ones(3, N_QUBITS + 1, dtype=torch.long)
        with pytest.raises(ValueError, match="initial_basis must have shape"):
            run_hi_nqs_skqd(
                _make_fake_hamiltonian(),
                mol_info,
                config=minimal_config_skqd,
                initial_basis=bad_basis,
            )
