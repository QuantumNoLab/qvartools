"""Tests for sparse Hamiltonian construction and raised dense limit.

These tests verify Issue #21 Phase 2:
- The dense config limit in ``matrix_elements_fast()`` is raised from 10K to 50K.
- A new ``build_sparse_hamiltonian()`` method returns a ``scipy.sparse.coo_matrix``.
- Sparse eigenvalues match dense eigenvalues within tight tolerance.
- ``gpu_solve_fermion()`` falls back to sparse path for large bases.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
import torch

# ---------------------------------------------------------------------------
# Test 1: Dense limit raised from 10K to 50K
# ---------------------------------------------------------------------------


class TestDenseLimitRaised:
    """Verify ``matrix_elements_fast`` now allows up to 50K configs."""

    def test_50001_raises_memory_error(self):
        """matrix_elements_fast should raise MemoryError at 50001+ configs."""
        pyscf = pytest.importorskip("pyscf")  # noqa: F841
        from qvartools.hamiltonians import (
            MolecularHamiltonian,
            compute_molecular_integrals,
        )

        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        integrals = compute_molecular_integrals(geometry, basis="sto-3g")
        ham = MolecularHamiltonian(integrals)

        # Create a fake (50001, num_sites) tensor — values don't matter,
        # the guard fires before any computation.
        fake_configs = torch.zeros(50001, ham.num_sites, dtype=torch.long)
        with pytest.raises(MemoryError, match="50000"):
            ham.matrix_elements_fast(fake_configs)

    def test_10001_does_not_raise(self):
        """matrix_elements_fast should NOT raise MemoryError at 10001 configs.

        We only verify the guard does not trigger — the actual matrix build
        will produce nonsense for fake configs, but we never inspect it.
        The old 10K limit would have raised here; 50K does not.
        """
        pyscf = pytest.importorskip("pyscf")  # noqa: F841
        from qvartools.hamiltonians import (
            MolecularHamiltonian,
            compute_molecular_integrals,
        )

        geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
        integrals = compute_molecular_integrals(geometry, basis="sto-3g")
        ham = MolecularHamiltonian(integrals)

        # 10001 configs: the old 10K guard would reject, the new 50K must not.
        # We use real valid configs (repeated) to avoid downstream errors.
        all_configs = ham._generate_all_configs()
        n = all_configs.shape[0]
        # Tile to exceed 10K but stay under 50K
        reps = (10001 // n) + 1
        big_configs = all_configs.repeat(reps, 1)[:10001]

        # Should NOT raise MemoryError (would raise with old 10K limit)
        try:
            ham.matrix_elements_fast(big_configs)
        except MemoryError:
            pytest.fail("matrix_elements_fast raised MemoryError at 10001 configs")


# ---------------------------------------------------------------------------
# Test 2: build_sparse_hamiltonian returns COO matrix
# ---------------------------------------------------------------------------


class TestBuildSparseHamiltonian:
    """Verify the new ``build_sparse_hamiltonian`` method."""

    def test_returns_coo_matrix(self, h2_hamiltonian):
        """``build_sparse_hamiltonian`` should return ``scipy.sparse.coo_matrix``."""
        pytest.importorskip("pyscf")
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        H_sparse = ham.build_sparse_hamiltonian(configs)
        assert isinstance(H_sparse, scipy.sparse.coo_matrix)

    def test_sparse_shape_matches_configs(self, h2_hamiltonian):
        """Sparse matrix shape should be ``(n_configs, n_configs)``."""
        pytest.importorskip("pyscf")
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        n = configs.shape[0]
        H_sparse = ham.build_sparse_hamiltonian(configs)
        assert H_sparse.shape == (n, n)

    def test_sparse_is_symmetric(self, h2_hamiltonian):
        """The sparse Hamiltonian should be Hermitian (symmetric for real)."""
        pytest.importorskip("pyscf")
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        H_sparse = ham.build_sparse_hamiltonian(configs)
        H_dense = H_sparse.toarray()
        np.testing.assert_allclose(
            H_dense,
            H_dense.T,
            atol=1e-12,
            err_msg="Sparse Hamiltonian is not symmetric",
        )

    def test_empty_basis_raises(self, h2_hamiltonian):
        """build_sparse_hamiltonian should raise ValueError for empty configs."""
        pytest.importorskip("pyscf")
        ham = h2_hamiltonian
        empty = torch.zeros(0, ham.num_sites, dtype=torch.long)
        with pytest.raises(ValueError, match="Empty basis"):
            ham.build_sparse_hamiltonian(empty)


# ---------------------------------------------------------------------------
# Test 3: Sparse eigenvalues match dense eigenvalues
# ---------------------------------------------------------------------------


class TestSparseEigenvaluesMatchDense:
    """Compare eigenvalues from sparse and dense Hamiltonian construction."""

    def test_ground_state_energy_matches(self, h2_hamiltonian):
        """Lowest eigenvalue from sparse H should match dense H within 1e-10."""
        pytest.importorskip("pyscf")
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()

        # Dense path (existing)
        H_dense = ham.matrix_elements_fast(configs)
        H_dense_np = H_dense.detach().cpu().numpy()
        evals_dense = np.linalg.eigh(H_dense_np)[0]

        # Sparse path (new)
        H_sparse = ham.build_sparse_hamiltonian(configs)
        H_sparse_dense = H_sparse.toarray()
        evals_sparse = np.linalg.eigh(H_sparse_dense)[0]

        np.testing.assert_allclose(
            evals_sparse[0],
            evals_dense[0],
            atol=1e-10,
            err_msg="Sparse ground-state energy does not match dense",
        )

    def test_full_spectrum_matches(self, h2_hamiltonian):
        """All eigenvalues from sparse H should match dense H within 1e-10."""
        pytest.importorskip("pyscf")
        ham = h2_hamiltonian
        configs = ham._generate_all_configs()

        H_dense = ham.matrix_elements_fast(configs)
        H_dense_np = H_dense.detach().cpu().numpy()
        evals_dense = np.sort(np.linalg.eigh(H_dense_np)[0])

        H_sparse = ham.build_sparse_hamiltonian(configs)
        H_sparse_dense = H_sparse.toarray()
        evals_sparse = np.sort(np.linalg.eigh(H_sparse_dense)[0])

        np.testing.assert_allclose(
            evals_sparse,
            evals_dense,
            atol=1e-10,
            err_msg="Sparse spectrum does not match dense spectrum",
        )


# ---------------------------------------------------------------------------
# Test 4: gpu_solve_fermion sparse fallback
# ---------------------------------------------------------------------------


class TestGpuSolveFermionSparseFallback:
    """Verify ``gpu_solve_fermion`` uses sparse path for large bases."""

    def test_sparse_path_dispatched(self, h2_hamiltonian, monkeypatch):
        """Force the sparse dispatch path via monkeypatch and verify it works."""
        pytest.importorskip("pyscf")
        import qvartools._utils.gpu.diagnostics as diag_mod

        # Set threshold to 1 so even the small H2 basis triggers sparse path
        monkeypatch.setattr(diag_mod, "SPARSE_H_THRESHOLD", 1)

        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        energy, eigvec, (occ_a, occ_b) = diag_mod.gpu_solve_fermion(configs, ham)

        assert isinstance(energy, float)
        assert energy < 0.0
        assert eigvec.shape[0] == configs.shape[0]

    def test_small_system_still_works(self, h2_hamiltonian):
        """gpu_solve_fermion should still work for small systems (H2)."""
        pytest.importorskip("pyscf")
        from qvartools._utils.gpu.diagnostics import gpu_solve_fermion

        ham = h2_hamiltonian
        configs = ham._generate_all_configs()
        energy, eigvec, (occ_a, occ_b) = gpu_solve_fermion(configs, ham)

        # H2 ground state energy should be around -1.85 Ha (with nuclear repulsion)
        assert isinstance(energy, float)
        assert energy < 0.0
        assert eigvec.shape[0] == configs.shape[0]
