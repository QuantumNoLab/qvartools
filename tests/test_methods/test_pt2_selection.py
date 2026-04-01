"""Tests for PT2 configuration selection (ADR-005, PR #30 reimplementation).

TDD Red-Green-Refactor for all PT2 selection components.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qvartools.methods.nqs.hi_nqs_sqd import HINQSSQDConfig

# ---------------------------------------------------------------------------
# P0: Config field tests (no PySCF needed)
# ---------------------------------------------------------------------------


class TestHINQSSQDConfigPT2Fields:
    """Verify new PT2 fields exist with correct defaults."""

    def test_use_pt2_selection_default_false(self):
        cfg = HINQSSQDConfig()
        assert cfg.use_pt2_selection is False

    def test_pt2_top_k_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.pt2_top_k == 2000

    def test_max_basis_size_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.max_basis_size == 10_000

    def test_convergence_window_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.convergence_window == 3

    def test_initial_temperature_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.initial_temperature == 1.0

    def test_final_temperature_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.final_temperature == 0.3

    def test_teacher_weight_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.teacher_weight == 1.0

    def test_energy_weight_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.energy_weight == 0.1

    def test_entropy_weight_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.entropy_weight == 0.05

    def test_frozen(self):
        cfg = HINQSSQDConfig()
        with pytest.raises(AttributeError):
            cfg.use_pt2_selection = True  # type: ignore[misc]

    def test_backward_compat_existing_fields(self):
        """Existing fields must retain their defaults."""
        cfg = HINQSSQDConfig()
        assert cfg.n_iterations == 10
        assert cfg.n_samples_per_iter == 10_000
        assert cfg.temperature == 1.0
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# P1: Standalone helper tests
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
class TestComputePT2Scores:
    """Test PT2 Epstein-Nesbet scoring."""

    def test_import(self):
        from qvartools.methods.nqs._pt2_helpers import compute_pt2_scores

        assert callable(compute_pt2_scores)

    def test_scores_shape(self, h2_hamiltonian):
        from qvartools.methods.nqs._pt2_helpers import compute_pt2_scores

        ham = h2_hamiltonian
        all_configs = ham._generate_all_configs()
        H = ham.matrix_elements_fast(all_configs).numpy()
        eigvals, eigvecs = np.linalg.eigh(H)
        e0 = eigvals[0]
        coeffs = eigvecs[:, 0]

        # Score candidates NOT in the basis (use subset as "basis", rest as candidates)
        basis = all_configs[:2]
        candidates = all_configs[2:]
        scores = compute_pt2_scores(candidates, basis, coeffs[:2], ham, e0)
        assert scores.shape == (candidates.shape[0],)
        assert np.all(scores >= 0)

    def test_candidate_in_basis_gets_zero_score(self, h2_hamiltonian):
        """Candidates already in the basis should get score 0 (external only)."""
        from qvartools.methods.nqs._pt2_helpers import compute_pt2_scores

        ham = h2_hamiltonian
        all_configs = ham._generate_all_configs()
        H = ham.matrix_elements_fast(all_configs).numpy()
        eigvals, eigvecs = np.linalg.eigh(H)

        # Score configs that ARE in the basis
        scores = compute_pt2_scores(
            all_configs, all_configs, eigvecs[:, 0], ham, eigvals[0]
        )
        # All candidates are in the basis → all should be 0
        assert np.all(scores == 0.0)


class TestEvictByCoefficient:
    """Test ASCI-style coefficient-based eviction (no PySCF needed)."""

    def test_import(self):
        from qvartools.methods.nqs._pt2_helpers import evict_by_coefficient

        assert callable(evict_by_coefficient)

    def test_keeps_correct_size(self):
        from qvartools.methods.nqs._pt2_helpers import evict_by_coefficient

        rng = np.random.default_rng(42)
        basis = torch.randint(0, 2, (10, 4), dtype=torch.long)
        coeffs = rng.standard_normal(10)
        trimmed_basis, trimmed_coeffs = evict_by_coefficient(basis, coeffs, max_size=5)
        assert trimmed_basis.shape[0] == 5
        assert trimmed_coeffs.shape[0] == 5

    def test_keeps_largest_coefficients(self):
        from qvartools.methods.nqs._pt2_helpers import evict_by_coefficient

        basis = torch.arange(5).unsqueeze(1).long()  # [[0],[1],[2],[3],[4]]
        coeffs = np.array([0.1, 0.9, 0.05, 0.8, 0.02])
        trimmed_basis, trimmed_coeffs = evict_by_coefficient(basis, coeffs, max_size=2)
        # Should keep indices 1 (|0.9|²) and 3 (|0.8|²)
        kept_values = set(trimmed_basis.squeeze().tolist())
        assert kept_values == {1, 3}

    def test_noop_when_under_limit(self):
        from qvartools.methods.nqs._pt2_helpers import evict_by_coefficient

        basis = torch.randint(0, 2, (3, 4), dtype=torch.long)
        coeffs = np.array([0.5, 0.3, 0.2])
        trimmed_basis, trimmed_coeffs = evict_by_coefficient(basis, coeffs, max_size=10)
        assert trimmed_basis.shape[0] == 3

    def test_invalid_max_size_raises(self):
        from qvartools.methods.nqs._pt2_helpers import evict_by_coefficient

        with pytest.raises(ValueError, match="max_size must be >= 1"):
            evict_by_coefficient(torch.zeros(3, 4), np.zeros(3), max_size=0)


class TestComputeTemperature:
    """Test linear temperature schedule (no PySCF needed)."""

    def test_import(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        assert callable(compute_temperature)

    def test_first_iteration_returns_initial(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        t = compute_temperature(iteration=0, max_iterations=10, t_init=2.0, t_final=0.2)
        assert t == pytest.approx(2.0)

    def test_last_iteration_returns_final(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        t = compute_temperature(iteration=9, max_iterations=10, t_init=2.0, t_final=0.2)
        assert t == pytest.approx(0.2)

    def test_monotonically_decreasing(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        temps = [compute_temperature(i, 10, 2.0, 0.2) for i in range(10)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1]

    def test_single_iteration(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        t = compute_temperature(iteration=0, max_iterations=1, t_init=2.0, t_final=0.2)
        assert t == pytest.approx(2.0)

    def test_invalid_max_iterations_raises(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            compute_temperature(0, 0, 1.0, 0.1)

    def test_negative_iteration_raises(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        with pytest.raises(ValueError, match="iteration must be >= 0"):
            compute_temperature(-1, 10, 1.0, 0.1)

    def test_iteration_beyond_max_clamps(self):
        from qvartools.methods.nqs._pt2_helpers import compute_temperature

        t = compute_temperature(
            iteration=20, max_iterations=10, t_init=2.0, t_final=0.2
        )
        assert t == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# P2: 3-term loss tests
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
class TestTrainNqsTeacher3TermLoss:
    """Test enhanced _train_nqs_teacher with 3-term loss."""

    def test_accepts_weight_params(self, h2_hamiltonian):
        """_train_nqs_teacher should accept teacher/energy/entropy weights."""
        from qvartools.methods.nqs.hi_nqs_sqd import _train_nqs_teacher
        from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer

        ham = h2_hamiltonian
        n_orb = ham.integrals.n_orbitals
        nqs = AutoregressiveTransformer(
            n_orbitals=n_orb,
            n_alpha=ham.integrals.n_alpha,
            n_beta=ham.integrals.n_beta,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
        )
        configs = ham.get_hf_state().unsqueeze(0)
        coeffs = np.array([1.0])

        # Should accept the new weight parameters without error
        losses = _train_nqs_teacher(
            nqs,
            configs,
            coeffs,
            n_orb,
            lr=1e-3,
            epochs=2,
            teacher_weight=1.0,
            energy_weight=0.1,
            entropy_weight=0.05,
            hamiltonian=ham,
        )
        assert len(losses) == 2
        assert all(isinstance(v, float) for v in losses)

    def test_backward_compat_defaults(self, h2_hamiltonian):
        """Without new kwargs, behavior is unchanged (teacher-only loss)."""
        from qvartools.methods.nqs.hi_nqs_sqd import _train_nqs_teacher
        from qvartools.nqs.transformer.autoregressive import AutoregressiveTransformer

        ham = h2_hamiltonian
        n_orb = ham.integrals.n_orbitals
        nqs = AutoregressiveTransformer(
            n_orbitals=n_orb,
            n_alpha=ham.integrals.n_alpha,
            n_beta=ham.integrals.n_beta,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
        )
        configs = ham.get_hf_state().unsqueeze(0)
        coeffs = np.array([1.0])

        # Original signature still works (no new kwargs)
        losses = _train_nqs_teacher(
            nqs,
            configs,
            coeffs,
            n_orb,
            lr=1e-3,
            epochs=2,
        )
        assert len(losses) == 2
