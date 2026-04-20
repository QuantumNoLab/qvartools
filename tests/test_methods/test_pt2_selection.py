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
        assert cfg.energy_weight == 0.0

    def test_entropy_weight_default(self):
        cfg = HINQSSQDConfig()
        assert cfg.entropy_weight == 0.0

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


@pytest.mark.pyscf
class TestComputeEPT2:
    """Test EN-PT2 energy correction."""

    @pytest.fixture()
    def h2_valid_configs(self, h2_hamiltonian):
        """Return only particle-number-valid configs for H2."""
        ham = h2_hamiltonian
        n_orb = ham.integrals.n_orbitals
        n_alpha = ham.integrals.n_alpha
        n_beta = ham.integrals.n_beta
        all_configs = ham._generate_all_configs()
        # Filter for correct particle number
        alpha_count = all_configs[:, :n_orb].sum(dim=1)
        beta_count = all_configs[:, n_orb:].sum(dim=1)
        valid_mask = (alpha_count == n_alpha) & (beta_count == n_beta)
        return all_configs[valid_mask]

    def test_import(self):
        from qvartools.methods.nqs._pt2_helpers import compute_e_pt2

        assert callable(compute_e_pt2)

    def test_returns_float(self, h2_hamiltonian):
        from qvartools.methods.nqs._pt2_helpers import compute_e_pt2

        ham = h2_hamiltonian
        # Use HF state as 1-config basis
        hf = ham.get_hf_state().unsqueeze(0)
        e_var = float(ham.diagonal_element(hf[0]))
        e_pt2 = compute_e_pt2(hf, np.array([1.0]), ham, e_var)
        assert isinstance(e_pt2, float)

    def test_e_pt2_is_negative(self, h2_hamiltonian):
        """E_PT2 from HF reference should be negative (lowers energy)."""
        from qvartools.methods.nqs._pt2_helpers import compute_e_pt2

        ham = h2_hamiltonian
        hf = ham.get_hf_state().unsqueeze(0)
        e_var = float(ham.diagonal_element(hf[0]))
        e_pt2 = compute_e_pt2(hf, np.array([1.0]), ham, e_var)
        assert e_pt2 < 0, f"E_PT2 should be negative, got {e_pt2}"

    def test_corrected_energy_closer_to_fci(self, h2_hamiltonian, h2_valid_configs):
        """E_var + E_PT2 should be closer to FCI than E_var alone."""
        from qvartools.methods.nqs._pt2_helpers import compute_e_pt2

        ham = h2_hamiltonian
        valid = h2_valid_configs
        H = ham.matrix_elements_fast(valid).numpy()
        eigvals, _ = np.linalg.eigh(H)
        e_fci = eigvals[0]

        # Use HF as 1-config basis (guaranteed lowest diagonal energy)
        hf = ham.get_hf_state().unsqueeze(0)
        e_var = float(ham.diagonal_element(hf[0]))
        e_pt2 = compute_e_pt2(hf, np.array([1.0]), ham, e_var)
        e_corrected = e_var + e_pt2

        error_var = abs(e_var - e_fci)
        error_corrected = abs(e_corrected - e_fci)
        assert error_corrected < error_var, (
            f"E_var+E_PT2 ({e_corrected:.8f}, err={error_corrected:.6f}) "
            f"should be closer to FCI ({e_fci:.8f}) than E_var ({e_var:.8f}, err={error_var:.6f})"
        )

    def test_full_basis_gives_zero(self, h2_hamiltonian, h2_valid_configs):
        """When basis spans all valid configs, E_PT2 should be ~0."""
        from qvartools.methods.nqs._pt2_helpers import compute_e_pt2

        ham = h2_hamiltonian
        valid = h2_valid_configs
        H = ham.matrix_elements_fast(valid).numpy()
        eigvals, eigvecs = np.linalg.eigh(H)

        e_pt2 = compute_e_pt2(valid, eigvecs[:, 0], ham, eigvals[0])
        assert abs(e_pt2) < 1e-10, f"Full-basis E_PT2 should be ~0, got {e_pt2}"


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

    def test_energy_weight_without_hamiltonian_raises(self, h2_hamiltonian):
        """energy_weight > 0 with hamiltonian=None must raise ValueError."""
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

        with pytest.raises(ValueError, match="energy_weight.*requires hamiltonian"):
            _train_nqs_teacher(
                nqs,
                configs,
                coeffs,
                n_orb,
                lr=1e-3,
                epochs=1,
                energy_weight=0.1,
                hamiltonian=None,
            )

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


# ---------------------------------------------------------------------------
# P3: Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
class TestRunHiNqsSqdPT2Integration:
    """Test PT2 selection integrated into run_hi_nqs_sqd."""

    @pytest.fixture()
    def minimal_mol_info(self, h2_hamiltonian):
        ham = h2_hamiltonian
        return {
            "n_orbitals": ham.integrals.n_orbitals,
            "n_alpha": ham.integrals.n_alpha,
            "n_beta": ham.integrals.n_beta,
            "n_qubits": 2 * ham.integrals.n_orbitals,
        }

    def test_pt2_off_backward_compat(self, h2_hamiltonian, minimal_mol_info):
        """use_pt2_selection=False should work identically to before."""
        from unittest.mock import patch

        from qvartools.methods.nqs.hi_nqs_sqd import run_hi_nqs_sqd

        cfg = HINQSSQDConfig(
            n_iterations=1,
            n_samples_per_iter=10,
            n_batches=1,
            nqs_train_epochs=1,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
            use_pt2_selection=False,
        )

        mock_return = (-1.0, np.array([1.0]), (np.array([0.5]), np.array([0.5])))
        with (
            patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False),
            patch(
                "qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion",
                return_value=mock_return,
            ),
        ):
            result = run_hi_nqs_sqd(h2_hamiltonian, minimal_mol_info, config=cfg)
        assert result.method == "HI+NQS+SQD"
        assert isinstance(result.energy, float)

    def test_pt2_code_path_exists(self, h2_hamiltonian, minimal_mol_info):
        """Verify the PT2 filtering code path is wired in run_hi_nqs_sqd.

        We verify by checking that `compute_pt2_scores` is imported and
        referenced in the module, and that the conditional gate
        `cfg.use_pt2_selection` controls it. Direct invocation is hard to
        test with mocks on small systems (H2 has only 16 configs, so
        unique_new quickly becomes 0). Full end-to-end PT2 verification
        is done in integration tests with larger molecules.
        """
        import qvartools.methods.nqs.hi_nqs_sqd as mod

        # compute_pt2_scores must be imported into the module
        assert hasattr(mod, "compute_pt2_scores"), (
            "compute_pt2_scores not imported in hi_nqs_sqd"
        )
        assert hasattr(mod, "evict_by_coefficient"), (
            "evict_by_coefficient not imported in hi_nqs_sqd"
        )
        assert hasattr(mod, "compute_temperature"), (
            "compute_temperature not imported in hi_nqs_sqd"
        )

        # Verify the gate is in the source code
        import inspect

        source = inspect.getsource(mod.run_hi_nqs_sqd)
        assert "use_pt2_selection" in source
        assert "compute_pt2_scores" in source
        assert "evict_by_coefficient" in source
        assert "compute_temperature" in source

    def test_pt2_on_uses_temperature_annealing(self, h2_hamiltonian, minimal_mol_info):
        """When PT2 is on, temperature should anneal from initial to final."""
        from unittest.mock import patch

        from qvartools.methods.nqs.hi_nqs_sqd import run_hi_nqs_sqd

        cfg = HINQSSQDConfig(
            n_iterations=2,
            n_samples_per_iter=10,
            n_batches=1,
            nqs_train_epochs=1,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
            use_pt2_selection=True,
            initial_temperature=2.0,
            final_temperature=0.5,
        )

        torch.manual_seed(42)
        mock_return = (-1.0, np.array([1.0]), (np.array([0.5]), np.array([0.5])))
        with (
            patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False),
            patch(
                "qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion",
                return_value=mock_return,
            ),
        ):
            result = run_hi_nqs_sqd(h2_hamiltonian, minimal_mol_info, config=cfg)

        # Smoke test: verify PT2 with temperature annealing runs without error.
        # Verifying actual temperature values would require patching
        # AutoregressiveTransformer.sample, which is fragile on small systems.
        assert result.method == "HI+NQS+SQD"


# ---------------------------------------------------------------------------
# P3b: E_PT2 integration tests
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
class TestEPT2Integration:
    """Test E_PT2 correction integrated into run_hi_nqs_sqd."""

    @pytest.fixture()
    def minimal_mol_info(self, h2_hamiltonian):
        ham = h2_hamiltonian
        return {
            "n_orbitals": ham.integrals.n_orbitals,
            "n_alpha": ham.integrals.n_alpha,
            "n_beta": ham.integrals.n_beta,
            "n_qubits": 2 * ham.integrals.n_orbitals,
        }

    def test_config_has_compute_pt2_correction_field(self):
        """HINQSSQDConfig should have compute_pt2_correction field."""
        cfg = HINQSSQDConfig()
        assert hasattr(cfg, "compute_pt2_correction")
        assert cfg.compute_pt2_correction is False  # opt-in

    def test_result_metadata_contains_e_pt2(self, h2_hamiltonian, minimal_mol_info):
        """When compute_pt2_correction=True, metadata should have e_pt2."""
        from unittest.mock import patch

        from qvartools.methods.nqs.hi_nqs_sqd import run_hi_nqs_sqd

        cfg = HINQSSQDConfig(
            n_iterations=2,
            n_samples_per_iter=10,
            n_batches=1,
            nqs_train_epochs=1,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
            compute_pt2_correction=True,
        )

        # Mock must return coeffs matching batch size
        def mock_solver(batch_configs, hamiltonian):
            n = batch_configs.shape[0]
            coeffs = np.zeros(n)
            coeffs[0] = 1.0
            return (-1.0, coeffs, (np.array([0.5]), np.array([0.5])))

        with (
            patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False),
            patch(
                "qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion",
                side_effect=mock_solver,
            ),
        ):
            result = run_hi_nqs_sqd(h2_hamiltonian, minimal_mol_info, config=cfg)

        assert "e_pt2" in result.metadata
        assert "corrected_energy" in result.metadata
        assert isinstance(result.metadata["e_pt2"], float)
        assert isinstance(result.metadata["corrected_energy"], float)

    def test_e_pt2_off_has_no_correction(self, h2_hamiltonian, minimal_mol_info):
        """When compute_pt2_correction=False, metadata should NOT have e_pt2."""
        from unittest.mock import patch

        from qvartools.methods.nqs.hi_nqs_sqd import run_hi_nqs_sqd

        cfg = HINQSSQDConfig(
            n_iterations=1,
            n_samples_per_iter=10,
            n_batches=1,
            nqs_train_epochs=1,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
            compute_pt2_correction=False,
        )

        mock_return = (-1.0, np.array([1.0]), (np.array([0.5]), np.array([0.5])))
        with (
            patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False),
            patch(
                "qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion",
                return_value=mock_return,
            ),
        ):
            result = run_hi_nqs_sqd(h2_hamiltonian, minimal_mol_info, config=cfg)

        assert "e_pt2" not in result.metadata

    def test_metadata_contains_pt2_e0_and_wall_time(
        self, h2_hamiltonian, minimal_mol_info
    ):
        """Metadata should include pt2_e0 (full-basis energy) and wall time."""
        from unittest.mock import patch

        from qvartools.methods.nqs.hi_nqs_sqd import run_hi_nqs_sqd

        cfg = HINQSSQDConfig(
            n_iterations=2,
            n_samples_per_iter=10,
            n_batches=1,
            nqs_train_epochs=1,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
            compute_pt2_correction=True,
        )

        def mock_solver(batch_configs, hamiltonian):
            n = batch_configs.shape[0]
            coeffs = np.zeros(n)
            coeffs[0] = 1.0
            return (-1.0, coeffs, (np.array([0.5]), np.array([0.5])))

        with (
            patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False),
            patch(
                "qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion",
                side_effect=mock_solver,
            ),
        ):
            result = run_hi_nqs_sqd(h2_hamiltonian, minimal_mol_info, config=cfg)

        assert "pt2_e0" in result.metadata, "metadata missing pt2_e0"
        assert "pt2_wall_time" in result.metadata, "metadata missing pt2_wall_time"
        assert isinstance(result.metadata["pt2_e0"], float)
        assert result.metadata["pt2_wall_time"] >= 0

    def test_corrected_energy_uses_pt2_e0(self, h2_hamiltonian, minimal_mol_info):
        """corrected_energy must equal pt2_e0 + e_pt2, NOT best_energy + e_pt2."""
        from unittest.mock import patch

        from qvartools.methods.nqs.hi_nqs_sqd import run_hi_nqs_sqd

        cfg = HINQSSQDConfig(
            n_iterations=2,
            n_samples_per_iter=10,
            n_batches=1,
            nqs_train_epochs=1,
            embed_dim=16,
            n_heads=2,
            n_layers=1,
            compute_pt2_correction=True,
        )

        def mock_solver(batch_configs, hamiltonian):
            n = batch_configs.shape[0]
            coeffs = np.zeros(n)
            coeffs[0] = 1.0
            return (-1.0, coeffs, (np.array([0.5]), np.array([0.5])))

        with (
            patch("qvartools.methods.nqs.hi_nqs_sqd._IBM_SQD_AVAILABLE", False),
            patch(
                "qvartools.methods.nqs.hi_nqs_sqd.gpu_solve_fermion",
                side_effect=mock_solver,
            ),
        ):
            result = run_hi_nqs_sqd(h2_hamiltonian, minimal_mol_info, config=cfg)

        if "e_pt2" in result.metadata:
            expected = result.metadata["pt2_e0"] + result.metadata["e_pt2"]
            assert abs(result.metadata["corrected_energy"] - expected) < 1e-12, (
                f"corrected_energy ({result.metadata['corrected_energy']}) != "
                f"pt2_e0 ({result.metadata['pt2_e0']}) + e_pt2 ({result.metadata['e_pt2']})"
            )


# ---------------------------------------------------------------------------
# P4: CIPSI sparse fallback
# ---------------------------------------------------------------------------


@pytest.mark.pyscf
class TestCIPSISparsefallback:
    """Test CIPSI uses sparse diag when basis exceeds threshold."""

    def test_cipsi_uses_sparse_for_large_basis(self, h2_hamiltonian, monkeypatch):
        """When basis > threshold, CIPSI should call build_sparse_hamiltonian."""
        from unittest.mock import MagicMock, patch

        from qvartools.solvers.subspace.cipsi import CIPSISolver

        original_build = h2_hamiltonian.build_sparse_hamiltonian
        spy = MagicMock(side_effect=original_build)
        monkeypatch.setattr(h2_hamiltonian, "build_sparse_hamiltonian", spy)

        with patch("qvartools.solvers.subspace.cipsi._SPARSE_DIAG_THRESHOLD", 1):
            solver = CIPSISolver(max_iterations=2, expansion_size=2)
            result = solver.solve(h2_hamiltonian, {"name": "test"})

        assert result.energy is not None
        assert result.method == "CIPSI"
        assert spy.called, (
            "build_sparse_hamiltonian was not called when basis > threshold"
        )

    def test_cipsi_sparse_energy_matches_dense(self, h2_hamiltonian):
        """Sparse and dense CIPSI paths should give the same energy on H2."""
        from unittest.mock import patch

        from qvartools.solvers.subspace.cipsi import CIPSISolver

        # Dense path (default threshold = 10K, H2 basis always < 10K)
        solver = CIPSISolver(max_iterations=3, expansion_size=5)
        dense_result = solver.solve(h2_hamiltonian, {"name": "test"})

        # Sparse path (force threshold to 1)
        with patch("qvartools.solvers.subspace.cipsi._SPARSE_DIAG_THRESHOLD", 1):
            sparse_result = solver.solve(h2_hamiltonian, {"name": "test"})

        assert dense_result.energy is not None
        assert sparse_result.energy is not None
        assert abs(dense_result.energy - sparse_result.energy) < 1e-8, (
            f"Dense ({dense_result.energy}) vs Sparse ({sparse_result.energy}) "
            f"differ by {abs(dense_result.energy - sparse_result.energy):.2e}"
        )
