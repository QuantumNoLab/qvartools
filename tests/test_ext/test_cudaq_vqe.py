"""Tests for CUDA-QX VQE pipeline integration."""

from __future__ import annotations

import pytest

cudaq = pytest.importorskip("cudaq")
cudaq_solvers = pytest.importorskip("cudaq_solvers")

H2_GEOMETRY = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]
# Use CPU simulator in tests for portability (no GPU required)
_TEST_TARGET = "qpp-cpu"


class TestCudaqVQE:
    """Tests for run_cudaq_vqe."""

    def test_import(self) -> None:
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        assert callable(run_cudaq_vqe)

    def test_h2_vqe_uccsd_reaches_chemical_accuracy(self) -> None:
        """H2 VQE-UCCSD should reach chemical accuracy (< 1.6 mHa)."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        result = run_cudaq_vqe(
            geometry=H2_GEOMETRY, basis="sto-3g", method="vqe", target=_TEST_TARGET
        )
        assert result["energy"] < -1.13
        assert result["error_mha"] < 1.6

    def test_h2_adapt_vqe(self) -> None:
        """H2 ADAPT-VQE should also reach chemical accuracy."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        result = run_cudaq_vqe(
            geometry=H2_GEOMETRY,
            basis="sto-3g",
            method="adapt-vqe",
            target=_TEST_TARGET,
        )
        assert result["energy"] < -1.13
        assert result["error_mha"] < 1.6

    def test_returns_expected_keys(self) -> None:
        """Result dict should have standard keys."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        result = run_cudaq_vqe(
            geometry=H2_GEOMETRY, basis="sto-3g", method="vqe", target=_TEST_TARGET
        )
        for key in [
            "energy",
            "fci_energy",
            "error_mha",
            "wall_time",
            "n_params",
            "iterations",
            "method",
            "n_qubits",
            "n_electrons",
            "optimal_parameters",
            "hf_energy",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_invalid_method_raises(self) -> None:
        """Invalid method should raise ValueError."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        with pytest.raises(ValueError, match="method must be one of"):
            run_cudaq_vqe(geometry=H2_GEOMETRY, basis="sto-3g", method="invalid")

    def test_method_field_matches_input(self) -> None:
        """Result method field should match the input method."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        for method in ["vqe", "adapt-vqe"]:
            result = run_cudaq_vqe(
                geometry=H2_GEOMETRY, basis="sto-3g", method=method, target=_TEST_TARGET
            )
            assert result["method"] == method

    def test_vqe_energy_below_hf(self) -> None:
        """VQE energy should be lower than HF energy (variational principle)."""
        from qvartools._ext.cudaq_vqe import run_cudaq_vqe

        result = run_cudaq_vqe(
            geometry=H2_GEOMETRY, basis="sto-3g", method="vqe", target=_TEST_TARGET
        )
        assert result["energy"] < result["hf_energy"]
