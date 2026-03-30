"""Tests for sbd subprocess integration."""

from __future__ import annotations

import os
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest
import torch


class TestSbdSubprocess:
    """Tests for sbd_diagonalize via subprocess."""

    def test_sbd_available(self) -> None:
        """sbd_available should return a bool."""
        from qvartools._ext.sbd_subprocess import sbd_available

        # May be True or False depending on environment
        assert isinstance(sbd_available(), bool)

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not __import__(
            "qvartools._ext.sbd_subprocess", fromlist=["sbd_available"]
        ).sbd_available(),
        reason="sbd binary not found",
    )
    def test_lih_energy_matches_reference(self) -> None:
        """sbd energy on LiH should match our dense diag within 1e-6 Ha."""
        pyscf = pytest.importorskip("pyscf")  # noqa: F841

        from qvartools._ext.sbd_subprocess import sbd_diagonalize
        from qvartools._utils.formatting.bitstring_format import split_spin_strings
        from qvartools.molecules import get_molecule

        ham, mol_info = get_molecule("lih", device="cpu")
        integrals = ham.integrals

        # Generate basis
        from qvartools.krylov.circuits.sqd import SQDConfig, SQDSolver

        solver = SQDSolver(ham, config=SQDConfig())
        basis = solver._generate_essential_configs(torch.device("cpu"))

        alpha, beta = split_spin_strings(basis, n_orbitals=integrals.n_orbitals)

        energy = sbd_diagonalize(
            h1e=integrals.h1e,
            h2e=integrals.h2e,
            n_orb=integrals.n_orbitals,
            n_elec=integrals.n_electrons,
            nuclear_repulsion=integrals.nuclear_repulsion,
            alpha_strings=alpha,
            beta_strings=beta,
        )

        # Our dense diag result for comparison
        result = solver.run(basis, progress=False)
        our_energy = result["energy"]

        # sbd should match within 1e-6 Ha
        assert abs(energy - our_energy) < 1e-6, (
            f"sbd={energy:.10f} vs ours={our_energy:.10f}, "
            f"diff={abs(energy - our_energy):.2e}"
        )


class TestFcidumpFallback:
    """Tests for _write_fcidump numpy fallback vs PySCF path."""

    def test_numpy_fallback_matches_pyscf(self) -> None:
        """FCIDUMP written via numpy fallback must match PySCF output."""
        pyscf = pytest.importorskip("pyscf")  # noqa: F841
        from pyscf.tools import fcidump as pyscf_fcidump

        from qvartools._ext.sbd_subprocess import _write_fcidump

        rng = np.random.default_rng(42)
        n_orb = 3

        # Symmetric h1e
        h1e = rng.standard_normal((n_orb, n_orb))
        h1e = (h1e + h1e.T) / 2

        # Symmetric h2e with 8-fold symmetry (chemist notation)
        h2e = rng.standard_normal((n_orb, n_orb, n_orb, n_orb))
        h2e = (h2e + h2e.transpose(1, 0, 2, 3)) / 2
        h2e = (h2e + h2e.transpose(0, 1, 3, 2)) / 2
        h2e = (h2e + h2e.transpose(2, 3, 0, 1)) / 2

        n_elec = 2
        nuc = 1.23456789

        with tempfile.TemporaryDirectory() as tmpdir:
            # PySCF path (normal import)
            pyscf_path = os.path.join(tmpdir, "pyscf.fcidump")
            _write_fcidump(pyscf_path, h1e, h2e, n_orb, n_elec, nuc)

            # Numpy fallback path: hide pyscf.tools so the inner
            # ``from pyscf.tools import fcidump`` raises ImportError
            numpy_path = os.path.join(tmpdir, "numpy.fcidump")
            with mock.patch.dict(sys.modules, {"pyscf.tools": None}):
                _write_fcidump(numpy_path, h1e, h2e, n_orb, n_elec, nuc)

            # Parse both files with PySCF and compare integrals
            pyscf_data = pyscf_fcidump.read(pyscf_path, verbose=0)
            numpy_data = pyscf_fcidump.read(numpy_path, verbose=0)

            np.testing.assert_allclose(
                pyscf_data["H1"],
                numpy_data["H1"],
                atol=1e-14,
                err_msg="h1e mismatch between PySCF and numpy fallback",
            )
            np.testing.assert_allclose(
                pyscf_data["H2"],
                numpy_data["H2"],
                atol=1e-14,
                err_msg="h2e mismatch between PySCF and numpy fallback",
            )
            assert abs(pyscf_data["ECORE"] - numpy_data["ECORE"]) < 1e-14
