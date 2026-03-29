"""Tests for sbd subprocess integration."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("pyscf")


class TestSbdSubprocess:
    """Tests for sbd_diagonalize via subprocess."""

    def test_sbd_available(self) -> None:
        """sbd_available should return True if binary exists."""
        from qvartools._ext.sbd_subprocess import sbd_available

        # May be True or False depending on environment
        assert isinstance(sbd_available(), bool)

    @pytest.mark.skipif(
        not __import__(
            "qvartools._ext.sbd_subprocess", fromlist=["sbd_available"]
        ).sbd_available(),
        reason="sbd binary not found",
    )
    def test_lih_energy_matches_reference(self) -> None:
        """sbd energy on LiH should match our dense diag within 1e-6 Ha."""
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
