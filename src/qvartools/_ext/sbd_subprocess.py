"""Subprocess-based wrapper for r-ccs-cms/sbd GPU-native diagonalization.

Phase 1 integration: calls sbd as an external process via FCIDUMP files.
This will be replaced by nanobind bindings in Phase 2 (ADR-003).

Usage::

    from qvartools._ext.sbd_subprocess import sbd_diagonalize
    energy = sbd_diagonalize(integrals, alpha_strings, beta_strings)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Path to compiled sbd binary (set via env var or auto-detect)
_SBD_BINARY = os.environ.get("QVARTOOLS_SBD_BINARY", "")


def _find_sbd_binary() -> str | None:
    """Find the sbd diag binary."""
    if _SBD_BINARY and os.path.isfile(_SBD_BINARY):
        return _SBD_BINARY
    # Common locations
    for path in [
        "/tmp/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag",
        os.path.expanduser("~/.local/bin/sbd-diag"),
    ]:
        if os.path.isfile(path):
            return path
    return None


def sbd_available() -> bool:
    """Check if sbd binary is available."""
    return _find_sbd_binary() is not None


def _write_fcidump(
    path: str,
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_orb: int,
    n_elec: int,
    nuclear_repulsion: float,
) -> None:
    """Write molecular integrals in FCIDUMP format."""
    with open(path, "w") as f:
        f.write(f" &FCI NORB={n_orb:3d},NELEC={n_elec},MS2=0,\n")
        f.write("  ORBSYM=" + ",".join(["1"] * n_orb) + ",\n")
        f.write("  ISYM=1,\n")
        f.write(" &END\n")
        for p in range(n_orb):
            for q in range(p + 1):
                for r in range(n_orb):
                    for s in range(r + 1):
                        if (p * (p + 1) // 2 + q) >= (r * (r + 1) // 2 + s):
                            val = h2e[p, q, r, s]
                            if abs(val) > 1e-12:
                                f.write(
                                    f"{val:23.16e} {p + 1:4d} {q + 1:4d} {r + 1:4d} {s + 1:4d}\n"
                                )
        for p in range(n_orb):
            for q in range(p + 1):
                val = h1e[p, q]
                if abs(val) > 1e-12:
                    f.write(f"{val:23.16e} {p + 1:4d} {q + 1:4d} {0:4d} {0:4d}\n")
        f.write(f"{nuclear_repulsion:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}\n")


def _write_bitstrings(path: str, strings: torch.Tensor) -> None:
    """Write occupation strings as bitstring file (right-to-left ordering)."""
    with open(path, "w") as f:
        for row in strings:
            bits = "".join(str(int(b)) for b in reversed(row.tolist()))
            f.write(bits + "\n")


def sbd_diagonalize(
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_orb: int,
    n_elec: int,
    nuclear_repulsion: float,
    alpha_strings: torch.Tensor,
    beta_strings: torch.Tensor,
    n_states: int = 1,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
    n_threads: int = 4,
) -> float:
    """Run sbd diagonalization via subprocess.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integrals, shape ``(n_orb, n_orb)``.
    h2e : np.ndarray
        Two-electron integrals, shape ``(n_orb, n_orb, n_orb, n_orb)``.
    n_orb : int
        Number of spatial orbitals.
    n_elec : int
        Total number of electrons.
    nuclear_repulsion : float
        Nuclear repulsion energy.
    alpha_strings : torch.Tensor
        Unique alpha occupation strings, shape ``(n_alpha, n_orb)``.
    beta_strings : torch.Tensor
        Unique beta occupation strings, shape ``(n_beta, n_orb)``.
    n_states : int
        Number of eigenstates to compute.
    tolerance : float
        Davidson convergence tolerance.
    max_iterations : int
        Maximum Davidson iterations.
    n_threads : int
        OpenMP threads.

    Returns
    -------
    float
        Ground state energy (electronic + nuclear repulsion).
    """
    binary = _find_sbd_binary()
    if binary is None:
        raise RuntimeError(
            "sbd binary not found. Set QVARTOOLS_SBD_BINARY env var "
            "or compile from https://github.com/r-ccs-cms/sbd"
        )

    with tempfile.TemporaryDirectory(prefix="qvartools_sbd_") as tmpdir:
        fcidump_path = os.path.join(tmpdir, "fcidump.txt")
        alpha_path = os.path.join(tmpdir, "alpha.txt")
        beta_path = os.path.join(tmpdir, "beta.txt")

        _write_fcidump(fcidump_path, h1e, h2e, n_orb, n_elec, nuclear_repulsion)
        _write_bitstrings(alpha_path, alpha_strings)
        _write_bitstrings(beta_path, beta_strings)

        cmd = [
            "mpirun",
            "--allow-run-as-root",
            "-np",
            "1",
            "-x",
            f"OMP_NUM_THREADS={n_threads}",
            binary,
            "--fcidump",
            fcidump_path,
            "--adetfile",
            alpha_path,
            "--bdetfile",
            beta_path,
            "--method",
            "0",
            "--block",
            "10",
            "--iteration",
            str(max_iterations),
            "--tolerance",
            str(tolerance),
            "--init",
            "0",
            "--rdm",
            "0",
            "--nstate",
            str(n_states),
        ]

        logger.debug("sbd command: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(
                f"sbd failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        # Parse energy from stdout
        for line in result.stdout.splitlines():
            if "Energy =" in line:
                energy_str = line.split("Energy =")[1].strip()
                return float(energy_str)

        raise RuntimeError(
            f"Could not parse energy from sbd output:\n{result.stdout[:1000]}"
        )
