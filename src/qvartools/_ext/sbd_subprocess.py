"""Subprocess-based wrapper for r-ccs-cms/sbd GPU-native diagonalization.

Phase 1 integration: calls sbd as an external process via FCIDUMP files.
This will be replaced by nanobind bindings in Phase 2 (ADR-003).

Usage::

    from qvartools._ext.sbd_subprocess import sbd_diagonalize
    energy = sbd_diagonalize(
        h1e, h2e, n_orb, n_elec, nuclear_repulsion,
        alpha_strings, beta_strings,
    )
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _find_sbd_binary() -> str | None:
    """Find the sbd diag binary."""
    env_binary = os.environ.get("QVARTOOLS_SBD_BINARY", "")
    if env_binary and os.path.isfile(env_binary) and os.access(env_binary, os.X_OK):
        return env_binary
    for path in [
        os.path.expanduser("~/.local/bin/sbd-diag"),
        "/usr/local/bin/sbd-diag",
    ]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _find_mpirun() -> str | None:
    """Find mpirun executable."""
    return shutil.which("mpirun")


def sbd_available() -> bool:
    """Check if sbd binary and mpirun are both available."""
    return _find_sbd_binary() is not None and _find_mpirun() is not None


def _write_fcidump(
    path: str,
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_orb: int,
    n_elec: int,
    nuclear_repulsion: float,
    ms2: int = 0,
) -> None:
    """Write molecular integrals in FCIDUMP format.

    Uses ``pyscf.tools.fcidump`` when available (fastest, handles all
    symmetry reduction internally).  Falls back to a vectorised numpy
    implementation that avoids the O(n⁴) Python loop.
    """
    try:
        from pyscf.tools import fcidump as pyscf_fcidump

        pyscf_fcidump.from_integrals(
            path, h1e, h2e, n_orb, n_elec, nuc=nuclear_repulsion, ms=ms2
        )
        return
    except ImportError:
        pass

    # Vectorised fallback: generate all symmetry-reduced index pairs at once
    with open(path, "w") as f:
        f.write(f" &FCI NORB={n_orb:3d},NELEC={n_elec},MS2={ms2},\n")
        f.write("  ORBSYM=" + ",".join(["1"] * n_orb) + ",\n")
        f.write("  ISYM=1,\n")
        f.write(" &END\n")

        # Two-electron integrals: (p,q) with q<=p, (r,s) with s<=r, pq>=rs
        pq_p, pq_q = np.tril_indices(n_orb)
        pq_idx = pq_p * (pq_p + 1) // 2 + pq_q
        n_pairs = len(pq_p)

        pq_all = np.repeat(np.arange(n_pairs), n_pairs)
        rs_all = np.tile(np.arange(n_pairs), n_pairs)
        mask = pq_idx[pq_all] >= pq_idx[rs_all]
        pq_sel, rs_sel = pq_all[mask], rs_all[mask]

        p, q = pq_p[pq_sel], pq_q[pq_sel]
        r, s = pq_p[rs_sel], pq_q[rs_sel]
        vals = h2e[p, q, r, s]

        nz = np.abs(vals) > 1e-12
        for v, pi, qi, ri, si in zip(
            vals[nz], p[nz] + 1, q[nz] + 1, r[nz] + 1, s[nz] + 1
        ):
            f.write(f"{v:23.16e} {pi:4d} {qi:4d} {ri:4d} {si:4d}\n")

        # One-electron integrals: q<=p
        h1_p, h1_q = np.tril_indices(n_orb)
        h1_vals = h1e[h1_p, h1_q]
        h1_nz = np.abs(h1_vals) > 1e-12
        for v, pi, qi in zip(h1_vals[h1_nz], h1_p[h1_nz] + 1, h1_q[h1_nz] + 1):
            f.write(f"{v:23.16e} {pi:4d} {qi:4d} {0:4d} {0:4d}\n")

        f.write(f"{nuclear_repulsion:23.16e} {0:4d} {0:4d} {0:4d} {0:4d}\n")


def _write_bitstrings(path: str, strings: torch.Tensor) -> None:
    """Write occupation strings as bitstring file (right-to-left ordering)."""
    arr = strings.detach().cpu().numpy()
    with open(path, "w") as f:
        for row in arr:
            bits = "".join(str(int(b)) for b in row[::-1])
            f.write(bits + "\n")


def sbd_diagonalize(
    h1e: np.ndarray,
    h2e: np.ndarray,
    n_orb: int,
    n_elec: int,
    nuclear_repulsion: float,
    alpha_strings: torch.Tensor,
    beta_strings: torch.Tensor,
    *,
    ms2: int = 0,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
    n_threads: int = 4,
    timeout: int = 600,
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
    ms2 : int, optional
        Twice the spin projection (default ``0`` for singlet).
    tolerance : float, optional
        Davidson convergence tolerance.
    max_iterations : int, optional
        Maximum Davidson iterations.
    n_threads : int, optional
        Number of OpenMP threads per MPI rank (default ``4``).
        Passed via ``mpirun -x OMP_NUM_THREADS``.  The subprocess
        runs with a single MPI rank (``-np 1``).
    timeout : int, optional
        Subprocess timeout in seconds (default ``600``).

    Returns
    -------
    float
        Ground state energy (electronic + nuclear repulsion).

    Raises
    ------
    ValueError
        If integral shapes are inconsistent with ``n_orb``, or if
        ``alpha_strings``/``beta_strings`` have wrong rank or column count.
    RuntimeError
        If the sbd binary or mpirun is not found, the subprocess fails,
        or the energy cannot be parsed from the output.
    """
    # --- Input validation ---
    h1e = np.asarray(h1e)
    h2e = np.asarray(h2e)
    if h1e.ndim != 2 or h1e.shape != (n_orb, n_orb):
        raise ValueError(f"h1e must have shape ({n_orb}, {n_orb}), got {h1e.shape}")
    if h2e.ndim != 4 or h2e.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError(
            f"h2e must have shape ({n_orb}, {n_orb}, {n_orb}, {n_orb}), got {h2e.shape}"
        )
    for name, arr in [("alpha_strings", alpha_strings), ("beta_strings", beta_strings)]:
        if arr.ndim != 2 or arr.shape[1] != n_orb:
            raise ValueError(
                f"{name} must have shape (n_strings, {n_orb}), got {tuple(arr.shape)}"
            )
        if arr.is_floating_point():
            raise ValueError(f"{name} must be integer dtype, got {arr.dtype}")

    binary = _find_sbd_binary()
    if binary is None:
        raise RuntimeError(
            "sbd binary not found. Set QVARTOOLS_SBD_BINARY env var "
            "or compile from https://github.com/r-ccs-cms/sbd"
        )
    mpirun = _find_mpirun()
    if mpirun is None:
        raise RuntimeError("mpirun not found in PATH. Install OpenMPI.")

    with tempfile.TemporaryDirectory(prefix="qvartools_sbd_") as tmpdir:
        fcidump_path = os.path.join(tmpdir, "fcidump.txt")
        alpha_path = os.path.join(tmpdir, "alpha.txt")
        beta_path = os.path.join(tmpdir, "beta.txt")

        _write_fcidump(
            fcidump_path, h1e, h2e, n_orb, n_elec, nuclear_repulsion, ms2=ms2
        )
        _write_bitstrings(alpha_path, alpha_strings)
        _write_bitstrings(beta_path, beta_strings)

        cmd = [mpirun]
        if hasattr(os, "getuid") and os.getuid() == 0:
            cmd.append("--allow-run-as-root")
        cmd += [
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
        ]

        logger.debug("sbd command: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            raise RuntimeError(
                f"sbd failed (exit {result.returncode})\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stdout: {(result.stdout or '')[:500]}\n"
                f"Stderr: {(result.stderr or '')[:500]}"
            )

        # Parse the LAST "Energy =" line (final result, not intermediate)
        # Take only the first token after "Energy =" to tolerate units/extra text
        energy = None
        for line in result.stdout.splitlines():
            if "Energy =" in line:
                field = line.split("Energy =", 1)[1].strip()
                if not field:
                    continue
                try:
                    energy = float(field.split()[0])
                except ValueError:
                    continue
        if energy is not None:
            if not math.isfinite(energy):
                raise RuntimeError(f"sbd returned non-finite energy: {energy}")
            return energy

        raise RuntimeError(
            f"Could not parse energy from sbd output:\n{result.stdout[:1000]}"
        )
