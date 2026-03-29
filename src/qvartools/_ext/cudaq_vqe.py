"""CUDA-QX VQE and ADAPT-VQE pipeline wrapper.

Provides GPU-accelerated VQE using NVIDIA CUDA-Q + CUDA-QX Solvers.
Supports VQE with UCCSD ansatz and ADAPT-VQE with spin-complement GSD
operator pool.

Usage::

    from qvartools._ext.cudaq_vqe import run_cudaq_vqe
    result = run_cudaq_vqe(
        geometry=[("H", (0., 0., 0.)), ("H", (0., 0., 0.7474))],
        basis="sto-3g",
        method="adapt-vqe",
    )
    print(result["energy"], result["error_mha"])

Requires: cudaq >= 0.13, cudaq-solvers >= 0.5
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)

_VALID_METHODS = {"vqe", "adapt-vqe"}


def run_cudaq_vqe(
    geometry: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    method: str = "vqe",
    optimizer: str = "cobyla",
    max_iterations: int = 200,
    target: str = "nvidia",
    verbose: bool = False,
) -> dict[str, Any]:
    """Run VQE or ADAPT-VQE using CUDA-QX Solvers on GPU.

    Parameters
    ----------
    geometry : list of (str, (float, float, float))
        Molecular geometry in Angstroms.
    basis : str
        Gaussian basis set (default ``"sto-3g"``).
    charge : int
        Net charge (default ``0``).
    spin : int
        Number of unpaired electrons, i.e. ``2S`` (default ``0`` for
        singlet).  This matches PySCF convention: 0 = singlet,
        1 = doublet, 2 = triplet.
    method : str
        ``"vqe"`` for VQE-UCCSD or ``"adapt-vqe"`` for ADAPT-VQE.
    optimizer : str
        Optimizer name (default ``"cobyla"``).  Applied to both VQE
        and ADAPT-VQE.
    max_iterations : int
        Maximum optimizer iterations.
    target : str
        CUDA-Q simulator target (default ``"nvidia"`` for GPU).
        Use ``"qpp-cpu"`` for CPU-only environments.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``energy`` : float — final VQE/ADAPT-VQE energy (Ha).
        ``fci_energy`` : float or None — CASCI FCI reference.
        ``hf_energy`` : float or None — Hartree-Fock reference.
        ``error_mha`` : float or None — error vs FCI (mHa).
        ``wall_time`` : float — wall-clock time (s).
        ``n_params`` : int — number of optimized parameters.
        ``iterations`` : int — optimizer iterations.
        ``method`` : str — ``"vqe"`` or ``"adapt-vqe"``.
        ``n_qubits`` : int — problem qubit count.
        ``n_electrons`` : int — active-space electron count.
        ``optimal_parameters`` : list[float] — optimal variational params.

    Raises
    ------
    ValueError
        If *method* is not ``"vqe"`` or ``"adapt-vqe"``.
    RuntimeError
        If VQE fails to converge (energy is NaN or inf).

    Notes
    -----
    ``fci_energy`` is the CASCI FCI energy within the active space,
    not the full-space FCI.  For small molecules with minimal basis
    (e.g. H2/sto-3g), the active space equals the full space.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    import cudaq
    import cudaq_solvers as solvers

    try:
        cudaq.set_target(target)
    except RuntimeError:
        if verbose:
            logger.warning("Failed to set target '%s', falling back to qpp-cpu", target)
        cudaq.set_target("qpp-cpu")

    molecule = solvers.create_molecule(
        geometry=geometry,
        basis=basis,
        spin=spin,
        charge=charge,
        casci=True,
    )

    n_qubits: int = molecule.n_orbitals * 2
    n_electrons: int = molecule.n_electrons
    hamiltonian = molecule.hamiltonian

    hf_energy: float | None = molecule.energies.get("hf_energy", None)
    fci_energy: float | None = molecule.energies.get("fci_energy", None)

    logger.info(
        "CUDA-QX %s: %d qubits, %d electrons, basis=%s",
        method,
        n_qubits,
        n_electrons,
        basis,
    )

    t0 = time.perf_counter()

    if method == "adapt-vqe":
        energy, params, n_params, iterations = _run_adapt_vqe(
            molecule=molecule,
            n_electrons=n_electrons,
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    else:
        energy, params, n_params, iterations = _run_vqe_uccsd(
            n_qubits=n_qubits,
            n_electrons=n_electrons,
            spin=spin,
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    wall_time = time.perf_counter() - t0

    if not math.isfinite(energy):
        raise RuntimeError(
            f"{method} failed to converge: energy={energy}. "
            f"Try increasing max_iterations or using a different optimizer."
        )

    error_mha: float | None = (
        abs(energy - fci_energy) * 1000 if fci_energy is not None else None
    )

    return {
        "energy": energy,
        "fci_energy": fci_energy,
        "hf_energy": hf_energy,
        "error_mha": error_mha,
        "wall_time": wall_time,
        "n_params": n_params,
        "iterations": iterations,
        "method": method,
        "n_qubits": n_qubits,
        "n_electrons": n_electrons,
        "optimal_parameters": params,
    }


def _run_vqe_uccsd(
    n_qubits: int,
    n_electrons: int,
    spin: int,
    hamiltonian: Any,
    optimizer: str,
    max_iterations: int,
    verbose: bool,
) -> tuple[float, list[float], int, int]:
    """VQE with UCCSD ansatz."""
    import cudaq
    import cudaq_solvers as solvers

    num_params = solvers.stateprep.get_num_uccsd_parameters(n_electrons, n_qubits)

    _nq = n_qubits
    _ne = n_electrons
    _spin = spin

    @cudaq.kernel
    def uccsd_kernel(thetas: list[float]):
        q = cudaq.qvector(_nq)
        for i in range(_ne):
            x(q[i])
        solvers.stateprep.uccsd(q, thetas, _ne, _spin)

    energy, params, data = solvers.vqe(
        uccsd_kernel,
        hamiltonian,
        initial_parameters=[0.0] * num_params,
        optimizer=optimizer,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    return energy, list(params), num_params, len(data)


def _run_adapt_vqe(
    molecule: Any,
    n_electrons: int,
    hamiltonian: Any,
    optimizer: str,
    max_iterations: int,
    verbose: bool,
) -> tuple[float, list[float], int, int]:
    """ADAPT-VQE with spin-complement GSD operator pool."""
    import cudaq
    import cudaq_solvers as solvers

    operators = solvers.get_operator_pool(
        "spin_complement_gsd",
        num_orbitals=molecule.n_orbitals,
    )

    @cudaq.kernel
    def initial_state(q: cudaq.qview):
        for i in range(n_electrons):
            x(q[i])

    energy, params, ops = solvers.adapt_vqe(
        initial_state,
        hamiltonian,
        operators,
        optimizer=optimizer,
        max_iter=max_iterations,
        verbose=verbose,
    )

    return energy, list(params), len(params), len(ops)
