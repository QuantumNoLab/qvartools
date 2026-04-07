"""Quantum Circuit SKQD --- HF-only -> Trotterized exp_pauli Krylov.

Pipeline: Uses only the HF reference state as the initial state for
quantum circuit Krylov evolution. No NF training, no DCI scaffolding.
The QuantumCircuitSKQD prepares the HF state and evolves it via
Trotterized exp_pauli circuits to build the Krylov subspace.

Uses: QuantumCircuitSKQD.from_molecular_hamiltonian() with
QuantumSKQDConfig(initial_state="hf").
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config

from qvartools.krylov.circuits.circuit_skqd import QuantumCircuitSKQD, QuantumSKQDConfig
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        "Quantum Circuit SKQD: HF-only -> Trotterized Krylov evolution."
    )
    parser.add_argument(
        "--max-krylov-dim",
        type=int,
        default=None,
        help="Maximum Krylov subspace dimension.",
    )
    parser.add_argument(
        "--num-trotter-steps",
        type=int,
        default=None,
        help="Trotter steps per evolution U.",
    )
    parser.add_argument(
        "--trotter-order",
        type=int,
        default=None,
        help="Suzuki-Trotter order (1 or 2).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Measurement shots per Krylov state.",
    )
    parser.add_argument(
        "--total-evolution-time",
        type=float,
        default=None,
        help="Total evolution time T for U = e^{-iHT}.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["auto", "cudaq", "classical", "exact", "lanczos"],
        help="Sampling backend.",
    )
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(config.get("molecule", "h2"), device=device)
    n_qubits = mol_info["n_qubits"]
    print(f"Molecule : {mol_info['name']}")
    print(f"Qubits   : {n_qubits}")
    print(f"Basis set: {mol_info['basis']}")
    print(f"Device   : {device}")
    print("=" * 60)

    # --- Exact energy ---
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    if exact_energy is not None:
        print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    else:
        print("FCI reference unavailable for this system.")
    print("-" * 60)

    # --- Build QuantumSKQDConfig for HF-only initial state ---
    cfg = QuantumSKQDConfig(
        initial_state="hf",
        max_krylov_dim=config.get("max_krylov_dim", 12),
        total_evolution_time=config.get("total_evolution_time", np.pi),
        num_trotter_steps=config.get("num_trotter_steps", 1),
        trotter_order=config.get("trotter_order", 2),
        shots=config.get("shots", 100_000),
        backend=config.get("backend", "auto"),
    )

    print(f"Krylov dim   : {cfg.max_krylov_dim}")
    print(f"Trotter steps: {cfg.num_trotter_steps}")
    print(f"Trotter order: {cfg.trotter_order}")
    print(f"Shots        : {cfg.shots}")
    print(f"Backend      : {cfg.backend}")
    print("-" * 60)

    # --- Build QuantumCircuitSKQD from molecular Hamiltonian ---
    t_start = time.perf_counter()

    qskqd = QuantumCircuitSKQD.from_molecular_hamiltonian(hamiltonian, config=cfg)

    # --- Run Krylov evolution ---
    results = qskqd.run(progress=True)

    wall_time = time.perf_counter() - t_start

    # --- Results summary ---
    final_energy = results["best_energy"]
    error_mha = (
        (final_energy - exact_energy) * 1000.0 if exact_energy is not None else None
    )
    within = (
        "YES"
        if (error_mha is not None and abs(error_mha) < CHEMICAL_ACCURACY_MHA)
        else ("NO" if error_mha is not None else "N/A")
    )

    print(f"\n{'=' * 60}")
    print("QUANTUM CIRCUIT SKQD RESULTS (HF-only -> Trotterized Krylov)")
    print(f"{'=' * 60}")

    energies = results.get("energies", [])
    if energies:
        print("\n  Energy convergence per Krylov dimension:")
        for i, e in enumerate(energies):
            step_err = (e - exact_energy) * 1000.0 if exact_energy is not None else None
            step_str = f"  (error: {step_err:.4f} mHa)" if step_err is not None else ""
            print(f"    k={i + 2:>3}: {e:.10f} Ha{step_str}")

    print(f"\nFinal energy : {final_energy:.10f} Ha")
    if exact_energy is not None:
        print(f"Exact energy : {exact_energy:.10f} Ha")
    else:
        print("Exact energy : N/A")
    if error_mha is not None:
        print(f"Error        : {error_mha:.4f} mHa")
        print(f"Chemical acc.: {within}")
    basis_sizes = results.get("basis_sizes", [])
    print(f"Basis size   : {basis_sizes[-1] if basis_sizes else '?'}")
    print(f"Backend      : {results.get('backend', '?')}")
    print(f"Wall time    : {wall_time:.2f} s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
