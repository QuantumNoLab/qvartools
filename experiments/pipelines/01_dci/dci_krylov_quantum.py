"""Quantum Circuit SKQD --- Direct-CI (HF+S+D) -> Trotterized exp_pauli Krylov.

Pipeline: Jordan-Wigner mapping -> QuantumCircuitSKQD -> Trotterized
Krylov state generation -> cumulative basis sampling -> diagonalise.

Uses Direct-CI (HF + singles + doubles) as the initial basis with
quantum circuit Krylov evolution via Trotterized exp_pauli.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config

from qvartools.methods.quantum_circuit.molecular import (
    QuantumSKQDMethodConfig,
    run_quantum_skqd,
)
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        "Quantum Circuit SKQD: Direct-CI (HF+S+D) -> Trotterized Krylov."
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

    # --- Build config ---
    method_config = QuantumSKQDMethodConfig(
        max_krylov_dim=config.get("max_krylov_dim", 12),
        total_evolution_time=config.get("total_evolution_time", np.pi),
        num_trotter_steps=config.get("num_trotter_steps", 1),
        trotter_order=config.get("trotter_order", 2),
        shots=config.get("shots", 100_000),
        backend=config.get("backend", "auto"),
        use_gpu=device != "cpu",
        device=device,
    )

    # --- Run pipeline ---
    result = run_quantum_skqd(hamiltonian, mol_info, config=method_config)

    # --- Results ---
    final_energy = result.energy
    error_mha = (
        (final_energy - exact_energy) * 1000.0 if exact_energy is not None else None
    )
    within = (
        "YES"
        if (error_mha is not None and abs(error_mha) < CHEMICAL_ACCURACY_MHA)
        else ("NO" if error_mha is not None else "N/A")
    )
    wall_time = result.wall_time

    print(f"\n{'=' * 60}")
    print("QUANTUM CIRCUIT SKQD RESULTS (Direct-CI -> Trotterized Krylov)")
    print(f"{'=' * 60}")

    energies = result.metadata.get("energies_per_krylov", [])
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
    print(f"Basis size   : {result.diag_dim}")
    print(f"Backend      : {result.metadata.get('backend', '?')}")
    print(f"Wall time    : {wall_time:.2f} s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
