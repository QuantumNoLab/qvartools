"""Pipeline 06b: Iterative NQS warmup + Quantum Circuit Krylov.

Hybrid two-phase pipeline:
  Phase 1 -- Run iterative NQS+SQD for a few warmup iterations to train
             the autoregressive NQS and obtain a good basis via eigenvector
             feedback.
  Phase 2 -- Use QuantumCircuitSKQD (Trotterized time-evolution) as the
             final energy estimator on the molecular Hamiltonian.

This combines the sampling power of iterative NQS with the quantum
Krylov diagonalization circuit for the energy estimate.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

# Make the experiments package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config  # noqa: E402

from qvartools.methods.nqs.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
from qvartools.methods.quantum_circuit.molecular import (
    QuantumSKQDMethodConfig,
    run_quantum_skqd,
)
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # --- Parse CLI / YAML config ---
    parser = create_base_parser(
        "Pipeline 06b: Iterative NQS warmup + Quantum Circuit Krylov."
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=None,
        help="Number of NQS warmup iterations (Phase 1).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="NQS samples per warmup iteration.",
    )
    parser.add_argument(
        "--max-krylov-dim",
        type=int,
        default=None,
        help="Maximum Krylov dimension for quantum circuit phase.",
    )
    parser.add_argument("--verbose", action="store_true", default=None)
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = detect_device()

    warmup_iters = config.get("warmup_iterations", 3)
    n_samp = config.get("n_samples_per_iter", config.get("n_samples", 5000))
    max_krylov = config.get("max_krylov_dim", 12)

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(args.molecule, device=device)
    print(f"Molecule            : {mol_info['name']}")
    print(f"Qubits              : {mol_info['n_qubits']}")
    print(f"Warmup iterations   : {warmup_iters}")
    print(f"Samples/iter        : {n_samp}")
    print(f"Quantum Krylov dim  : {max_krylov}")
    print("=" * 60)

    # --- Augment mol_info with orbital/electron counts ---
    mol_info["n_orbitals"] = hamiltonian.integrals.n_orbitals
    mol_info["n_alpha"] = hamiltonian.integrals.n_alpha
    mol_info["n_beta"] = hamiltonian.integrals.n_beta

    # --- Exact reference ---
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    if exact_energy is not None:
        print(f"Exact (FCI) energy: {exact_energy:.10f} Ha\n")
    else:
        print("FCI reference unavailable for this system.\n")

    # ================================================================
    # Phase 1: Iterative NQS warmup (few iterations to train NQS)
    # ================================================================
    print("=" * 60)
    print("PHASE 1: Iterative NQS warmup (SQD)")
    print("=" * 60)

    warmup_config = HINQSSQDConfig(
        n_iterations=warmup_iters,
        n_samples_per_iter=n_samp,
        nqs_train_epochs=config.get("nqs_train_epochs", 50),
        nqs_lr=config.get("nqs_lr", 1e-3),
        energy_tol=config.get("convergence_tol", config.get("energy_tol", 1e-5)),
        use_ibm_solver=config.get("use_ibm_solver", False),
        device=device,
    )

    t_start = time.perf_counter()
    warmup_result = run_hi_nqs_sqd(hamiltonian, mol_info, config=warmup_config)
    t_warmup = time.perf_counter() - t_start

    warmup_energy = warmup_result.energy
    warmup_error = (
        (warmup_energy - exact_energy) * 1000.0 if exact_energy is not None else None
    )
    print(f"\nNQS warmup energy : {warmup_energy:.10f} Ha")
    if warmup_error is not None:
        print(f"Warmup error      : {warmup_error:.4f} mHa")
    print(f"Warmup wall time  : {t_warmup:.2f} s")

    # Print warmup convergence history
    warmup_energies = warmup_result.metadata.get("energy_history", [])
    if warmup_energies:
        print("\nWarmup iteration-by-iteration convergence:")
        for i, e in enumerate(warmup_energies):
            err = (e - exact_energy) * 1000.0 if exact_energy is not None else None
            err_str = f"  ({err:.4f} mHa)" if err is not None else ""
            print(f"  iter {i + 1:>3}: {e:.10f} Ha{err_str}")

    # ================================================================
    # Phase 2: Quantum circuit Krylov for final energy
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Quantum Circuit Krylov (SKQD)")
    print("=" * 60)

    qskqd_config = QuantumSKQDMethodConfig(
        max_krylov_dim=max_krylov,
        backend=config.get("quantum_backend", "auto"),
        device=device,
    )

    t_q_start = time.perf_counter()
    result = run_quantum_skqd(hamiltonian, mol_info, config=qskqd_config)
    t_quantum = time.perf_counter() - t_q_start
    total_wall_time = time.perf_counter() - t_start

    final_energy = result.energy

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("PIPELINE 06b: ITERATIVE NQS + QUANTUM KRYLOV RESULTS")
    print("=" * 60)
    error_mha = (
        (final_energy - exact_energy) * 1000.0 if exact_energy is not None else None
    )
    within = (
        "YES"
        if (error_mha is not None and abs(error_mha) < CHEMICAL_ACCURACY_MHA)
        else ("NO" if error_mha is not None else "N/A")
    )
    warmup_str = f"{warmup_error:.4f} mHa" if warmup_error is not None else "N/A"
    print(f"Warmup energy : {warmup_energy:.10f} Ha  ({warmup_str})")
    print(f"Final energy  : {final_energy:.10f} Ha")
    if exact_energy is not None:
        print(f"Exact energy  : {exact_energy:.10f} Ha")
    else:
        print("Exact energy  : N/A")
    if error_mha is not None:
        print(f"Error         : {error_mha:.4f} mHa")
        print(f"Chemical acc. : {within} (threshold = {CHEMICAL_ACCURACY_MHA} mHa)")
    print(f"Converged     : {result.converged}")
    print(f"Final basis   : {result.diag_dim}")
    print(f"Warmup time   : {t_warmup:.2f} s")
    print(f"Quantum time  : {t_quantum:.2f} s")
    print(f"Total time    : {total_wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
