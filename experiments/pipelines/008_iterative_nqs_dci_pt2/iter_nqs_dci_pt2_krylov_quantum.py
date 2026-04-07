"""Pipeline 08b: NF training + DCI merge + PT2 expansion -> Quantum Circuit Krylov.

Pipeline:
  Stage 1:   NF-NQS training (physics-guided mixed-objective)
  Stage 2:   Diversity-aware basis extraction (merges NF + HF+S+D essentials)
  Stage 2.5: PT2 expansion via Hamiltonian connections
  Stage 3:   Quantum circuit Krylov via QuantumCircuitSKQD (Trotterized)

Same as Group 03 for stages 1-2.5, then quantum Krylov for stage 3.
"""

from __future__ import annotations

import logging
import sys
import time
from math import comb
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config  # noqa: E402

from qvartools import FlowGuidedKrylovPipeline, PipelineConfig
from qvartools.krylov.expansion.krylov_expand import expand_basis_via_connections
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


def get_training_params(n_configs: int) -> dict:
    if n_configs <= 10:
        return dict(
            max_epochs=100,
            min_epochs=30,
            samples_per_batch=500,
            nf_hidden_dims=[128, 128],
            nqs_hidden_dims=[128, 128, 128],
        )
    elif n_configs <= 300:
        return dict(
            max_epochs=150,
            min_epochs=50,
            samples_per_batch=1000,
            nf_hidden_dims=[128, 128],
            nqs_hidden_dims=[128, 128, 128],
        )
    elif n_configs <= 2000:
        return dict(
            max_epochs=200,
            min_epochs=80,
            samples_per_batch=1500,
            nf_hidden_dims=[256, 256],
            nqs_hidden_dims=[256, 256, 256],
        )
    elif n_configs <= 5000:
        return dict(
            max_epochs=300,
            min_epochs=100,
            samples_per_batch=2000,
            nf_hidden_dims=[256, 256],
            nqs_hidden_dims=[256, 256, 256],
        )
    else:
        return dict(
            max_epochs=400,
            min_epochs=150,
            samples_per_batch=3000,
            nf_hidden_dims=[512, 512],
            nqs_hidden_dims=[512, 512, 512],
        )


def get_quantum_krylov_params(n_configs: int) -> dict:
    if n_configs <= 300:
        return dict(max_krylov_dim=8, shots=100_000)
    elif n_configs <= 5000:
        return dict(max_krylov_dim=10, shots=200_000)
    else:
        return dict(max_krylov_dim=12, shots=200_000)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        "Pipeline 08b: NF + DCI + PT2 -> Quantum Circuit Krylov.",
    )
    parser.add_argument("--teacher-weight", type=float, default=None)
    parser.add_argument("--physics-weight", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--samples-per-batch", type=int, default=None)
    parser.add_argument("--pt2-max-new", type=int, default=None)
    parser.add_argument("--pt2-n-ref", type=int, default=None)
    parser.add_argument("--max-krylov-dim", type=int, default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = detect_device()

    hamiltonian, mol_info = get_molecule(args.molecule, device=device)
    n_qubits = mol_info["n_qubits"]
    print(f"Molecule : {mol_info['name']}")
    print(f"Qubits   : {n_qubits}")
    print(f"Basis set: {mol_info['basis']}")
    print(f"Device   : {device}")
    print("=" * 60)

    n_orb = hamiltonian.integrals.n_orbitals
    n_alpha = hamiltonian.integrals.n_alpha
    n_beta = hamiltonian.integrals.n_beta
    n_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    if exact_energy is not None:
        print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    else:
        print("FCI reference unavailable for this system.")
    print("-" * 60)

    train_defaults = get_training_params(n_configs)
    qkrylov_defaults = get_quantum_krylov_params(n_configs)

    t_start = time.perf_counter()

    # === Stage 1-2: NF training + DCI merge ===
    print("\n[Stage 1-2] NF-NQS training + DCI merge...")
    pipe_config = PipelineConfig(
        skip_nf_training=False,
        subspace_mode="classical_krylov",
        teacher_weight=config.get("teacher_weight", 0.5),
        physics_weight=config.get("physics_weight", 0.4),
        entropy_weight=config.get("entropy_weight", 0.1),
        device=device,
        max_epochs=config.get("max_epochs", train_defaults["max_epochs"]),
        min_epochs=config.get("min_epochs", train_defaults["min_epochs"]),
        samples_per_batch=config.get(
            "samples_per_batch", train_defaults["samples_per_batch"]
        ),
        nf_hidden_dims=config.get("nf_hidden_dims", train_defaults["nf_hidden_dims"]),
        nqs_hidden_dims=config.get(
            "nqs_hidden_dims", train_defaults["nqs_hidden_dims"]
        ),
    )

    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipe_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    history = pipeline.train_flow_nqs(progress=True)
    n_epochs = len(history.get("total_loss", []))
    basis = pipeline.extract_and_select_basis()

    # === Stage 2.5: PT2 expansion ===
    pt2_max_new = config.get("pt2_max_new", 500)
    pt2_n_ref = config.get("pt2_n_ref", 10)
    basis_before = basis.shape[0]
    basis = expand_basis_via_connections(
        basis,
        hamiltonian,
        max_new=pt2_max_new,
        n_ref=pt2_n_ref,
    )
    print(f"  PT2 expansion: {basis_before} -> {basis.shape[0]} configs")

    nf_time = time.perf_counter() - t_start

    # === Stage 3: Quantum Circuit Krylov ===
    print("\nRunning Quantum Circuit SKQD (Trotterized Krylov evolution)...")
    max_krylov_dim = config.get("max_krylov_dim", qkrylov_defaults["max_krylov_dim"])
    shots = config.get("shots", qkrylov_defaults["shots"])
    backend = config.get("backend", "auto")

    method_config = QuantumSKQDMethodConfig(
        max_krylov_dim=max_krylov_dim,
        shots=shots,
        backend=backend,
        device=device,
    )

    t_quantum = time.perf_counter()
    result = run_quantum_skqd(hamiltonian, mol_info, config=method_config)
    quantum_time = time.perf_counter() - t_quantum
    wall_time = time.perf_counter() - t_start

    final_energy = result.energy
    error_mha = (
        (final_energy - exact_energy) * 1000.0 if exact_energy is not None else None
    )
    within = (
        "YES"
        if (error_mha is not None and abs(error_mha) < CHEMICAL_ACCURACY_MHA)
        else ("NO" if error_mha is not None else "N/A")
    )

    print("\n" + "=" * 60)
    print("PIPELINE 08b: NF+DCI+PT2 -> QUANTUM KRYLOV RESULTS")
    print("=" * 60)
    print(f"NF training      : {n_epochs} epochs")
    print(f"NF+DCI basis     : {basis_before} configs")
    print(f"PT2 expanded     : {basis.shape[0]} configs")
    print(f"Quantum time     : {quantum_time:.2f} s")
    print(f"\nFinal energy : {final_energy:.10f} Ha")
    if exact_energy is not None:
        print(f"Exact energy : {exact_energy:.10f} Ha")
    else:
        print("Exact energy : N/A")
    if error_mha is not None:
        print(f"Error        : {error_mha:.4f} mHa")
        print(f"Chemical acc.: {within}")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
