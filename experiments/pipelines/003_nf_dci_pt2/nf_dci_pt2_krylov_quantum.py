"""NF + DCI + PT2 -> Quantum Circuit Krylov pipeline.

Pipeline:
  1. NF-NQS training (physics-guided mixed-objective)
  2. Diversity-aware basis extraction (merges NF + HF+S+D essential configs)
  3. PT2 basis expansion via Hamiltonian connections
  4. Quantum circuit SKQD (Trotterized Krylov evolution)

The PT2 expansion step grows the merged basis by following single- and
double-excitation connections from the most important reference
configurations.  The expanded basis is then handed off to the quantum
circuit Krylov engine which applies Trotterized time evolution and
projects the Hamiltonian into the accumulated Krylov subspace.
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
    """Return 'cuda' if available, else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_training_params(n_configs: int) -> dict:
    """Scale training hyperparameters based on Hilbert-space size."""
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # --- Parse CLI / YAML config ---
    parser = create_base_parser(
        "NF + DCI + PT2 -> Quantum Circuit Krylov.",
    )
    parser.add_argument("--teacher-weight", type=float, default=None)
    parser.add_argument("--physics-weight", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--samples-per-batch", type=int, default=None)
    parser.add_argument(
        "--max-krylov-dim",
        type=int,
        default=None,
        help="Maximum Krylov subspace dimension for quantum circuit.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        help="Measurement shots per Krylov state.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Quantum backend: auto, cudaq, classical, exact, lanczos.",
    )
    parser.add_argument(
        "--pt2-max-new",
        type=int,
        default=None,
        help="Max new configurations added by PT2 expansion.",
    )
    parser.add_argument(
        "--pt2-n-ref",
        type=int,
        default=None,
        help="Number of reference configurations for PT2 expansion.",
    )
    parser.add_argument("--verbose", action="store_true", default=None)
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = detect_device()

    # --- Load molecule ---
    hamiltonian, mol_info = get_molecule(config.get("molecule", "h2"), device=device)
    n_qubits = mol_info["n_qubits"]
    print(f"Molecule : {mol_info['name']}")
    print(f"Qubits   : {n_qubits}")
    print(f"Basis set: {mol_info['basis']}")
    print(f"Device   : {device}")
    print("=" * 60)

    # --- Compute Hilbert-space size ---
    n_orb = hamiltonian.integrals.n_orbitals
    n_alpha = hamiltonian.integrals.n_alpha
    n_beta = hamiltonian.integrals.n_beta
    n_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    print(f"Hilbert space: {n_configs:,} configs")

    # --- Compute exact energy for comparison ---
    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    if exact_energy is not None:
        print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    else:
        print("FCI reference unavailable for this system.")
    print("-" * 60)

    # --- Auto-scale defaults, then override with config values ---
    train_defaults = get_training_params(n_configs)

    teacher_weight = config.get("teacher_weight", 0.5)
    physics_weight = config.get("physics_weight", 0.4)
    entropy_weight = config.get("entropy_weight", 0.1)
    max_epochs = config.get("max_epochs", train_defaults["max_epochs"])
    min_epochs = config.get("min_epochs", train_defaults["min_epochs"])
    samples_per_batch = config.get(
        "samples_per_batch", train_defaults["samples_per_batch"]
    )
    nf_hidden_dims = config.get("nf_hidden_dims", train_defaults["nf_hidden_dims"])
    nqs_hidden_dims = config.get("nqs_hidden_dims", train_defaults["nqs_hidden_dims"])

    # --- Stages 1-2: NF training + basis extraction (via pipeline) ---
    pipe_config = PipelineConfig(
        skip_nf_training=False,
        subspace_mode="classical_krylov",
        teacher_weight=teacher_weight,
        physics_weight=physics_weight,
        entropy_weight=entropy_weight,
        device=device,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        samples_per_batch=samples_per_batch,
        nf_hidden_dims=nf_hidden_dims,
        nqs_hidden_dims=nqs_hidden_dims,
    )

    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipe_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    t_start = time.perf_counter()

    # Stage 1: NF-NQS training
    history = pipeline.train_flow_nqs(progress=True)
    n_epochs = len(history.get("total_loss", []))

    # Stage 2: Basis extraction + diversity selection
    basis = pipeline.extract_and_select_basis()
    print(f"NF+DCI merged basis: {basis.shape[0]} configs")

    # --- Stage 2.5: PT2 expansion ---
    pt2_max_new = config.get("pt2_max_new", 500)
    pt2_n_ref = config.get("pt2_n_ref", 10)

    basis_before_pt2 = basis.shape[0]
    basis = expand_basis_via_connections(
        basis,
        hamiltonian,
        max_new=pt2_max_new,
        n_ref=pt2_n_ref,
    )
    print(
        f"PT2 expansion: {basis_before_pt2} -> {basis.shape[0]} configs "
        f"(+{basis.shape[0] - basis_before_pt2})"
    )

    # --- Stage 3: Quantum Circuit Krylov ---
    qskqd_max_krylov = config.get("max_krylov_dim", 12)
    qskqd_shots = config.get("shots", 100_000)
    qskqd_backend = config.get("backend", "auto")

    method_config = QuantumSKQDMethodConfig(
        max_krylov_dim=qskqd_max_krylov,
        shots=qskqd_shots,
        backend=qskqd_backend,
        device=device,
    )

    print(
        f"\nQuantum Krylov: dim={qskqd_max_krylov}, shots={qskqd_shots}, "
        f"backend={qskqd_backend}"
    )

    qskqd_result = run_quantum_skqd(
        hamiltonian,
        mol_info,
        config=method_config,
    )

    wall_time = time.perf_counter() - t_start

    # --- Results summary ---
    print("\n" + "=" * 60)
    print("NF + DCI + PT2 -> QUANTUM CIRCUIT KRYLOV RESULTS")
    print("=" * 60)
    print(f"Stage 1 (NF-NQS training):  {n_epochs} epochs")
    print(f"Stage 2 (Basis selection):  {basis_before_pt2} configs")
    print(f"Stage 2.5 (PT2 expansion): {basis_before_pt2} -> {basis.shape[0]} configs")
    print(f"  PT2 max_new = {pt2_max_new}, n_ref = {pt2_n_ref}")

    final_energy = qskqd_result.energy
    error_mha = (
        (final_energy - exact_energy) * 1000.0 if exact_energy is not None else None
    )

    if qskqd_result.metadata:
        energies_per_step = qskqd_result.metadata.get("energies_per_step")
        if energies_per_step:
            print("\n  Quantum Krylov energy convergence:")
            for i, e in enumerate(energies_per_step):
                label = "k=0" if i == 0 else f"k={i}"
                print(f"    {label:>8}: {e:.10f} Ha")

        basis_sizes = qskqd_result.metadata.get("basis_sizes_per_step")
        if basis_sizes:
            print(f"\n  Basis sizes per step: {basis_sizes}")

    print(f"\nFinal energy : {final_energy:.10f} Ha")
    if exact_energy is not None:
        print(f"Exact energy : {exact_energy:.10f} Ha")
    else:
        print("Exact energy : N/A")
    if error_mha is not None:
        print(f"Error        : {error_mha:.4f} mHa")
        within = (
            "YES"
            if (error_mha is not None and abs(error_mha) < CHEMICAL_ACCURACY_MHA)
            else ("NO" if error_mha is not None else "N/A")
        )
        print(f"Chemical acc.: {within} (threshold = {CHEMICAL_ACCURACY_MHA} mHa)")
    print(f"Diag dim     : {qskqd_result.diag_dim}")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
