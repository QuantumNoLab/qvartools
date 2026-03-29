"""Pipeline 08c: NF training + DCI merge + PT2 expansion -> Iterative NQS + SQD.

Pipeline:
  Stage 1:   NF-NQS training (physics-guided mixed-objective)
  Stage 2:   Diversity-aware basis extraction (merges NF + HF+S+D essentials)
  Stage 2.5: PT2 expansion via Hamiltonian connections
  Stage 3:   Iterative NQS + SQD (batch diag with eigenvector feedback)

Same as Group 03 for stages 1-2.5, then Group 06 iterative NQS+SQD for stage 3.
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
from qvartools.methods.nqs.hi_nqs_sqd import HINQSSQDConfig, run_hi_nqs_sqd
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser(
        "Pipeline 08c: NF + DCI + PT2 -> Iterative NQS + SQD.",
    )
    parser.add_argument("--teacher-weight", type=float, default=None)
    parser.add_argument("--physics-weight", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--samples-per-batch", type=int, default=None)
    parser.add_argument("--pt2-max-new", type=int, default=None)
    parser.add_argument("--pt2-n-ref", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--n-samples-per-iter", type=int, default=None)
    parser.add_argument("--nqs-train-epochs", type=int, default=None)
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
    print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    print("-" * 60)

    train_defaults = get_training_params(n_configs)

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

    nf_pt2_time = time.perf_counter() - t_start

    # === Stage 3: Iterative NQS + SQD ===
    print("\n[Stage 3] Iterative NQS + SQD...")
    mol_info["n_orbitals"] = n_orb
    mol_info["n_alpha"] = n_alpha
    mol_info["n_beta"] = n_beta

    sqd_config = HINQSSQDConfig(
        n_iterations=config.get("max_iterations", 10),
        n_samples_per_iter=config.get("n_samples_per_iter", 5000),
        nqs_train_epochs=config.get("nqs_train_epochs", 50),
        device=device,
    )
    nqs_result = run_hi_nqs_sqd(
        hamiltonian, mol_info, config=sqd_config, initial_basis=basis
    )

    wall_time = time.perf_counter() - t_start

    final_energy = nqs_result.energy
    error_mha = (final_energy - exact_energy) * 1000.0
    within = "YES" if abs(error_mha) < CHEMICAL_ACCURACY_MHA else "NO"

    energy_history = nqs_result.metadata.get("energy_history", [])
    if energy_history:
        print("\n  Iterative NQS+SQD convergence:")
        for i, e in enumerate(energy_history):
            err = (e - exact_energy) * 1000.0
            print(f"    iter {i + 1:>3}: {e:.10f} Ha  (error: {err:.4f} mHa)")

    print("\n" + "=" * 60)
    print("PIPELINE 08c: NF+DCI+PT2 -> ITERATIVE NQS+SQD RESULTS")
    print("=" * 60)
    print(f"NF training      : {n_epochs} epochs")
    print(f"NF+DCI basis     : {basis_before} configs")
    print(f"PT2 expanded     : {basis.shape[0]} configs")
    print(f"NF+PT2 time      : {nf_pt2_time:.1f}s")
    print(f"NQS converged    : {nqs_result.converged}")
    print(f"\nFinal energy : {final_energy:.10f} Ha")
    print(f"Exact energy : {exact_energy:.10f} Ha")
    print(f"Error        : {error_mha:.4f} mHa")
    print(f"Chemical acc.: {within}")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
