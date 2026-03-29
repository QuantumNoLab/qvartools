"""Pipeline 07a: NF training + DCI merge -> Iterative NQS + Krylov.

Pipeline:
  Stage 1: NF-NQS training (physics-guided mixed-objective)
  Stage 2: Diversity-aware basis extraction (merges NF + HF+S+D essentials)
  Stage 3: Iterative NQS + Krylov expansion (H-connections) with
           eigenvector feedback, seeded from the NF+DCI merged basis.

Same as Group 02 for stages 1-2, then Group 06 iterative NQS for stage 3.
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
from qvartools.methods.nqs.hi_nqs_skqd import HINQSSKQDConfig, run_hi_nqs_skqd
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def detect_device() -> str:
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

    parser = create_base_parser(
        "Pipeline 07a: NF + DCI merge -> Iterative NQS + Krylov.",
    )
    parser.add_argument("--teacher-weight", type=float, default=None)
    parser.add_argument("--physics-weight", type=float, default=None)
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--samples-per-batch", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--n-samples-per-iter", type=int, default=None)
    parser.add_argument("--nqs-train-epochs", type=int, default=None)
    parser.add_argument("--krylov-max-new", type=int, default=None)
    parser.add_argument("--krylov-n-ref", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = detect_device()

    # --- Load molecule ---
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
    print(f"Hilbert space: {n_configs:,} configs")

    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    print("-" * 60)

    train_defaults = get_training_params(n_configs)

    t_start = time.perf_counter()

    # === Stage 1-2: NF training + DCI merge (same as Group 02) ===
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
    nf_dci_energy = pipeline.results.get("final_energy")

    print(f"  NF training: {n_epochs} epochs")
    print(f"  NF+DCI merged basis: {basis.shape[0]} configs")
    if nf_dci_energy is not None:
        print(f"  NF+DCI energy: {nf_dci_energy:.10f} Ha")

    nf_time = time.perf_counter() - t_start

    # === Stage 3: Iterative NQS + Krylov (same as Group 06) ===
    print("\n[Stage 3] Iterative NQS + Krylov expansion...")
    mol_info["n_orbitals"] = n_orb
    mol_info["n_alpha"] = n_alpha
    mol_info["n_beta"] = n_beta

    skqd_config = HINQSSKQDConfig(
        n_iterations=config.get("max_iterations", 10),
        n_samples_per_iter=config.get("n_samples_per_iter", 5000),
        nqs_train_epochs=config.get("nqs_train_epochs", 30),
        krylov_max_new=config.get("krylov_max_new", 200),
        krylov_n_ref=config.get("krylov_n_ref", 5),
        device=device,
    )
    nqs_result = run_hi_nqs_skqd(
        hamiltonian, mol_info, config=skqd_config, initial_basis=basis
    )

    wall_time = time.perf_counter() - t_start

    # === Results ===
    final_energy = nqs_result.energy
    if nf_dci_energy is not None:
        final_energy = min(final_energy, nf_dci_energy)
    error_mha = (final_energy - exact_energy) * 1000.0
    within = "YES" if abs(error_mha) < CHEMICAL_ACCURACY_MHA else "NO"

    energy_history = nqs_result.metadata.get("energy_history", [])
    if energy_history:
        print("\n  Iterative NQS+SKQD convergence:")
        for i, e in enumerate(energy_history):
            err = (e - exact_energy) * 1000.0
            print(f"    iter {i + 1:>3}: {e:.10f} Ha  (error: {err:.4f} mHa)")

    print("\n" + "=" * 60)
    print("PIPELINE 07a: NF+DCI MERGE -> ITERATIVE NQS+KRYLOV RESULTS")
    print("=" * 60)
    print(f"NF training      : {n_epochs} epochs ({nf_time:.1f}s)")
    print(f"NF+DCI basis     : {basis.shape[0]} configs")
    print(f"NQS iterations   : {nqs_result.metadata.get('n_iterations', '?')}")
    print(f"NQS converged    : {nqs_result.converged}")
    print(f"NQS final basis  : {nqs_result.diag_dim}")
    print(f"\nFinal energy : {final_energy:.10f} Ha")
    print(f"Exact energy : {exact_energy:.10f} Ha")
    print(f"Error        : {error_mha:.4f} mHa")
    print(f"Chemical acc.: {within}")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
