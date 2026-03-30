"""NF-only + SQD --- NF basis (no DCI merge) -> noise -> S-CORE -> batch diag.

Pipeline: Train NF-NQS, use NF-only basis (ablation: no HF+S+D scaffolding),
then run SQD with noise injection and self-consistent recovery.
"""

from __future__ import annotations

import logging
import sys
import time
from math import comb
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config_loader import create_base_parser, load_config

from qvartools import FlowGuidedKrylovPipeline, PipelineConfig
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def get_noise_rate(n_qubits: int) -> float:
    return 0.03 if n_qubits <= 4 else 0.05


def get_shots_multiplier(n_unique: int, n_qubits: int) -> int:
    target_shots = 20_000
    return max(10, min(200, target_shots // max(n_unique, 1)))


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
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    parser = create_base_parser("NF-only + SQD: NF basis -> noise -> S-CORE.")
    parser.add_argument("--sqd-noise-rate", type=float, default=None)
    parser.add_argument("--sqd-num-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=None)
    args, config = load_config(parser)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hamiltonian, mol_info = get_molecule(config.get("molecule", "h2"), device=device)
    n_qubits = mol_info["n_qubits"]
    n_orb = hamiltonian.integrals.n_orbitals
    n_alpha = hamiltonian.integrals.n_alpha
    n_beta = hamiltonian.integrals.n_beta
    n_configs = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

    print(f"Molecule : {mol_info['name']} ({n_qubits} qubits, {n_configs:,} configs)")
    print(f"Device   : {device}")
    print("=" * 60)

    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    if exact_energy is not None:
        print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    else:
        print("FCI reference unavailable for this system.")

    noise_rate = config.get("sqd_noise_rate", get_noise_rate(n_qubits))
    td = get_training_params(n_configs)

    pipe_config = PipelineConfig(
        skip_nf_training=False,
        subspace_mode="sqd",
        sqd_noise_rate=noise_rate,
        sqd_num_batches=config.get("sqd_num_batches", 5),
        sqd_self_consistent_iters=config.get("sqd_self_consistent_iters", 5),
        teacher_weight=config.get("teacher_weight", 0.5),
        physics_weight=config.get("physics_weight", 0.4),
        entropy_weight=config.get("entropy_weight", 0.1),
        max_epochs=config.get("max_epochs", td["max_epochs"]),
        min_epochs=config.get("min_epochs", td["min_epochs"]),
        samples_per_batch=config.get("samples_per_batch", td["samples_per_batch"]),
        nf_hidden_dims=config.get("nf_hidden_dims", td["nf_hidden_dims"]),
        nqs_hidden_dims=config.get("nqs_hidden_dims", td["nqs_hidden_dims"]),
        device=device,
    )

    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipe_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    t_start = time.perf_counter()
    pipeline.train_flow_nqs(progress=True)

    nf_only = torch.unique(pipeline.trainer.accumulated_basis.clone(), dim=0)
    pipeline._essential_configs = nf_only.to(device)

    basis = pipeline.extract_and_select_basis()
    n_unique = basis.shape[0]

    shots_mult = get_shots_multiplier(n_unique, n_qubits)
    pipeline.nf_basis = basis.repeat(shots_mult, 1)

    pipeline.run_subspace_diag(progress=True)
    wall_time = time.perf_counter() - t_start

    final_energy = pipeline.results.get(
        "final_energy", pipeline.results.get("combined_energy")
    )
    error_mha = (
        (final_energy - exact_energy) * 1000.0
        if (final_energy is not None and exact_energy is not None)
        else None
    )

    print(f"\n{'=' * 60}")
    print("NF-ONLY + SQD RESULTS")
    print(f"{'=' * 60}")
    if final_energy is not None:
        print(f"Final energy : {final_energy:.10f} Ha")
    else:
        print("Final energy : N/A")
    if exact_energy is not None:
        print(f"Exact energy : {exact_energy:.10f} Ha")
    else:
        print("Exact energy : N/A")
    if error_mha is not None:
        within = (
            "YES"
            if (error_mha is not None and abs(error_mha) < CHEMICAL_ACCURACY_MHA)
            else ("NO" if error_mha is not None else "N/A")
        )
        print(f"Error        : {error_mha:.4f} mHa")
        print(f"Chemical acc.: {within} (threshold = {CHEMICAL_ACCURACY_MHA} mHa)")
    print(f"Wall time    : {wall_time:.2f} s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
