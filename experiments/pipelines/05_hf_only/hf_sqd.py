"""HF-only -> SQD batch diagonalisation with noise + S-CORE.

Pipeline: Uses only the HF reference state (1 configuration) as the
starting point. Replicates it to simulate shot sampling, injects noise,
then runs SQD with self-consistent configuration recovery.

Uses skip_nf_training=True, subspace_mode="sqd". After generating
HF+S+D via train_flow_nqs(), overrides with HF-only, then replicates
and runs SQD.
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
from qvartools.molecules import get_molecule
from qvartools.solvers import FCISolver

CHEMICAL_ACCURACY_MHA = 1.6


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_noise_rate(n_qubits: int) -> float:
    return 0.03 if n_qubits <= 4 else 0.05


def get_shots_multiplier(n_unique: int, n_qubits: int) -> int:
    target_shots = 20_000
    return max(10, min(200, target_shots // max(n_unique, 1)))


def get_sqd_params(n_configs: int) -> dict:
    if n_configs <= 2000:
        sqd_num_batches = 5
    elif n_configs <= 5000:
        sqd_num_batches = 8
    else:
        sqd_num_batches = 10
    return dict(
        sqd_num_batches=sqd_num_batches,
        sqd_self_consistent_iters=5,
        sqd_use_spin_symmetry=n_configs <= 5000,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = create_base_parser("HF-only -> SQD (noise -> S-CORE).")
    parser.add_argument("--sqd-num-batches", type=int, default=None)
    parser.add_argument("--sqd-self-consistent-iters", type=int, default=None)
    parser.add_argument("--sqd-noise-rate", type=float, default=None)
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
    print(f"Hilbert space: {n_configs:,} configs")

    fci_result = FCISolver().solve(hamiltonian, mol_info)
    exact_energy = fci_result.energy
    if exact_energy is not None:
        print(f"Exact (FCI) energy: {exact_energy:.10f} Ha")
    else:
        print("FCI reference unavailable for this system.")
    print("-" * 60)

    noise_rate = config.get("sqd_noise_rate", get_noise_rate(n_qubits))
    sqd_defaults = get_sqd_params(n_configs)

    pipeline_config = PipelineConfig(
        skip_nf_training=True,
        subspace_mode="sqd",
        sqd_noise_rate=noise_rate,
        device=device,
        sqd_num_batches=config.get("sqd_num_batches", sqd_defaults["sqd_num_batches"]),
        sqd_self_consistent_iters=config.get(
            "sqd_self_consistent_iters", sqd_defaults["sqd_self_consistent_iters"]
        ),
        sqd_use_spin_symmetry=config.get(
            "sqd_use_spin_symmetry", sqd_defaults["sqd_use_spin_symmetry"]
        ),
    )

    pipeline = FlowGuidedKrylovPipeline(
        hamiltonian=hamiltonian,
        config=pipeline_config,
        exact_energy=exact_energy,
        auto_adapt=True,
    )

    t_start = time.perf_counter()

    # Stage 1: Generate HF+S+D (Direct-CI)
    pipeline.train_flow_nqs(progress=True)

    # Override: HF state only
    hf_state = pipeline.reference_state.clone().unsqueeze(0)
    pipeline._essential_configs = hf_state  # noqa: SLF001
    print(f"Overriding basis: HF state only ({hf_state.shape[0]} config)")

    # Stage 2: Extract basis (just HF)
    basis = pipeline.extract_and_select_basis()
    n_unique = basis.shape[0]
    print(f"HF-only basis: {n_unique} config(s)")

    # Replicate for shot simulation
    shots_mult = get_shots_multiplier(n_unique, n_qubits)
    total_shots = n_unique * shots_mult
    replicated_basis = basis.repeat(shots_mult, 1)
    pipeline.nf_basis = replicated_basis
    print(f"Shots multiplier: {shots_mult}x -> {total_shots} total shots")
    print(f"Noise rate: {noise_rate}")

    # Stage 3: SQD batch diag with noise + S-CORE
    pipeline.run_subspace_diag(progress=True)

    wall_time = time.perf_counter() - t_start

    # Results
    print("\n" + "=" * 60)
    print("HF-ONLY -> SQD RESULTS (noise -> S-CORE)")
    print("=" * 60)
    print(f"Initial basis: {n_unique} config (HF only)")
    print(f"Noise rate   : {noise_rate}")
    print(f"Total shots  : {total_shots}")

    final_energy = pipeline.results.get(
        "final_energy", pipeline.results.get("combined_energy")
    )
    error_mha = pipeline.results.get("error_mha")
    if error_mha is None and final_energy is not None and exact_energy is not None:
        error_mha = (final_energy - exact_energy) * 1000.0

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
        print(f"Chemical acc.: {within}")
    print(f"Wall time    : {wall_time:.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
