#!/usr/bin/env python
"""Diagnostic: measure actual torch/numpy conversion costs in the pipeline.

Run: python experiments/diagnose_roundtrips.py [molecule] [--device cuda]

Reports:
  1. Actual SQD batch sizes
  2. Time spent in each conversion step
  3. Eigensolve time vs transfer time ratio
  4. SKQD eigensolve conversion overhead
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def time_fn(fn, *args, **kwargs):
    """Time a function call, return (result, elapsed_seconds)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


# ---------------------------------------------------------------------------
# 1. SQD batch size and eigensolve profiling
# ---------------------------------------------------------------------------


def diagnose_sqd(molecule: str, device_str: str) -> None:
    """Profile SQD _diagonalize_batch conversions."""
    try:
        from qvartools.molecules import get_molecule
    except ImportError:
        print("PySCF not available, skipping SQD diagnosis")
        return

    print(f"\n{'=' * 60}")
    print(f"SQD DIAGNOSIS: {molecule} on {device_str}")
    print(f"{'=' * 60}")

    hamiltonian, mol_info = get_molecule(molecule, device=device_str)
    n_qubits = mol_info["n_qubits"]
    print(f"  n_qubits: {n_qubits}")

    # Generate a realistic basis (HF + singles + doubles)
    from qvartools.pipeline import FlowGuidedKrylovPipeline
    from qvartools.pipeline_config import PipelineConfig

    cfg = PipelineConfig(
        use_particle_conserving_flow=True,
        nf_hidden_dims=[32],
        nqs_hidden_dims=[32],
        samples_per_batch=100,
        num_batches=2,
        max_epochs=5,
        min_epochs=2,
        max_accumulated_basis=200,
        skip_skqd=True,
        device=device_str,
    )
    pipeline = FlowGuidedKrylovPipeline(hamiltonian, config=cfg)
    pipeline.train_flow_nqs(progress=False)
    basis = pipeline.extract_and_select_basis()
    print(f"  Basis size: {basis.shape[0]}")
    print(f"  Basis device: {basis.device}")

    # Simulate SQD batching
    from qvartools.krylov.circuits.sqd import SQDConfig, SQDSolver

    sqd_cfg = SQDConfig(num_batches=3, use_spin_symmetry_enhancement=False)
    solver = SQDSolver(hamiltonian, config=sqd_cfg)
    batch_size = max(2, len(basis) // 3)
    batches = solver._create_batches(basis, batch_size, 3)  # noqa: SLF001

    print(f"\n  Batch sizes: {[len(b) for b in batches]}")

    # Profile one batch
    for i, batch in enumerate(batches[:2]):
        n = len(batch)
        if n == 0:
            continue

        # Build H on GPU
        H_matrix = hamiltonian.matrix_elements(batch, batch)
        H_gpu = H_matrix.detach().double()
        if H_gpu.is_complex():
            H_gpu = H_gpu.real
        H_gpu = 0.5 * (H_gpu + H_gpu.T)

        print(f"\n  --- Batch {i} (n={n}) ---")
        print(f"  H_gpu device: {H_gpu.device}, dtype: {H_gpu.dtype}")
        print(f"  H_gpu size: {H_gpu.shape} = {H_gpu.nelement() * 8 / 1024:.1f} KB")

        # Current path: GPU → numpy → eigh → numpy → GPU
        _, t_to_np = time_fn(lambda: H_gpu.detach().cpu().numpy())
        H_np = H_gpu.detach().cpu().numpy()

        _, t_eigh_np = time_fn(lambda: np.linalg.eigh(H_np))
        evals_np, evecs_np = np.linalg.eigh(H_np)

        _, t_to_gpu = time_fn(
            lambda: torch.from_numpy(evecs_np[:, 0].copy()).double().to(H_gpu.device)
        )

        t_roundtrip = t_to_np + t_eigh_np + t_to_gpu
        print("  CURRENT PATH (numpy roundtrip):")
        print(f"    GPU→numpy:   {t_to_np * 1000:.3f} ms")
        print(f"    np.eigh:     {t_eigh_np * 1000:.3f} ms")
        print(f"    numpy→GPU:   {t_to_gpu * 1000:.3f} ms")
        print(f"    TOTAL:       {t_roundtrip * 1000:.3f} ms")
        print(
            f"    Transfer %:  {(t_to_np + t_to_gpu) / t_roundtrip * 100:.1f}%"
            if t_roundtrip > 0
            else ""
        )

        # Proposed path: torch.linalg.eigh directly on GPU
        if H_gpu.is_cuda:
            _, t_eigh_torch = time_fn(lambda: torch.linalg.eigh(H_gpu))
            print("  PROPOSED PATH (torch.linalg.eigh on GPU):")
            print(f"    torch.eigh:  {t_eigh_torch * 1000:.3f} ms")
            print(
                f"    Speedup:     {t_roundtrip / t_eigh_torch:.1f}x"
                if t_eigh_torch > 0
                else ""
            )
        else:
            print("  (CPU device — torch.linalg.eigh would be similar to numpy)")


# ---------------------------------------------------------------------------
# 2. SKQD eigensolve profiling
# ---------------------------------------------------------------------------


def diagnose_skqd(molecule: str, device_str: str) -> None:
    """Profile SKQD _solve_generalised_eigenproblem conversions."""
    try:
        from qvartools.molecules import get_molecule
    except ImportError:
        print("PySCF not available, skipping SKQD diagnosis")
        return

    print(f"\n{'=' * 60}")
    print(f"SKQD EIGENSOLVE DIAGNOSIS: {molecule} on {device_str}")
    print(f"{'=' * 60}")

    hamiltonian, mol_info = get_molecule(molecule, device=device_str)

    # Build a small projected Hamiltonian
    from qvartools.krylov.basis.skqd import _solve_generalised_eigenproblem

    # Generate H_proj and S_proj of realistic sizes
    for n in [10, 50, 100, 200, 500]:
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        H = (A + A.T).astype(np.float64)
        S = np.eye(n, dtype=np.float64) + 0.01 * (A @ A.T) / n

        print(f"\n  --- n={n} ---")

        # Current: includes numpy→torch→eigh→numpy conversion
        _, t_current = time_fn(
            lambda: _solve_generalised_eigenproblem(H, S, 2, 1e-8, use_gpu=True)
        )
        # CPU-only
        _, t_cpu = time_fn(
            lambda: _solve_generalised_eigenproblem(H, S, 2, 1e-8, use_gpu=False)
        )
        print(f"    GPU path:  {t_current * 1000:.3f} ms")
        print(f"    CPU path:  {t_cpu * 1000:.3f} ms")
        print(f"    GPU/CPU:   {t_current / t_cpu:.2f}x" if t_cpu > 0 else "")


# ---------------------------------------------------------------------------
# 3. Overall conversion count in a real pipeline run
# ---------------------------------------------------------------------------


def diagnose_conversion_count() -> None:
    """Count actual torch↔numpy conversions during a pipeline run."""
    import functools

    counts = {
        "to_numpy": 0,
        "from_numpy": 0,
        "total_bytes_to_cpu": 0,
        "total_bytes_to_gpu": 0,
    }

    # Monkey-patch torch.Tensor.numpy to count calls
    orig_numpy = torch.Tensor.numpy

    @functools.wraps(orig_numpy)
    def counted_numpy(self, *args, **kwargs):
        counts["to_numpy"] += 1
        counts["total_bytes_to_cpu"] += self.nelement() * self.element_size()
        return orig_numpy(self, *args, **kwargs)

    orig_from_numpy = torch.from_numpy

    @functools.wraps(orig_from_numpy)
    def counted_from_numpy(arr):
        counts["from_numpy"] += 1
        counts["total_bytes_to_gpu"] += arr.nbytes
        return orig_from_numpy(arr)

    torch.Tensor.numpy = counted_numpy  # type: ignore[assignment]
    torch.from_numpy = counted_from_numpy  # type: ignore[assignment]

    try:
        from qvartools.hamiltonians.spin.heisenberg import HeisenbergHamiltonian
        from qvartools.pipeline import FlowGuidedKrylovPipeline
        from qvartools.pipeline_config import PipelineConfig

        ham = HeisenbergHamiltonian(num_spins=6, Jx=1.0, Jy=1.0, Jz=1.0)
        mol_info = {"name": "Heisenberg-6", "n_qubits": 6}

        cfg = PipelineConfig(
            use_particle_conserving_flow=False,
            nf_hidden_dims=[32],
            nqs_hidden_dims=[32],
            samples_per_batch=50,
            num_batches=2,
            max_epochs=3,
            min_epochs=1,
            max_accumulated_basis=100,
            max_krylov_dim=3,
            shots_per_krylov=500,
            skip_skqd=False,
            device="cpu",
        )

        pipeline = FlowGuidedKrylovPipeline(ham, config=cfg)
        pipeline.run(progress=False)

        print(f"\n{'=' * 60}")
        print("CONVERSION COUNT (Heisenberg-6, CPU, 3 Krylov steps)")
        print(f"{'=' * 60}")
        print(f"  torch→numpy calls:  {counts['to_numpy']}")
        print(f"  numpy→torch calls:  {counts['from_numpy']}")
        print(f"  Bytes to CPU:       {counts['total_bytes_to_cpu'] / 1024:.1f} KB")
        print(f"  Bytes from numpy:   {counts['total_bytes_to_gpu'] / 1024:.1f} KB")
    finally:
        torch.Tensor.numpy = orig_numpy  # type: ignore[assignment]
        torch.from_numpy = orig_from_numpy  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    molecule = sys.argv[1] if len(sys.argv) > 1 else "h2"
    device_str = "cuda" if "--device" in sys.argv and "cuda" in sys.argv else "cpu"
    if torch.cuda.is_available() and device_str == "cpu":
        print(
            "NOTE: CUDA available but using CPU. Pass --device cuda for GPU diagnosis."
        )

    diagnose_conversion_count()

    if device_str == "cuda" and torch.cuda.is_available():
        diagnose_sqd(molecule, device_str)
        diagnose_skqd(molecule, device_str)
    else:
        print("\nSkipping GPU-specific diagnosis (no CUDA or --device cpu)")
        diagnose_sqd(molecule, "cpu")


if __name__ == "__main__":
    main()
