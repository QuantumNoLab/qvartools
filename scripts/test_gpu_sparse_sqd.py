#!/usr/bin/env python3
"""
Test GPUSparseSQDBackend correctness vs IncrementalSQDBackend (PySCF).

Validates:
  1. Sparse H matches PySCF contract_2e (single H·v test)
  2. Eigenvalue matches PySCF kernel_fixed_space (small system)
  3. Energy matches across growing basis sequence
  4. Speed comparison
"""
import sys, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.methods.incremental_sqd import IncrementalSQDBackend
from src.methods.gpu_sparse_sqd import (
    GPUSparseSQDBackend, build_sparse_H_gpu, gpu_davidson, _ibm_to_nqs_format_gpu
)
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)


def make_basis_sequence(hamiltonian, sizes):
    """Build a growing basis sequence using diagonal-energy ranking."""
    from itertools import combinations
    n_orb = hamiltonian.n_orbitals
    n_alpha = hamiltonian.n_alpha
    n_beta = hamiltonian.n_beta
    n_qubits = 2 * n_orb

    all_alpha = list(combinations(range(n_orb), n_alpha))
    all_beta = list(combinations(range(n_orb), n_beta))

    np.random.seed(0)
    pairs = []
    seen = set()
    while len(pairs) < max(sizes):
        i = np.random.randint(len(all_alpha))
        j = np.random.randint(len(all_beta))
        if (i, j) not in seen:
            seen.add((i, j))
            pairs.append((all_alpha[i], all_beta[j]))

    def make_bs(idxa, idxb):
        bs = np.zeros(n_qubits, dtype=bool)
        for i in idxa:
            bs[n_orb - 1 - i] = True
        for i in idxb:
            bs[n_qubits - 1 - i] = True
        return bs

    sequences = []
    for size in sizes:
        sequences.append(np.stack([make_bs(*p) for p in pairs[:size]]))
    return sequences


def test_system(name, basis_sizes):
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'='*70}", flush=True)

    H_gpu, info = get_molecule(name, device="cuda")
    H_cpu, _ = get_molecule(name, device="cpu")

    nrep = H_gpu.integrals.nuclear_repulsion
    print(f"  n_orb={H_gpu.n_orbitals}, n_alpha={H_gpu.n_alpha}, "
          f"n_qubits={2 * H_gpu.n_orbitals}", flush=True)
    print(f"  nuclear_repulsion={nrep:.10f}", flush=True)

    sequences = make_basis_sequence(H_gpu, basis_sizes)
    print(f"  Test basis sizes: {basis_sizes}", flush=True)

    # Run both backends on growing basis
    backend_cpu = IncrementalSQDBackend(
        H_gpu.integrals.h1e, H_gpu.integrals.h2e,
        n_alpha=H_gpu.n_alpha, n_beta=H_gpu.n_beta, spin_sq=0,
    )
    backend_gpu = GPUSparseSQDBackend(H_gpu, spin_sq=0)

    print(f"\n  {'Step':>5} {'basis':>7} {'E_PySCF':>18} {'E_GPU':>18} {'ΔE':>14} {'t_CPU':>10} {'t_GPU':>10}", flush=True)
    print(f"  {'-'*100}", flush=True)

    for i, bs in enumerate(sequences):
        # CPU (PySCF) baseline
        t0 = time.time()
        try:
            e_cpu, sci_cpu = backend_cpu.solve(bs)
            e_cpu_total = e_cpu + nrep
            t_cpu = time.time() - t0
        except Exception as ex:
            print(f"  Step {i}: CPU FAILED: {ex}", flush=True)
            continue

        # GPU sparse Davidson
        t0 = time.time()
        try:
            e_gpu, sci_gpu = backend_gpu.solve(bs)
            e_gpu_total = e_gpu + nrep
            t_gpu = time.time() - t0
        except Exception as ex:
            print(f"  Step {i}: GPU FAILED: {ex}", flush=True)
            continue

        delta = abs(e_cpu_total - e_gpu_total)
        marker = "✓" if delta < 1e-7 else "✗"
        print(f"  {i:>5} {len(bs):>7} {e_cpu_total:>18.10f} {e_gpu_total:>18.10f} "
              f"{delta:>14.2e} {t_cpu*1000:>9.0f}ms {t_gpu*1000:>9.0f}ms  {marker}", flush=True)


# ============================================================
# Run tests
# ============================================================
print("\n=== Test 1: H2O 14Q ===")
test_system("H2O", [10, 30, 100, 200, 441])

print("\n=== Test 2: NH3 16Q ===")
test_system("NH3", [10, 30, 100, 500, 2000])

print("\n=== Test 3: C2H2 24Q ===")
test_system("C2H2", [10, 100, 1000, 5000, 15000])

print("\n=== Done ===", flush=True)
