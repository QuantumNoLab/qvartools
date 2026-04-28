#!/usr/bin/env python3
"""
C2H4 SCI with no basis cap.

SCI (CIPSI) runs without any max_basis_size limit,
using sparse eigsh when basis > 10000 to avoid OOM.
Goal: get a tight SCI reference energy for C2H4 (28Q).

Previous run with basis=10000 cap: -77.2351408123 Ha (744s)
v3 Config B achieved: -77.2352038733 Ha (801s)
"""
import sys, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

H, info = get_molecule("C2H4")
print(f"C2H4: 28Q, 14 orb, ({H.n_alpha},{H.n_beta})e, Hilbert=9,018,009", flush=True)
print(f"Running SCI with NO basis cap (sparse eigsh for basis>10000)", flush=True)
print(f"expansion_size=1000, convergence_threshold=1e-8, max_iterations=200", flush=True)
print(flush=True)

solver = CIPSISolver(
    max_iterations=200,
    max_basis_size=0,           # no cap
    convergence_threshold=1e-8,
    expansion_size=1000,
)

t0 = time.time()

# Monkey-patch solve to print progress
original_solve = solver.solve

import logging
logging.basicConfig(level=logging.WARNING)  # suppress debug

# Run directly and print iteration info inline
from src.utils.config_hash import config_integer_hash
from collections import defaultdict

hf_config = H.get_hf_state()
if hf_config.dim() == 1:
    hf_config = hf_config.unsqueeze(0)

basis = hf_config.clone()
basis_hashes = set(config_integer_hash(basis))
prev_energy = None
converged = False
iteration_energies = []

for it in range(200):
    n_basis = basis.shape[0]
    iter_t0 = time.time()

    # Diagonalize
    if n_basis <= 10000:
        h_matrix = H.matrix_elements_fast(basis)
        h_np = np.asarray(h_matrix, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(h_np)
        e0 = float(eigvals[0])
        coeffs = eigvecs[:, 0]
    else:
        from src.utils.gpu_linalg import sparse_hamiltonian_eigsh
        eigvals, eigvecs = sparse_hamiltonian_eigsh(H, basis, k=1, which='SA')
        e0 = float(eigvals[0].cpu())
        coeffs = eigvecs[:, 0].cpu().numpy()

    iteration_energies.append(e0)

    delta_e = abs(e0 - prev_energy) if prev_energy is not None else float('inf')
    iter_time = time.time() - iter_t0
    elapsed = time.time() - t0

    print(f"  Iter {it:>4d}: E={e0:.10f} Ha, basis={n_basis:>7d}, "
          f"ΔE={delta_e:.2e}, iter_t={iter_time:.1f}s, total={elapsed:.0f}s", flush=True)

    if prev_energy is not None and delta_e < 1e-8:
        converged = True
        print(f"\n  Converged! |ΔE|={delta_e:.2e} < 1e-8", flush=True)
        break
    prev_energy = e0

    # Collect candidates
    numerator_accum = defaultdict(float)
    candidate_configs = {}

    for idx in range(n_basis):
        c_i = float(coeffs[idx])
        if abs(c_i) < 1e-14:
            continue
        config_i = basis[idx]
        connections, h_elements = H.get_connections(config_i)
        if connections is None or len(connections) == 0:
            continue
        conn_hashes = config_integer_hash(connections)
        for j, h_conn in enumerate(conn_hashes):
            if h_conn in basis_hashes:
                continue
            numerator_accum[h_conn] += c_i * float(h_elements[j])
            if h_conn not in candidate_configs:
                candidate_configs[h_conn] = connections[j]

    if not candidate_configs:
        print(f"  No new candidates; stopping.", flush=True)
        break

    # PT2 importance
    cand_hash_list = list(candidate_configs.keys())
    cand_tensor = torch.stack([candidate_configs[h] for h in cand_hash_list])
    h_diag = np.asarray(H.diagonal_elements_batch(cand_tensor), dtype=np.float64)

    importances = np.empty(len(cand_hash_list), dtype=np.float64)
    for k, h_key in enumerate(cand_hash_list):
        numer_sq = numerator_accum[h_key] ** 2
        denom = abs(e0 - h_diag[k])
        importances[k] = numer_sq / max(denom, 1e-14)

    n_add = min(1000, len(cand_hash_list))
    if n_add >= len(cand_hash_list):
        top_indices = np.arange(len(cand_hash_list))
    else:
        top_indices = np.argpartition(-importances, n_add)[:n_add]

    new_configs = cand_tensor[top_indices]
    new_hashes = [cand_hash_list[i] for i in top_indices]
    basis = torch.cat([basis, new_configs], dim=0)
    basis_hashes.update(new_hashes)

wall_time = time.time() - t0

print(f"\n{'='*70}", flush=True)
print(f"  C2H4 SCI (unlimited) RESULT", flush=True)
print(f"{'='*70}", flush=True)
print(f"  Energy:    {e0:.10f} Ha", flush=True)
print(f"  Basis:     {basis.shape[0]:,}", flush=True)
print(f"  Converged: {converged}", flush=True)
print(f"  Wall time: {wall_time:.1f}s ({wall_time/3600:.2f}h)", flush=True)
print(f"  vs SCI(10000): {(e0 - (-77.2351408123))*1000:+.4f} mHa", flush=True)
print(f"  vs v3 B:       {(e0 - (-77.2352038733))*1000:+.4f} mHa", flush=True)
