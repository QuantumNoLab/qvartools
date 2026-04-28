#!/usr/bin/env python3
"""
N2-CAS(10,20) 40Q and N2-CAS(10,26) 52Q: SCI with no basis cap.

Uses sparse eigsh for basis > 10000 to avoid OOM.
Previous job (156407) was cut off during N2-CAS(10,20) SCI.
"""
import sys, time
import numpy as np
import torch
from math import comb
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.utils.config_hash import config_integer_hash

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)


def run_sci_unlimited(H, info, expansion_size=1000, label=""):
    """Run SCI with no basis cap, printing per-iteration progress."""
    hf_config = H.get_hf_state()
    if hf_config.dim() == 1:
        hf_config = hf_config.unsqueeze(0)

    basis = hf_config.clone()
    basis_hashes = set(config_integer_hash(basis))
    prev_energy = None
    converged = False
    t_total = time.time()

    for it in range(300):
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

        delta_e = abs(e0 - prev_energy) if prev_energy is not None else float('inf')
        iter_time = time.time() - iter_t0
        elapsed = time.time() - t_total

        print(f"  [{label}] Iter {it:>4d}: E={e0:.10f} Ha, basis={n_basis:>8,}, "
              f"ΔE={delta_e:.2e}, iter={iter_time:.1f}s, total={elapsed:.0f}s", flush=True)

        if prev_energy is not None and delta_e < 1e-8:
            converged = True
            print(f"  [{label}] Converged! ΔE={delta_e:.2e}", flush=True)
            break
        prev_energy = e0

        # Collect candidates via single/double excitations
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
            print(f"  [{label}] No new candidates; stopping.", flush=True)
            break

        # PT2 importance scoring
        cand_hash_list = list(candidate_configs.keys())
        cand_tensor = torch.stack([candidate_configs[h] for h in cand_hash_list])
        h_diag = np.asarray(H.diagonal_elements_batch(cand_tensor), dtype=np.float64)

        importances = np.empty(len(cand_hash_list), dtype=np.float64)
        for k, h_key in enumerate(cand_hash_list):
            numer_sq = numerator_accum[h_key] ** 2
            denom = abs(e0 - h_diag[k])
            importances[k] = numer_sq / max(denom, 1e-14)

        n_add = min(expansion_size, len(cand_hash_list))
        if n_add >= len(cand_hash_list):
            top_indices = np.arange(len(cand_hash_list))
        else:
            top_indices = np.argpartition(-importances, n_add)[:n_add]

        new_configs = cand_tensor[top_indices]
        new_hashes = [cand_hash_list[i] for i in top_indices]
        basis = torch.cat([basis, new_configs], dim=0)
        basis_hashes.update(new_hashes)

    wall_time = time.time() - t_total
    return e0, basis.shape[0], converged, wall_time


SYSTEMS = [
    ("N2-CAS(10,20)", 1000),   # 40Q, Hilbert=240M
    ("N2-CAS(10,26)", 2000),   # 52Q, Hilbert=~5B
]

results = []

for mol_name, expansion in SYSTEMS:
    print(f"\n{'='*70}", flush=True)
    print(f"  {mol_name}  (expansion_size={expansion}, no basis cap)", flush=True)
    print(f"{'='*70}", flush=True)

    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    nq = info["n_qubits"]
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f"  {nq}Q, {n_orb} orb, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}", flush=True)

    e0, final_basis, converged, wall_time = run_sci_unlimited(
        H, info, expansion_size=expansion, label=mol_name
    )

    print(f"\n  Result: E={e0:.10f} Ha, basis={final_basis:,}, "
          f"converged={converged}, time={wall_time:.1f}s ({wall_time/3600:.2f}h)", flush=True)

    results.append(dict(mol=mol_name, nq=nq, E=e0, basis=final_basis,
                        converged=converged, time=wall_time))

print(f"\n{'='*70}", flush=True)
print(f"  N2 40Q/52Q SCI UNLIMITED SUMMARY", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Mol':<20} {'Q':>3}  {'Energy':>18}  {'basis':>9}  {'time':>8}  {'conv':>5}", flush=True)
for r in results:
    print(f"  {r['mol']:<20} {r['nq']:>3}  {r['E']:>18.10f}  {r['basis']:>9,}  "
          f"{r['time']:>7.0f}s  {'Y' if r['converged'] else 'N':>5}", flush=True)
