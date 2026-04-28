"""Measure actual Davidson dimension for N2-CAS(10,20) 40Q on 40k NQS samples.

The FCI space (240M) is infeasible — this is the first system where NQS
sampling should actually matter (can't saturate all 15,504 alpha strings
from only 40k samples).
"""
import numpy as np
import torch
from math import comb

from src.molecules import get_molecule
from src.nqs.transformer import AutoregressiveTransformer
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs


def measure(mol_name, n_samples_list, temperature=1.0, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    n_a_target = H.n_alpha
    n_b_target = H.n_beta
    n_qubits = 2 * n_orb

    ca = comb(n_orb, n_a_target)
    cb = comb(n_orb, n_b_target)
    fci_dim = ca * cb

    print(f"\n{'=' * 90}")
    print(f"  {mol_name}  n_orb={n_orb}  (n_a,n_b)=({n_a_target},{n_b_target})  T={temperature}")
    print(f"  C(n_orb, n_a) = {ca:,}    Full FCI dim = {fci_dim:,}")
    print(f"{'=' * 90}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Scale transformer to the 40Q tier (from hi_nqs_sqd.py)
    embed, heads, layers = 256, 8, 8
    nqs = AutoregressiveTransformer(
        n_orbitals=n_orb, n_alpha=n_a_target, n_beta=n_b_target,
        embed_dim=embed, n_heads=heads, n_layers=layers,
    ).to(device)
    nqs.eval()

    print(f"  {'N_samples':>10} {'N_det':>8} {'n_a':>7} {'n_b':>7} "
          f"{'n_a*n_b':>14} {'%FCI':>7} {'blowup vs N_det':>17}")
    print(f"  {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*14} {'-'*7} {'-'*17}")

    for ns in n_samples_list:
        with torch.no_grad():
            configs_gpu, _ = nqs.sample(ns, temperature=temperature)
            configs = configs_gpu.long().cpu()
            alpha_cnt = configs[:, :n_orb].sum(dim=1)
            beta_cnt = configs[:, n_orb:].sum(dim=1)
            valid = (alpha_cnt == n_a_target) & (beta_cnt == n_b_target)
            configs = configs[valid]
            configs = torch.unique(configs, dim=0).numpy().astype(bool)

        N_det = len(configs)
        bitstring_matrix = np.zeros((N_det, n_qubits), dtype=bool)
        bitstring_matrix[:, :n_orb] = configs[:, n_orb:]
        bitstring_matrix[:, n_orb:] = configs[:, :n_orb]
        ci_a, ci_b = bitstring_matrix_to_ci_strs(bitstring_matrix, open_shell=False)
        n_a = len(ci_a)
        n_b = len(ci_b)
        tensor = n_a * n_b
        pct_fci = 100.0 * tensor / fci_dim
        blowup = tensor / max(N_det, 1)

        print(f"  {ns:>10,} {N_det:>8,} {n_a:>7,} {n_b:>7,} "
              f"{tensor:>14,} {pct_fci:>6.2f}% {blowup:>16.1f}×")


if __name__ == "__main__":
    measure("N2-CAS(10,20)", [5_000, 20_000, 40_000], temperature=1.0)
    measure("N2-CAS(10,20)", [40_000], temperature=0.3)
    measure("N2-CAS(10,20)", [40_000], temperature=0.1)
