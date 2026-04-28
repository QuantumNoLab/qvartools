#!/bin/bash
#SBATCH --job-name=bench-ffsim
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_ffsim_%j.log
#SBATCH --error=bench_ffsim_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time, os, numpy as np, torch
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.nqs.transformer import AutoregressiveTransformer
from qiskit_addon_sqd.fermion import solve_fermion
import ffsim
from scipy.sparse.linalg import eigsh
from math import comb

MOLECULES = ['H2O', 'C2H2', 'C2H4']

for mol_name in MOLECULES:
    print('\n' + '='*70)
    print(f'  {mol_name}')
    print('='*70)

    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    n_alpha, n_beta = H.n_alpha, H.n_beta
    n_qubits = 2 * n_orb
    hcore = np.asarray(H.integrals.h1e, dtype=np.float64)
    eri = np.asarray(H.integrals.h2e, dtype=np.float64)
    nuc = float(H.integrals.nuclear_repulsion)
    dim = comb(n_orb, n_alpha) * comb(n_orb, n_beta)
    print(f'{n_orb} orbitals, ({n_alpha},{n_beta}) electrons, full CI dim={dim:,}')

    # ---- Sample shared configs ----
    nqs = AutoregressiveTransformer(n_orbitals=n_orb, n_alpha=n_alpha, n_beta=n_beta,
                                     embed_dim=128, n_heads=4, n_layers=4).cuda()
    with torch.no_grad():
        configs, _ = nqs.sample(2000, temperature=1.0)
        configs = configs.long().cpu()
        a_ok = configs[:, :n_orb].sum(1) == n_alpha
        b_ok = configs[:, n_orb:].sum(1) == n_beta
        configs = torch.unique(configs[a_ok & b_ok], dim=0)
    n_configs = len(configs)

    # IBM format
    bs = np.zeros((n_configs, n_qubits), dtype=bool)
    for s in range(n_configs):
        for j in range(n_orb):
            bs[s, n_orb-1-j] = bool(configs[s,j])
            bs[s, n_qubits-1-j] = bool(configs[s,j+n_orb])

    # Alpha/beta decomposition
    alpha_ints = set()
    beta_ints = set()
    for s in range(n_configs):
        a_int = sum(int(configs[s,j]) << j for j in range(n_orb))
        b_int = sum(int(configs[s,j+n_orb]) << j for j in range(n_orb))
        alpha_ints.add(a_int)
        beta_ints.add(b_int)
    n_a, n_b = len(alpha_ints), len(beta_ints)
    print(f'Configs: {n_configs}, unique alpha: {n_a}, beta: {n_b}, product: {n_a*n_b}')

    # ---- Method 1: IBM solve_fermion (PySCF SCI) ----
    print('\n  [1] IBM solve_fermion (PySCF SCI, CPU)...')
    t0 = time.time()
    try:
        e_ibm, _, _, _ = solve_fermion(bs, hcore, eri, spin_sq=0)
        e_ibm += nuc
        t_ibm = time.time() - t0
        print(f'      E = {e_ibm:.10f}, t = {t_ibm:.2f}s')
    except Exception as ex:
        t_ibm = time.time() - t0
        e_ibm = None
        print(f'      FAILED: {ex}, t = {t_ibm:.2f}s')

    # ---- Method 2: ffsim linear_operator + eigsh (full space) ----
    print('\n  [2] ffsim linear_operator + eigsh (full CI space)...')
    mol_ham = ffsim.MolecularHamiltonian(one_body_tensor=hcore, two_body_tensor=eri, constant=nuc)
    linop = ffsim.linear_operator(mol_ham, norb=n_orb, nelec=(n_alpha, n_beta))
    t0 = time.time()
    try:
        evals, _ = eigsh(linop, k=1, which='SA')
        e_ffsim_full = float(evals[0])
        t_ffsim_full = time.time() - t0
        print(f'      E = {e_ffsim_full:.10f}, t = {t_ffsim_full:.2f}s  (full dim={dim:,})')
    except Exception as ex:
        t_ffsim_full = time.time() - t0
        e_ffsim_full = None
        print(f'      FAILED: {ex}, t = {t_ffsim_full:.2f}s')

    # ---- Method 3: ffsim with subspace (same configs as IBM) ----
    print('\n  [3] ffsim subspace linear_operator + eigsh...')
    alpha_list = sorted(alpha_ints)
    beta_list = sorted(beta_ints)
    t0 = time.time()
    try:
        linop_sub = ffsim.linear_operator(
            mol_ham, norb=n_orb, nelec=(n_alpha, n_beta),
            spin_alpha_ci_strs=np.array(alpha_list, dtype=np.int64),
            spin_beta_ci_strs=np.array(beta_list, dtype=np.int64),
        )
        evals_sub, _ = eigsh(linop_sub, k=1, which='SA')
        e_ffsim_sub = float(evals_sub[0])
        t_ffsim_sub = time.time() - t0
        print(f'      E = {e_ffsim_sub:.10f}, t = {t_ffsim_sub:.2f}s  (subspace dim={n_a*n_b})')
    except Exception as ex:
        t_ffsim_sub = time.time() - t0
        e_ffsim_sub = None
        print(f'      FAILED: {ex}, t = {t_ffsim_sub:.2f}s')

    # ---- Summary ----
    print(f'\n  --- {mol_name} Summary ---')
    print(f'  {\"Method\":<35} {\"Energy\":>18} {\"Time\":>8}')
    print(f'  {\"-\"*63}')
    if e_ibm is not None:
        print(f'  {\"IBM solve_fermion (PySCF SCI)\":<35} {e_ibm:>18.10f} {t_ibm:>7.2f}s')
    if e_ffsim_full is not None:
        print(f'  {\"ffsim full CI\":<35} {e_ffsim_full:>18.10f} {t_ffsim_full:>7.2f}s')
    if e_ffsim_sub is not None:
        print(f'  {\"ffsim subspace (same configs)\":<35} {e_ffsim_sub:>18.10f} {t_ffsim_sub:>7.2f}s')
    if e_ibm and e_ffsim_sub:
        print(f'  IBM vs ffsim-sub energy diff: {abs(e_ibm-e_ffsim_sub)*1000:.6f} mHa')
    if t_ibm and t_ffsim_sub:
        print(f'  Speedup (ffsim-sub / IBM): {t_ibm/t_ffsim_sub:.1f}x')
"

echo "=== Done: $(date) ==="
