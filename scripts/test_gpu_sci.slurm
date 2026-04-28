#!/bin/bash
#SBATCH --job-name=test-gpu-sci
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=test_gpu_sci_%j.log
#SBATCH --error=test_gpu_sci_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time, numpy as np, torch
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.nqs.transformer import AutoregressiveTransformer
from src.solvers.gpu_sci import gpu_solve_fermion
from qiskit_addon_sqd.fermion import solve_fermion

MOLECULES = ['H2O', 'C2H2']

for mol_name in MOLECULES:
    print('\n' + '='*70)
    print(f'  {mol_name}: Correctness + Speed test')
    print('='*70)

    H, info = get_molecule(mol_name)
    n_orb = H.n_orbitals
    n_alpha, n_beta = H.n_alpha, H.n_beta
    n_qubits = 2 * n_orb
    hcore = np.asarray(H.integrals.h1e, dtype=np.float64)
    eri = np.asarray(H.integrals.h2e, dtype=np.float64)
    nuc = float(H.integrals.nuclear_repulsion)

    print(f'  {n_orb} orbitals, ({n_alpha},{n_beta}) electrons')

    # Sample shared configs
    nqs = AutoregressiveTransformer(n_orbitals=n_orb, n_alpha=n_alpha, n_beta=n_beta,
                                     embed_dim=128, n_heads=4, n_layers=4).cuda()
    with torch.no_grad():
        configs, _ = nqs.sample(2000, temperature=1.0)
        configs = configs.long().cpu()
        a_ok = configs[:, :n_orb].sum(1) == n_alpha
        b_ok = configs[:, n_orb:].sum(1) == n_beta
        configs = torch.unique(configs[a_ok & b_ok], dim=0)

    # Convert to IBM format
    bs = np.zeros((len(configs), n_qubits), dtype=bool)
    for s in range(len(configs)):
        for j in range(n_orb):
            bs[s, n_orb-1-j] = bool(configs[s,j])
            bs[s, n_qubits-1-j] = bool(configs[s,j+n_orb])
    print(f'  Configs: {len(bs)}')

    # ---- IBM CPU (reference) ----
    print('\n  [IBM CPU] solve_fermion...')
    t0 = time.time()
    e_ibm, _, occ_ibm, _ = solve_fermion(bs, hcore, eri, spin_sq=0)
    e_ibm += nuc
    t_ibm = time.time() - t0
    print(f'  E = {e_ibm:.12f}, t = {t_ibm:.2f}s')

    # ---- GPU SCI ----
    print('\n  [GPU SCI] gpu_solve_fermion...')
    t0 = time.time()
    e_gpu, _, occ_gpu = gpu_solve_fermion(
        bs, hcore, eri, nuc, n_orb, (n_alpha, n_beta), device='cuda',
    )
    t_gpu = time.time() - t0
    print(f'  E = {e_gpu:.12f}, t = {t_gpu:.2f}s')

    # ---- Compare ----
    diff = abs(e_ibm - e_gpu) * 1000
    speedup = t_ibm / t_gpu if t_gpu > 0 else 0
    print(f'\n  --- {mol_name} Summary ---')
    print(f'  Energy diff: {diff:.6f} mHa')
    print(f'  IBM CPU:  {t_ibm:.2f}s')
    print(f'  GPU SCI:  {t_gpu:.2f}s')
    print(f'  Speedup:  {speedup:.1f}x')
    print(f'  Correct:  {\"YES\" if diff < 0.01 else \"NO\"} (threshold 0.01 mHa)')
"

echo "=== Done: $(date) ==="
