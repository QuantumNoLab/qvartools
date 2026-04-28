#!/bin/bash
#SBATCH --job-name=ibm-vs-gpu-c2h2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=compare_c2h2_%j.log
#SBATCH --error=compare_c2h2_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

/home/leo07010/GQE-MTS/.venv/bin/python -c "
import sys, time
sys.path.insert(0, '.')
import numpy as np
import torch
from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver
from src.nqs.transformer import AutoregressiveTransformer
from src.utils.gpu_linalg import gpu_eigh, sparse_hamiltonian_eigsh
from qiskit_addon_sqd.fermion import solve_fermion

MOL = 'C2H2'
N_SAMPLES = 500
MAX_ITER = 10
DENSE_LIMIT = 10000

H, info = get_molecule(MOL)
nq = info['n_qubits']
n_orb = H.n_orbitals
n_alpha, n_beta = H.n_alpha, H.n_beta
n_qubits = 2 * n_orb
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'{MOL}: {nq}Q, {n_orb} orbitals, {n_alpha}a {n_beta}b')

# SCI reference
print('Computing SCI reference...')
sci = CIPSISolver(max_iterations=50, max_basis_size=10000, expansion_size=500, convergence_threshold=1e-7).solve(H, info)
ref_E = sci.energy
print(f'SCI ref: {ref_E:.10f} Ha (basis={sci.diag_dim}, t={sci.wall_time:.1f}s)')

hcore = np.asarray(H.integrals.h1e, dtype=np.float64)
eri = np.asarray(H.integrals.h2e, dtype=np.float64)
nuc = float(H.integrals.nuclear_repulsion)

# =====================================================
# Step 1: Sample configs with ONE shared NQS
# =====================================================
print('\nSampling from shared NQS...')
nqs = AutoregressiveTransformer(n_orbitals=n_orb, n_alpha=n_alpha, n_beta=n_beta,
                                 embed_dim=128, n_heads=4, n_layers=4).to(device)

all_configs_gpu = torch.zeros(0, n_qubits, dtype=torch.long, device=device)
seen = set()
for it in range(MAX_ITER):
    with torch.no_grad():
        configs, _ = nqs.sample(N_SAMPLES, temperature=1.0)
        configs = configs.long()
    new_u = torch.unique(configs, dim=0)
    # Filter exact particle conservation (rare edge cases from temperature)
    a_counts = new_u[:, :n_orb].sum(dim=1)
    b_counts = new_u[:, n_orb:].sum(dim=1)
    valid = (a_counts == n_alpha) & (b_counts == n_beta)
    new_u = new_u[valid]
    truly_new = []
    for i in range(len(new_u)):
        h = tuple(new_u[i].cpu().tolist())
        if h not in seen:
            seen.add(h)
            truly_new.append(new_u[i])
    if truly_new:
        all_configs_gpu = torch.cat([all_configs_gpu, torch.stack(truly_new).to(device)])
    print(f'  Round {it}: +{len(truly_new)} new, total={all_configs_gpu.shape[0]}')
    if len(truly_new) == 0:
        break

print(f'Total unique configs: {all_configs_gpu.shape[0]}')

# Alpha/beta decomposition stats
u_a = torch.unique(all_configs_gpu[:, :n_orb], dim=0)
u_b = torch.unique(all_configs_gpu[:, n_orb:], dim=0)
print(f'Unique alpha: {u_a.shape[0]}, beta: {u_b.shape[0]}, product: {u_a.shape[0]*u_b.shape[0]}')

# Convert to IBM format (shared)
configs_cpu = all_configs_gpu.cpu().numpy()
cum_bs = np.zeros((len(configs_cpu), n_qubits), dtype=bool)
for s in range(len(configs_cpu)):
    for j in range(n_orb):
        cum_bs[s, n_orb-1-j] = bool(configs_cpu[s, j])
        cum_bs[s, n_qubits-1-j] = bool(configs_cpu[s, j+n_orb])

# =====================================================
# Step 2: IBM CPU diag (same configs)
# =====================================================
print('\n' + '='*60)
print('IBM CPU (solve_fermion) — same configs')
print('='*60)
ibm_t0 = time.time()
try:
    e, _, _, _ = solve_fermion(cum_bs, hcore, eri, spin_sq=0)
    ibm_E = e + nuc
    ibm_err = (ibm_E - ref_E) * 1000
    ibm_time = time.time() - ibm_t0
    print(f'  E = {ibm_E:.10f}, err = {ibm_err:.4f} mHa, t = {ibm_time:.1f}s')
except Exception as ex:
    print(f'  FAILED: {ex}')
    ibm_E = None
    ibm_time = time.time() - ibm_t0

# =====================================================
# Step 3: GPU alpha*beta diag (same configs)
# =====================================================
print('\n' + '='*60)
print('GPU alpha*beta — same configs')
print('='*60)
gpu_t0 = time.time()

# Expand
exp_a = u_a.repeat_interleave(u_b.shape[0], dim=0)
exp_b = u_b.repeat(u_a.shape[0], 1)
expanded = torch.cat([exp_a, exp_b], dim=1).to(device)
n_expanded = expanded.shape[0]
print(f'  Expanded basis: {n_expanded} configs')

try:
    if n_expanded <= DENSE_LIMIT:
        H_proj = H.matrix_elements_fast(expanded).double()
        evals, _ = gpu_eigh(H_proj, use_gpu=True)
        gpu_E = float(evals[0].cpu())
        del H_proj, evals
    else:
        print(f'  Using sparse eigsh (n={n_expanded} > {DENSE_LIMIT})')
        evals, _ = sparse_hamiltonian_eigsh(H, expanded, k=1, which='SA')
        gpu_E = float(evals[0].cpu())
        del evals
    torch.cuda.empty_cache()
    gpu_err = (gpu_E - ref_E) * 1000
    gpu_time = time.time() - gpu_t0
    print(f'  E = {gpu_E:.10f}, err = {gpu_err:.4f} mHa, t = {gpu_time:.1f}s')
except Exception as ex:
    print(f'  FAILED: {ex}')
    gpu_E = None
    gpu_time = time.time() - gpu_t0

# =====================================================
# Summary
# =====================================================
print('\n' + '='*60)
print(f'SUMMARY: {MOL} ({nq}Q) — same {all_configs_gpu.shape[0]} configs')
print('='*60)
print(f'  {\"Method\":<14} {\"Energy\":>18} {\"Error(mHa)\":>11} {\"Time\":>8} {\"Speedup\":>8}')
print(f'  {\"-\"*60}')
print(f'  {\"SCI ref\":<14} {ref_E:>18.10f} {\"0.0000\":>11} {sci.wall_time:>7.1f}s {\"--\":>8}')
if ibm_E:
    print(f'  {\"IBM CPU\":<14} {ibm_E:>18.10f} {ibm_err:>11.4f} {ibm_time:>7.1f}s {\"--\":>8}')
if gpu_E:
    speedup = ibm_time / gpu_time if ibm_time and gpu_time else 0
    print(f'  {\"GPU a*b\":<14} {gpu_E:>18.10f} {gpu_err:>11.4f} {gpu_time:>7.1f}s {speedup:>7.1f}x')
if ibm_E and gpu_E:
    diff = abs(ibm_E - gpu_E) * 1000
    print(f'\n  Energy difference (IBM vs GPU): {diff:.6f} mHa')
"

echo "=== Done: $(date) ==="
