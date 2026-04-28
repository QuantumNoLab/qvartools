#!/bin/bash
#SBATCH --job-name=hinqs-n2-52q
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=run_n2_52q_%j.log
#SBATCH --error=run_n2_52q_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from math import comb

mol_name = 'N2-CAS(10,26)'
print(f'Loading {mol_name}...')
H, info = get_molecule(mol_name)
nq = info['n_qubits']
n_orb = H.n_orbitals
n_alpha, n_beta = H.n_alpha, H.n_beta
hilbert = comb(n_orb, n_alpha) * comb(n_orb, n_beta)

print(f'{mol_name}: {nq}Q, {n_orb} orbitals, ({n_alpha},{n_beta}) electrons')
print(f'Hilbert space: {hilbert:,} configurations')
print(f'Nuclear repulsion: {H.integrals.nuclear_repulsion:.8f} Ha')
print()

# HI+NQS+SQD — no FCI/SCI reference needed
t0 = time.time()
cfg = HINQSSQDConfig(
    n_samples=5000,
    max_iterations=20,
    convergence_threshold=1e-6,
    convergence_window=3,
    num_batches=5,
)
r = run_hi_nqs_sqd(H, info, config=cfg)
wall = time.time() - t0

print(f'\n{\"=\"*60}')
print(f'N2 CAS(10,26) — 52 QUBITS')
print(f'{\"=\"*60}')
print(f'Best energy:  {r.energy:.10f} Ha' if r.energy else 'FAILED')
print(f'Basis size:   {r.diag_dim}')
print(f'Converged:    {r.converged}')
print(f'Iterations:   {r.metadata.get(\"iterations\", \"?\")}')
print(f'Wall time:    {wall:.1f}s')
if r.metadata.get('energy_history'):
    print(f'\nEnergy history:')
    for i, e in enumerate(r.metadata['energy_history']):
        print(f'  Iter {i}: {e:.10f} Ha')
"

echo "=== Done: $(date) ==="
