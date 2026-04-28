#!/bin/bash
#SBATCH --job-name=v3-40q-52q
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=24:00:00
#SBATCH --account=GOV109211
#SBATCH --output=run_40q_52q_v3_%j.log
#SBATCH --error=run_40q_52q_v3_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
echo "Start: $(date)"

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from math import comb

# 40Q: SCI = -109.2103366923 Ha, basis=10000 (from previous run)
# 52Q: no SCI data
TESTS = [
    ('N2-CAS(10,20)', 5000, 20),   # 40Q, top_k=5000
    ('N2-CAS(10,26)', 10000, 20),  # 52Q, top_k=10000
]

for mol_name, top_k, max_iter in TESTS:
    print('\n' + '='*70)
    print(f'  {mol_name} (top_k={top_k})')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f'{nq}Q, {n_orb} orbitals, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}')

    if mol_name == 'N2-CAS(10,20)':
        print(f'SCI reference (from cache): -109.2103366923 Ha, basis=10000')

    print(f'\nRunning HI+NQS+SQD v3 (top_k={top_k}, no basis cap)...')
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=10000,
        top_k=top_k,
        max_basis_size=0,
        max_iterations=max_iter,
        convergence_threshold=1e-6,
        convergence_window=3,
        nqs_steps=20,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    wall = time.time() - t0

    print(f'\n{\"=\"*60}')
    print(f'{mol_name} ({nq}Q) RESULTS')
    print(f'{\"=\"*60}')
    if mol_name == 'N2-CAS(10,20)':
        sci_E = -109.2103366923
        err_sci = (r.energy - sci_E) * 1000 if r.energy else None
        print(f'SCI ref:     {sci_E:.10f} Ha (basis=10000)')
        if err_sci is not None:
            print(f'Error vs SCI: {err_sci:.4f} mHa')
    print(f'Best energy: {r.energy:.10f} Ha' if r.energy else 'FAILED')
    print(f'Basis:       {r.diag_dim}')
    print(f'Converged:   {r.converged}')
    print(f'Iterations:  {r.metadata.get(\"iterations\", \"?\")}')
    print(f'Wall time:   {wall:.1f}s')
    if r.metadata.get('energy_history'):
        print(f'\nEnergy history:')
        for i, e in enumerate(r.metadata['energy_history']):
            print(f'  Iter {i}: {e:.10f} Ha')
"

echo "=== Done: $(date) ==="
