#!/bin/bash
#SBATCH --job-name=hinqs-parallel
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=test_parallel_%j.log
#SBATCH --error=test_parallel_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from src.utils.hamiltonian_cache import load_reference_energy, save_reference_energy
from src.solvers.sci import CIPSISolver

MOLECULES = ['C2H2', 'C2H4']

for mol_name in MOLECULES:
    print('\n' + '='*70)
    print(f'  {mol_name} (parallel SQD)')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    print(f'{mol_name}: {nq}Q, {H.n_orbitals} orbitals, {H.n_alpha}a {H.n_beta}b')

    # SCI reference (cached)
    cached = load_reference_energy(mol_name, method='sci')
    if cached is not None:
        sci_E = cached['energy']
        print(f'SCI ref (cached): {sci_E:.10f} Ha')
    else:
        print('Computing SCI reference...')
        sci = CIPSISolver(max_iterations=50, max_basis_size=10000, expansion_size=500, convergence_threshold=1e-7).solve(H, info)
        sci_E = sci.energy
        save_reference_energy(mol_name, 'sci', sci_E, sci.diag_dim, sci.wall_time)
        print(f'SCI ref: {sci_E:.10f} Ha (saved to cache)')

    # HI+NQS+SQD with parallel batches
    print('\nRunning HI+NQS+SQD (5 parallel batches)...')
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=5000,
        max_iterations=30,
        convergence_threshold=1e-6,
        convergence_window=3,
        num_batches=5,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    wall = time.time() - t0

    hi_err = (r.energy - sci_E) * 1000 if r.energy and sci_E else None

    print(f'\n--- {mol_name} ({nq}Q) RESULTS ---')
    print(f'SCI ref:     {sci_E:.10f} Ha')
    if r.energy:
        print(f'HI+NQS+SQD: {r.energy:.10f} Ha')
    if hi_err is not None:
        chem = 'YES' if abs(hi_err) < 1.6 else 'NO'
        print(f'Error:       {hi_err:.4f} mHa (chemical accuracy: {chem})')
    print(f'Basis:       {r.diag_dim}')
    print(f'Converged:   {r.converged}')
    print(f'Wall time:   {wall:.1f}s')
"

echo "=== Done: $(date) ==="
