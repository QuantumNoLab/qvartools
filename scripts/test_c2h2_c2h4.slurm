#!/bin/bash
#SBATCH --job-name=hinqs-c2h2-c2h4
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=test_c2h2_c2h4_%j.log
#SBATCH --error=test_c2h2_c2h4_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start:  $(date)"
echo "================"

cd /home/leo07010/NQS-SQD/nqs-sqd

/home/leo07010/GQE-MTS/.venv/bin/python -c "
import sys, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from src.utils.hamiltonian_cache import load_reference_energy, save_reference_energy

MOLECULES = ['C2H2', 'C2H4']

for mol_name in MOLECULES:
    print('\n' + '='*70)
    print(f'  {mol_name}')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    print(f'{mol_name}: {nq}Q, {H.n_orbitals} orbitals, {H.n_alpha}a {H.n_beta}b')

    # SCI reference — use cache if available
    cached = load_reference_energy(mol_name, method='sci')
    if cached is not None:
        sci_E = cached['energy']
        sci_basis = cached.get('basis_size', 0)
        sci_time = cached.get('wall_time', 0)
        print(f'SCI ref (cached): {sci_E:.10f} Ha (basis={sci_basis}, t={sci_time:.1f}s)')
    else:
        print('Computing SCI reference (first time, will be cached)...')
        sci = CIPSISolver(
            max_iterations=50, max_basis_size=10000,
            expansion_size=500, convergence_threshold=1e-7,
        ).solve(H, info)
        sci_E = sci.energy
        sci_basis = sci.diag_dim
        sci_time = sci.wall_time
        save_reference_energy(mol_name, 'sci', sci_E, sci_basis, sci_time)
        print(f'SCI ref: {sci_E:.10f} Ha (basis={sci_basis}, t={sci_time:.1f}s) [saved to cache]')

    # HI+NQS+SQD
    print('\nRunning HI+NQS+SQD (no recovery, IBM solve_fermion)...')
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=5000,
        max_iterations=30,
        convergence_threshold=1e-6,
        convergence_window=3,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    wall = time.time() - t0

    hi_err = (r.energy - sci_E) * 1000 if r.energy and sci_E else None

    print(f'\n--- {mol_name} ({nq}Q) RESULTS ---')
    print(f'SCI ref:     {sci_E:.10f} Ha (basis={sci_basis})')
    print(f'HI+NQS+SQD: {r.energy:.10f} Ha' if r.energy else 'HI+NQS+SQD: FAILED')
    if hi_err is not None:
        chem = 'YES' if abs(hi_err) < 1.6 else 'NO'
        print(f'Error:       {hi_err:.4f} mHa (chemical accuracy: {chem})')
    print(f'Basis:       {r.diag_dim}')
    print(f'Iterations:  {r.metadata.get(\"iterations\", \"?\")}')
    print(f'Converged:   {r.converged}')
    print(f'Wall time:   {wall:.1f}s')
"

echo "=== Done: $(date) ==="
