#!/bin/bash
#SBATCH --job-name=v2-tune
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_v2_tune_%j.log
#SBATCH --error=bench_v2_tune_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

echo "=== Job $SLURM_JOB_ID on $(hostname) GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
echo "Start: $(date)"

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from math import comb

MOLECULES = ['H2O', 'NH3', 'N2', 'C2H2']

THRESHOLDS = [1e-6, 1e-10, 1e-14, 0.0]  # 0.0 = no eviction

for mol_name in MOLECULES:
    print('\n' + '='*70)
    print(f'  {mol_name}')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f'{nq}Q, Hilbert={hilbert:,}')

    # FCI reference
    fci = FCISolver().solve(H, info)
    fci_E = fci.energy
    if fci_E:
        print(f'FCI: {fci_E:.10f} Ha')

    # SCI reference
    sci = CIPSISolver(max_iterations=30, max_basis_size=5000, expansion_size=500, convergence_threshold=1e-7).solve(H, info)
    ref_E = fci_E if fci_E else sci.energy
    sci_err = (sci.energy - ref_E) * 1000
    print(f'SCI: {sci.energy:.10f} Ha, err={sci_err:.4f} mHa, basis={sci.diag_dim}, t={sci.wall_time:.1f}s')

    # Test each threshold
    print(f'\n  {\"threshold\":<12} {\"Energy\":>18} {\"err(mHa)\":>10} {\"basis\":>7} {\"evicted\":>8} {\"rejected\":>8} {\"time\":>7}')
    print(f'  {\"-\"*72}')
    print(f'  {\"SCI\":<12} {sci.energy:>18.10f} {sci_err:>10.4f} {sci.diag_dim:>7} {\"--\":>8} {\"--\":>8} {sci.wall_time:>6.1f}s')

    for thr in THRESHOLDS:
        thr_name = f'{thr:.0e}' if thr > 0 else 'none'
        cfg = HINQSSQDConfig(
            n_samples=5000,
            max_iterations=15,
            convergence_threshold=1e-6,
            convergence_window=3,
            eviction_threshold=thr,
            energy_filter_margin=1.0,
        )
        t0 = time.time()
        r = run_hi_nqs_sqd(H, info, config=cfg)
        wall = time.time() - t0
        err = (r.energy - ref_E) * 1000 if r.energy else None

        # Count total evicted/rejected from log (approximate from metadata)
        iters = r.metadata.get('iterations', 0)
        err_s = f'{err:.4f}' if err is not None else 'FAIL'
        print(f'  {thr_name:<12} {r.energy:>18.10f} {err_s:>10} {r.diag_dim:>7} {\"\":>8} {\"\":>8} {wall:>6.1f}s')
"

echo "=== Done: $(date) ==="
