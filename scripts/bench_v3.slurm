#!/bin/bash
#SBATCH --job-name=v3-pt2
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_v3_%j.log
#SBATCH --error=bench_v3_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="
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

print('v3: NQS + PT2 selection + solve_fermion')
print('FCI now uses PySCF (same backend as solve_fermion)')
print()

results = []

for mol_name in MOLECULES:
    print('='*70)
    print(f'  {mol_name}')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f'{nq}Q, Hilbert={hilbert:,}')

    # FCI (PySCF)
    fci = FCISolver().solve(H, info)
    fci_E = fci.energy
    ref_E = fci_E
    if fci_E:
        print(f'FCI (PySCF): {fci_E:.10f} Ha')
    else:
        # Fallback to SCI as reference
        sci_ref = CIPSISolver(max_iterations=30, max_basis_size=5000, expansion_size=500).solve(H, info)
        ref_E = sci_ref.energy
        print(f'SCI ref: {ref_E:.10f} Ha (FCI too large)')

    # SCI
    sci = CIPSISolver(max_iterations=30, max_basis_size=5000, expansion_size=500, convergence_threshold=1e-7).solve(H, info)
    sci_err = (sci.energy - ref_E) * 1000
    print(f'SCI: {sci.energy:.10f}, err={sci_err:.4f} mHa, basis={sci.diag_dim}, t={sci.wall_time:.1f}s')

    # HI+NQS+SQD v3
    print()
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=5000,
        top_k=2000,
        max_basis_size=10000,
        max_iterations=15,
        convergence_threshold=1e-6,
        convergence_window=3,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    hi_time = time.time() - t0
    hi_err = (r.energy - ref_E) * 1000 if r.energy else None

    print(f'\n  {\"Method\":<18} {\"Energy\":>18} {\"err(mHa)\":>10} {\"basis\":>7} {\"time\":>7}')
    print(f'  {\"-\"*62}')
    if fci_E:
        print(f'  {\"FCI (PySCF)\":<18} {fci_E:>18.10f} {\"0.0000\":>10} {fci.diag_dim:>7} {fci.wall_time:>6.1f}s')
    print(f'  {\"SCI (CIPSI)\":<18} {sci.energy:>18.10f} {sci_err:>10.4f} {sci.diag_dim:>7} {sci.wall_time:>6.1f}s')
    if r.energy:
        hi_e_str = f'{r.energy:>18.10f}'
        hi_err_str = f'{hi_err:.4f}'
        below_fci = ' *** BELOW FCI' if fci_E and r.energy < fci_E - 0.0001 else ''
        print(f'  {\"HI+NQS+SQD v3\":<18} {hi_e_str} {hi_err_str:>10} {r.diag_dim:>7} {hi_time:>6.1f}s{below_fci}')

    results.append({
        'mol': mol_name, 'nq': nq,
        'sci_err': sci_err, 'sci_basis': sci.diag_dim, 'sci_t': sci.wall_time,
        'v3_err': hi_err, 'v3_basis': r.diag_dim, 'v3_t': hi_time,
    })
    print()

print('='*70)
print('  SUMMARY')
print('='*70)
print(f'  {\"Mol\":>6} {\"Q\":>3} | {\"SCI err\":>9} {\"basis\":>6} {\"time\":>6} | {\"v3 err\":>9} {\"basis\":>6} {\"time\":>6}')
print(f'  {\"-\"*60}')
for r in results:
    se = f'{r[\"sci_err\"]:.4f}' if r['sci_err'] is not None else 'N/A'
    ve = f'{r[\"v3_err\"]:.4f}' if r['v3_err'] is not None else 'N/A'
    print(f'  {r[\"mol\"]:>6} {r[\"nq\"]:>3} | {se:>9} {r[\"sci_basis\"]:>6} {r[\"sci_t\"]:>5.0f}s | {ve:>9} {r[\"v3_basis\"]:>6} {r[\"v3_t\"]:>5.0f}s')
"

echo "=== Done: $(date) ==="
