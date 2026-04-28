#!/bin/bash
#SBATCH --job-name=strict-large
#SBATCH --partition=normal
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_strict_large_%j.log
#SBATCH --error=bench_strict_large_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "SCI: sparse eigsh for basis>10K, threshold=1e-8, no cap"

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from math import comb

MOLECULES = [
    # (name, nqs_top_k, nqs_samples)
    ('C2H2',          2000,  5000),  # 24Q
    ('C2H4',          2000,  5000),  # 28Q
    ('N2-CAS(10,20)', 5000, 10000),  # 40Q
    ('N2-CAS(10,26)', 10000, 10000), # 52Q
]

results = []

for mol_name, top_k, n_samples in MOLECULES:
    print('\n' + '='*70)
    print(f'  {mol_name}')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f'{nq}Q, {n_orb} orb, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}')

    # FCI (if feasible)
    fci = FCISolver().solve(H, info)
    fci_E = fci.energy
    if fci_E:
        print(f'FCI: {fci_E:.10f} Ha')

    # ---- SCI: strict, no cap, sparse for >10K ----
    print('\nSCI (threshold=1e-8, no cap, sparse>10K)...')
    t0 = time.time()
    sci = CIPSISolver(
        max_iterations=200,
        max_basis_size=999999,
        expansion_size=500,
        convergence_threshold=1e-8,
    ).solve(H, info)
    sci_time = time.time() - t0
    sci_E = sci.energy
    ref_E = fci_E if fci_E else sci_E
    sci_err = (sci_E - ref_E) * 1000 if ref_E else 0
    print(f'SCI: {sci_E:.10f} Ha, basis={sci.diag_dim}, t={sci_time:.1f}s')

    # ---- NQS v3: strict, no cap ----
    print(f'\nNQS v3 (top_k={top_k}, threshold=1e-8, no cap)...')
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=n_samples,
        top_k=top_k,
        max_basis_size=999999,
        max_iterations=50,
        convergence_threshold=1e-8,
        convergence_window=3,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    hi_time = time.time() - t0
    hi_err = (r.energy - ref_E) * 1000 if r.energy and ref_E else None

    print(f'\n  {\"Method\":<18} {\"Energy\":>18} {\"err(mHa)\":>10} {\"basis\":>7} {\"time\":>8}')
    print(f'  {\"-\"*63}')
    if fci_E:
        print(f'  {\"FCI\":<18} {fci_E:>18.10f} {\"0.000000\":>10} {fci.diag_dim:>7} {fci.wall_time:>7.1f}s')
    print(f'  {\"SCI (strict)\":<18} {sci_E:>18.10f} {sci_err:>10.6f} {sci.diag_dim:>7} {sci_time:>7.1f}s')
    if r.energy:
        print(f'  {\"NQS v3\":<18} {r.energy:>18.10f} {hi_err:>10.6f} {r.diag_dim:>7} {hi_time:>7.1f}s')

    results.append({
        'mol': mol_name, 'nq': nq,
        'fci_E': fci_E,
        'sci_E': sci_E, 'sci_err': sci_err, 'sci_basis': sci.diag_dim, 'sci_t': sci_time,
        'nqs_E': r.energy, 'nqs_err': hi_err, 'nqs_basis': r.diag_dim, 'nqs_t': hi_time,
    })
    print()

print('='*70)
print('  STRICT CONVERGENCE SUMMARY (large systems)')
print('='*70)
print(f'  {\"Mol\":>15} {\"Q\":>3} | {\"SCI E\":>18} {\"err\":>10} {\"basis\":>7} {\"time\":>8} | {\"NQS E\":>18} {\"err\":>10} {\"basis\":>7} {\"time\":>8}')
print(f'  {\"-\"*99}')
for r in results:
    se = f'{r[\"sci_err\"]:.6f}'
    ne = f'{r[\"nqs_err\"]:.6f}' if r['nqs_err'] is not None else 'FAIL'
    sci_e = f'{r[\"sci_E\"]:.10f}'
    nqs_e = f'{r[\"nqs_E\"]:.10f}' if r['nqs_E'] else 'FAIL'
    print(f'  {r[\"mol\"]:>15} {r[\"nq\"]:>3} | {sci_e:>18} {se:>10} {r[\"sci_basis\"]:>7} {r[\"sci_t\"]:>7.0f}s | {nqs_e:>18} {ne:>10} {r[\"nqs_basis\"]:>7} {r[\"nqs_t\"]:>7.0f}s')
"

echo "=== Done: $(date) ==="
