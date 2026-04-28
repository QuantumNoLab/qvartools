#!/bin/bash
#SBATCH --job-name=v3-topk
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_topk_%j.log
#SBATCH --error=bench_topk_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) ==="

/home/leo07010/GQE-MTS/.venv/bin/python -u -c "
import sys, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from src.utils.hamiltonian_cache import load_reference_energy, save_reference_energy

# Each molecule with its specific top_k
TESTS = [
    ('H2O',  50),
    ('NH3',  500),
    ('N2',   500),
    ('C2H2', 2000),
]

results = []

for mol_name, top_k in TESTS:
    print('='*70)
    print(f'  {mol_name} (top_k={top_k})')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']

    # FCI (PySCF, cached)
    fci = FCISolver().solve(H, info)
    fci_E = fci.energy

    # SCI (cached or compute once)
    cached_sci = load_reference_energy(mol_name, 'sci')
    if cached_sci:
        sci_E = cached_sci['energy']
        sci_basis = cached_sci.get('basis_size', 0)
        sci_time = cached_sci.get('wall_time', 0)
        print(f'SCI (cached): {sci_E:.10f} Ha, basis={sci_basis}')
    else:
        sci = CIPSISolver(max_iterations=30, max_basis_size=5000, expansion_size=500, convergence_threshold=1e-7).solve(H, info)
        sci_E = sci.energy
        sci_basis = sci.diag_dim
        sci_time = sci.wall_time
        save_reference_energy(mol_name, 'sci', sci_E, sci_basis, sci_time)
        print(f'SCI: {sci_E:.10f} Ha, basis={sci_basis}, t={sci_time:.1f}s (saved)')

    ref_E = fci_E if fci_E else sci_E

    # HI+NQS+SQD v3 with specific top_k, no max_basis_size
    print(f'\nRunning v3 (top_k={top_k}, no basis cap)...')
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=5000,
        top_k=top_k,
        max_basis_size=0,  # no limit
        max_iterations=15,
        convergence_threshold=1e-6,
        convergence_window=3,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    hi_time = time.time() - t0
    hi_err = (r.energy - ref_E) * 1000 if r.energy and ref_E else None

    sci_err = (sci_E - ref_E) * 1000 if ref_E else 0

    print(f'\n  {\"Method\":<18} {\"Energy\":>18} {\"err(mHa)\":>10} {\"basis\":>7} {\"time\":>7}')
    print(f'  {\"-\"*62}')
    if fci_E:
        print(f'  {\"FCI\":<18} {fci_E:>18.10f} {\"0.0000\":>10} {fci.diag_dim:>7} {fci.wall_time:>6.1f}s')
    print(f'  {\"SCI\":<18} {sci_E:>18.10f} {sci_err:>10.4f} {sci_basis:>7} {sci_time:>6.1f}s')
    if r.energy:
        print(f'  {\"v3 top_k=\"+str(top_k):<18} {r.energy:>18.10f} {hi_err:>10.4f} {r.diag_dim:>7} {hi_time:>6.1f}s')

    results.append({
        'mol': mol_name, 'nq': nq, 'top_k': top_k,
        'sci_err': sci_err, 'sci_basis': sci_basis, 'sci_t': sci_time,
        'v3_err': hi_err, 'v3_basis': r.diag_dim, 'v3_t': hi_time,
    })
    print()

print('='*70)
print('  SUMMARY')
print('='*70)
print(f'  {\"Mol\":>6} {\"Q\":>3} {\"top_k\":>6} | {\"SCI err\":>9} {\"basis\":>6} {\"time\":>6} | {\"v3 err\":>9} {\"basis\":>6} {\"time\":>6}')
print(f'  {\"-\"*66}')
for r in results:
    se = f'{r[\"sci_err\"]:.4f}'
    ve = f'{r[\"v3_err\"]:.4f}' if r['v3_err'] is not None else 'FAIL'
    print(f'  {r[\"mol\"]:>6} {r[\"nq\"]:>3} {r[\"top_k\"]:>6} | {se:>9} {r[\"sci_basis\"]:>6} {r[\"sci_t\"]:>5.0f}s | {ve:>9} {r[\"v3_basis\"]:>6} {r[\"v3_t\"]:>5.0f}s')
"

echo "=== Done: $(date) ==="
