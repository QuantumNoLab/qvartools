#!/bin/bash
#SBATCH --job-name=v2-vs-sci
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_v2_vs_sci_%j.log
#SBATCH --error=bench_v2_vs_sci_%j.err

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
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
from math import comb

MOLECULES = ['H2O', 'BeH2', 'NH3', 'N2', 'C2H2']
results = []

for mol_name in MOLECULES:
    print('\n' + '='*70)
    print(f'  {mol_name}')
    print('='*70)

    H, info = get_molecule(mol_name)
    nq = info['n_qubits']
    n_orb = H.n_orbitals
    hilbert = comb(n_orb, H.n_alpha) * comb(n_orb, H.n_beta)
    print(f'{mol_name}: {nq}Q, {n_orb} orbitals, ({H.n_alpha},{H.n_beta})e, Hilbert={hilbert:,}')

    # ---- FCI (if feasible) ----
    fci_E = None
    fci = FCISolver().solve(H, info)
    if fci.energy is not None:
        fci_E = fci.energy
        print(f'FCI: {fci_E:.10f} Ha')

    # ---- SCI ----
    print('\nRunning SCI...')
    t0 = time.time()
    sci = CIPSISolver(max_iterations=30, max_basis_size=5000, expansion_size=500, convergence_threshold=1e-7).solve(H, info)
    sci_time = time.time() - t0
    sci_E = sci.energy
    ref_E = fci_E if fci_E is not None else sci_E
    sci_err = (sci_E - ref_E) * 1000 if ref_E else None
    print(f'SCI: {sci_E:.10f} Ha, basis={sci.diag_dim}, t={sci_time:.1f}s')

    # ---- HI+NQS+SQD v2 ----
    print('\nRunning HI+NQS+SQD v2...')
    t0 = time.time()
    cfg = HINQSSQDConfig(
        n_samples=5000,
        max_iterations=20,
        convergence_threshold=1e-6,
        convergence_window=3,
    )
    r = run_hi_nqs_sqd(H, info, config=cfg)
    hi_time = time.time() - t0
    hi_err = (r.energy - ref_E) * 1000 if r.energy and ref_E else None

    # ---- Summary ----
    print(f'\n--- {mol_name} ({nq}Q) ---')
    print(f'  {\"Method\":<20} {\"Energy\":>18} {\"Error(mHa)\":>11} {\"Basis\":>7} {\"Time\":>8}')
    print(f'  {\"-\"*66}')
    if fci_E:
        print(f'  {\"FCI\":<20} {fci_E:>18.10f} {\"0.0000\":>11} {fci.diag_dim:>7} {fci.wall_time:>7.1f}s')
    print(f'  {\"SCI (CIPSI)\":<20} {sci_E:>18.10f} {sci_err:>11.4f} {sci.diag_dim:>7} {sci_time:>7.1f}s')
    if r.energy:
        print(f'  {\"HI+NQS+SQD v2\":<20} {r.energy:>18.10f} {hi_err:>11.4f} {r.diag_dim:>7} {hi_time:>7.1f}s')
    else:
        print(f'  {\"HI+NQS+SQD v2\":<20} {\"FAILED\":>18}')

    chem = 1.6
    if hi_err is not None:
        winner = 'HI-NQS' if abs(hi_err) <= abs(sci_err) else 'SCI'
        print(f'  Winner: {winner} | HI-NQS chem_acc: {\"YES\" if abs(hi_err)<chem else \"NO\"}')

    results.append({
        'molecule': mol_name, 'qubits': nq,
        'sci_energy': sci_E, 'sci_err': sci_err, 'sci_basis': sci.diag_dim, 'sci_time': sci_time,
        'hi_energy': r.energy, 'hi_err': hi_err, 'hi_basis': r.diag_dim, 'hi_time': hi_time,
    })

# ---- Final table ----
print('\n' + '='*70)
print('  FINAL COMPARISON TABLE')
print('='*70)
print(f'  {\"Molecule\":>8} {\"Q\":>4} {\"SCI err\":>10} {\"SCI t\":>8} {\"v2 err\":>10} {\"v2 t\":>8} {\"Winner\":>8}')
print(f'  {\"-\"*58}')
for r in results:
    sci_s = f'{r[\"sci_err\"]:.4f}' if r['sci_err'] is not None else 'N/A'
    hi_s = f'{r[\"hi_err\"]:.4f}' if r['hi_err'] is not None else 'N/A'
    winner = ''
    if r['hi_err'] is not None and r['sci_err'] is not None:
        winner = 'v2' if abs(r['hi_err']) <= abs(r['sci_err']) else 'SCI'
    print(f'  {r[\"molecule\"]:>8} {r[\"qubits\"]:>4} {sci_s:>10} {r[\"sci_time\"]:>7.0f}s {hi_s:>10} {r[\"hi_time\"]:>7.0f}s {winner:>8}')
"

echo "=== Done: $(date) ==="
