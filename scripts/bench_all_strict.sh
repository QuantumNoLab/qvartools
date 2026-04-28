#!/bin/bash
#SBATCH --job-name=nqs_strict
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --account=GOV114009
#SBATCH --time=12:00:00
#SBATCH --output=bench_all_strict_%j.log
#SBATCH --error=bench_all_strict_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd
mkdir -p results

echo "=== Job $SLURM_JOB_ID | $(hostname) ==="
echo "Start: $(date)"
echo "All SCI + NQS v3 with strict convergence (1e-8, no basis cap)"
echo ""

PY=/home/leo07010/GQE-MTS/.venv/bin/python

# Run SCI and NQS for each molecule on separate GPUs in parallel
# Small molecules: SCI + NQS sequential (fast enough)
# Large molecules: SCI and NQS on separate GPUs in parallel

# ---- H2O (14Q) ----
CUDA_VISIBLE_DEVICES=0 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
H, info = get_molecule('H2O')
fci = FCISolver().solve(H, info)
print(f'H2O FCI: {fci.energy:.10f} Ha')
sci = CIPSISolver(max_iterations=200, max_basis_size=999999, expansion_size=500, convergence_threshold=1e-8).solve(H, info)
print(f'H2O SCI: {sci.energy:.10f} Ha, basis={sci.diag_dim}, t={sci.wall_time:.1f}s')
cfg = HINQSSQDConfig(n_samples=5000, top_k=50, max_basis_size=999999, max_iterations=50, convergence_threshold=1e-8, convergence_window=3)
r = run_hi_nqs_sqd(H, info, config=cfg)
err = (r.energy - fci.energy) * 1000 if r.energy and fci.energy else None
print(f'H2O NQS: {r.energy:.10f} Ha, err={err:.6f} mHa, basis={r.diag_dim}')
" &

# ---- NH3 (16Q) ----
CUDA_VISIBLE_DEVICES=1 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
H, info = get_molecule('NH3')
fci = FCISolver().solve(H, info)
print(f'NH3 FCI: {fci.energy:.10f} Ha')
sci = CIPSISolver(max_iterations=200, max_basis_size=999999, expansion_size=500, convergence_threshold=1e-8).solve(H, info)
print(f'NH3 SCI: {sci.energy:.10f} Ha, basis={sci.diag_dim}, t={sci.wall_time:.1f}s')
cfg = HINQSSQDConfig(n_samples=5000, top_k=500, max_basis_size=999999, max_iterations=50, convergence_threshold=1e-8, convergence_window=3)
r = run_hi_nqs_sqd(H, info, config=cfg)
err = (r.energy - fci.energy) * 1000 if r.energy and fci.energy else None
print(f'NH3 NQS: {r.energy:.10f} Ha, err={err:.6f} mHa, basis={r.diag_dim}')
" &

# ---- N2 (20Q) ----
CUDA_VISIBLE_DEVICES=2 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.sci import CIPSISolver
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
H, info = get_molecule('N2')
fci = FCISolver().solve(H, info)
print(f'N2 FCI: {fci.energy:.10f} Ha')
sci = CIPSISolver(max_iterations=200, max_basis_size=999999, expansion_size=500, convergence_threshold=1e-8).solve(H, info)
print(f'N2 SCI: {sci.energy:.10f} Ha, basis={sci.diag_dim}, t={sci.wall_time:.1f}s')
cfg = HINQSSQDConfig(n_samples=5000, top_k=500, max_basis_size=999999, max_iterations=50, convergence_threshold=1e-8, convergence_window=3)
r = run_hi_nqs_sqd(H, info, config=cfg)
err = (r.energy - fci.energy) * 1000 if r.energy and fci.energy else None
print(f'N2 NQS: {r.energy:.10f} Ha, err={err:.6f} mHa, basis={r.diag_dim}')
" &

# ---- C2H2 (24Q) — SCI on GPU3, NQS on GPU4 ----
CUDA_VISIBLE_DEVICES=3 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver
H, info = get_molecule('C2H2')
sci = CIPSISolver(max_iterations=200, max_basis_size=999999, expansion_size=500, convergence_threshold=1e-8).solve(H, info)
print(f'C2H2 SCI: {sci.energy:.10f} Ha, basis={sci.diag_dim}, t={sci.wall_time:.1f}s')
" &

CUDA_VISIBLE_DEVICES=4 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
H, info = get_molecule('C2H2')
cfg = HINQSSQDConfig(n_samples=5000, top_k=2000, max_basis_size=999999, max_iterations=50, convergence_threshold=1e-8, convergence_window=3)
r = run_hi_nqs_sqd(H, info, config=cfg)
print(f'C2H2 NQS: {r.energy:.10f} Ha, basis={r.diag_dim}')
" &

# ---- C2H4 (28Q) — SCI on GPU5, NQS on GPU6 ----
CUDA_VISIBLE_DEVICES=5 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.sci import CIPSISolver
H, info = get_molecule('C2H4')
sci = CIPSISolver(max_iterations=200, max_basis_size=999999, expansion_size=500, convergence_threshold=1e-8).solve(H, info)
print(f'C2H4 SCI: {sci.energy:.10f} Ha, basis={sci.diag_dim}, t={sci.wall_time:.1f}s')
" &

CUDA_VISIBLE_DEVICES=6 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
H, info = get_molecule('C2H4')
cfg = HINQSSQDConfig(n_samples=5000, top_k=2000, max_basis_size=999999, max_iterations=50, convergence_threshold=1e-8, convergence_window=3)
r = run_hi_nqs_sqd(H, info, config=cfg)
print(f'C2H4 NQS: {r.energy:.10f} Ha, basis={r.diag_dim}')
" &

# ---- N2-40Q — NQS on GPU7 (SCI too slow, use previous data) ----
CUDA_VISIBLE_DEVICES=7 $PY -u -c "
import sys; sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import run_hi_nqs_sqd, HINQSSQDConfig
H, info = get_molecule('N2-CAS(10,20)')
print(f'N2-40Q SCI ref (previous): -109.2103366923 Ha, basis=10000')
cfg = HINQSSQDConfig(n_samples=10000, top_k=5000, max_basis_size=999999, max_iterations=50, convergence_threshold=1e-8, convergence_window=3)
r = run_hi_nqs_sqd(H, info, config=cfg)
print(f'N2-40Q NQS: {r.energy:.10f} Ha, basis={r.diag_dim}')
" &

echo "All tasks launched on separate GPUs, waiting..."
wait

echo ""
echo "=== All done: $(date) ==="
