#!/bin/bash
#SBATCH --job-name=missing
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV109211
#SBATCH --output=missing_%j.log
#SBATCH --error=missing_%j.err

cd /home/leo07010/NQS-SQD/nqs-sqd

python -u -c "
import sys, time
sys.path.insert(0, '.')
from src.molecules import get_molecule
from src.solvers.ccsd import CCSDSolver, CCSDTSolver

# C2H2: need CCSD energy and time
H, info = get_molecule('C2H2')
print('=== C2H2 (24Q) ===', flush=True)
t0 = time.time()
r = CCSDSolver().solve(H, info)
print(f'  CCSD:    E={r.energy:.10f}, time={time.time()-t0:.1f}s', flush=True)

# H2S: need CCSD and CCSD(T) time
H, info = get_molecule('H2S')
print('=== H2S (26Q) ===', flush=True)
t0 = time.time()
r = CCSDSolver().solve(H, info)
print(f'  CCSD:    E={r.energy:.10f}, time={time.time()-t0:.1f}s', flush=True)
t0 = time.time()
r = CCSDTSolver().solve(H, info)
print(f'  CCSD(T): E={r.energy:.10f}, time={time.time()-t0:.1f}s', flush=True)

# C2H4: need CCSD energy and time
H, info = get_molecule('C2H4')
print('=== C2H4 (28Q) ===', flush=True)
t0 = time.time()
r = CCSDSolver().solve(H, info)
print(f'  CCSD:    E={r.energy:.10f}, time={time.time()-t0:.1f}s', flush=True)

print('Done!', flush=True)
"
