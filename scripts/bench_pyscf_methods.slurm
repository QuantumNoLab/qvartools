#!/bin/bash
#SBATCH --job-name=pyscf-4method
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:H100:4
#SBATCH --time=12:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_pyscf_methods_%j.log
#SBATCH --error=bench_pyscf_methods_%j.err

cd /home/leo07010/HI-NQS/nqs-sqd
echo "=== Job $SLURM_JOB_ID | $(hostname) | GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "PySCF 4-method parallel: CCSD | CCSD(T) | CASSCF | HCI"
echo "Molecules: H2O NH3 N2(20Q) C2H2(24Q) C2H4(28Q) N2-CAS(40Q) N2-CAS(52Q)"

# Allow PySCF to use 12 threads per process (4 processes × 12 = 48 CPUs)
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

/home/leo07010/.conda/envs/cudaq-env/bin/python -u scripts/bench_pyscf_methods.py

echo "=== Done: $(date) ==="
