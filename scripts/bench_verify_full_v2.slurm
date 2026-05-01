#!/bin/bash
#SBATCH --job-name=verify-v2
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_verify_v2_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh
python -u scratch_verify_full_basis_v2.py
echo "=== done: $(date) ==="
