#!/bin/bash
#SBATCH --job-name=ws-bug
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_warmstart_bug_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
source /home/leo07010/HI-VQE/scripts/setup_cuda_env.sh
python -u scratch_test_warmstart_bug.py
echo "=== done: $(date) ==="
