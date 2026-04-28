#!/bin/bash
#SBATCH --job-name=nqs-beta
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_nqs_beta_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
mkdir -p results/nqs_beta_reshape logs/nqs_beta_reshape

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_nqs_beta_reshape_worker.sh

echo "=== done: $(date) ==="
ls -l results/nqs_beta_reshape/
