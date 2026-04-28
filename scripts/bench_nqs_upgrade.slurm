#!/bin/bash
#SBATCH --job-name=nqs-upgrade
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_nqs_upgrade_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
mkdir -p results/nqs_upgrade logs/nqs_upgrade

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_nqs_upgrade_worker.sh

echo "=== done: $(date) ==="
ls -l results/nqs_upgrade/
