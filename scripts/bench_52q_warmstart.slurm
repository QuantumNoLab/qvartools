#!/bin/bash
#SBATCH --job-name=52q-ws
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_52q_ws_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
mkdir -p results/52q_warmstart logs/52q_warmstart

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_52q_warmstart_worker.sh

echo "=== done: $(date) ==="
ls -l results/52q_warmstart/
