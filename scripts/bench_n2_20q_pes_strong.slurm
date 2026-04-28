#!/bin/bash
#SBATCH --job-name=n2-20q-pes-strong
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_n2_20q_pes_strong_%j.log

echo "=== Job $SLURM_JOB_ID | launcher=$(hostname) ==="
echo "Start: $(date)"

cd /home/leo07010/HI-VQE
mkdir -p results/n2_20q_pes logs/n2_20q_pes

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_n2_20q_pes_strong_worker.sh

echo "=== done: $(date) ==="
ls -l results/n2_20q_pes/
