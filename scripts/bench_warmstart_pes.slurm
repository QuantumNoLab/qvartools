#!/bin/bash
#SBATCH --job-name=ws-pes
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_ws_pes_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
mkdir -p results/warmstart_pes logs/warmstart_pes

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_warmstart_pes_worker.sh

echo "=== done: $(date) ==="
ls -l results/warmstart_pes/
