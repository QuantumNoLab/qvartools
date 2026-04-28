#!/bin/bash
#SBATCH --job-name=n2-52q-mg
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_n2_52q_multigpu_%j.log

echo "=== Job $SLURM_JOB_ID ==="; date
cd /home/leo07010/HI-VQE
mkdir -p results/n2_52q_multigpu logs/n2_52q_multigpu

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_n2_52q_multigpu_worker.sh

echo "=== done: $(date) ==="
