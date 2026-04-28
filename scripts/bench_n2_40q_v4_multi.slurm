#!/bin/bash
#SBATCH --job-name=n2-40q-v4-mg
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_n2_40q_v4_multi_%j.log

echo "=== Job $SLURM_JOB_ID | launcher=$(hostname) | nodes=$SLURM_JOB_NODELIST ==="
echo "Start: $(date)"

cd /home/leo07010/HI-VQE
mkdir -p results/n2_40q_v4_multi logs/n2_40q_v4_multi

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_n2_40q_v4_multi_worker.sh

echo "=== srun complete: $(date) ==="
ls -l results/n2_40q_v4_multi/
