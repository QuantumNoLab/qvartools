#!/bin/bash
#SBATCH --job-name=n2-52q-cfg
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_n2_52q_config_scan_%j.log

echo "=== Job $SLURM_JOB_ID | launcher=$(hostname) ==="
echo "Start: $(date)"

cd /home/leo07010/HI-VQE
mkdir -p results/n2_52q_config_scan logs/n2_52q_config_scan

srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_n2_52q_config_scan_worker.sh

echo "=== done: $(date) ==="
ls -l results/n2_52q_config_scan/
