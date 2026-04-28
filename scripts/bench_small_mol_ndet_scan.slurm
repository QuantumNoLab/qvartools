#!/bin/bash
#SBATCH --job-name=small-mol-ndet
#SBATCH --account=gov114009
#SBATCH --partition=large
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/home/leo07010/HI-VQE/bench_small_mol_ndet_%j.log

echo "=== Job $SLURM_JOB_ID | launcher=$(hostname) | nodes=$SLURM_JOB_NODELIST ==="
echo "Start: $(date)"

cd /home/leo07010/HI-VQE
mkdir -p results/small_mol_ndet_scan logs/small_mol_ndet_scan

# 2 nodes x 8 GPUs = 16 parallel tasks total.
# srun launches one worker per node (ntasks=2). Each worker forks 8 subprocesses
# (one per local GPU, CUDA_VISIBLE_DEVICES=0..7). Global task ID = NODE_IDX*8 + local_gpu.
srun --kill-on-bad-exit=0 bash /home/leo07010/HI-VQE/scripts/bench_small_mol_ndet_worker.sh

echo "=== srun complete: $(date) ==="
ls -l results/small_mol_ndet_scan/
