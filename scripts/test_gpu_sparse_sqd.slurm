#!/bin/bash
#SBATCH --job-name=test-gpu-sqd
#SBATCH --account=gov114009
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/leo07010/HI-NQS/nqs-sqd/test_gpu_sparse_sqd_%j.log

echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "GPU sparse SQD correctness + speed test"

source /home/leo07010/.bashrc 2>/dev/null || true
conda activate cudaq-env 2>/dev/null || source activate cudaq-env 2>/dev/null || true

cd /home/leo07010/HI-NQS/nqs-sqd
python -u scripts/test_gpu_sparse_sqd.py

echo "End: $(date)"
