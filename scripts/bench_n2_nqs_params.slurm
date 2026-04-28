#!/bin/bash
#SBATCH --job-name=n2-nqs-sweep
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00
#SBATCH --account=gov114009
#SBATCH --output=bench_n2_nqs_params_%j.log
#SBATCH --error=bench_n2_nqs_params_%j.err

cd /home/leo07010/HI-NQS/nqs-sqd
echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "N2 40Q/52Q HI-NQS v3 parameter sweep: top_k={10K,15K} x n_samples={20K,30K}"

/home/leo07010/.conda/envs/cudaq-env/bin/python -u scripts/bench_n2_nqs_params.py

echo "=== Done: $(date) ==="
