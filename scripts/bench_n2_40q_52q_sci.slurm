#!/bin/bash
#SBATCH --job-name=n2-sci-unlim
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=12:00:00
#SBATCH --account=GOV109211
#SBATCH --output=bench_n2_40q_52q_sci_%j.log
#SBATCH --error=bench_n2_40q_52q_sci_%j.err

cd /home/leo07010/HI-NQS/nqs-sqd
echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "N2 CAS(10,20) 40Q + CAS(10,26) 52Q: SCI unlimited"

/home/leo07010/.conda/envs/cudaq-env/bin/python -u scripts/bench_n2_40q_52q_sci.py

echo "=== Done: $(date) ==="
