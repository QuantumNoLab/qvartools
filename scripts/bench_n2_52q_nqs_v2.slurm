#!/bin/bash
#SBATCH --job-name=n2-52q-v2
#SBATCH --account=mst114318
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/leo07010/HI-NQS/nqs-sqd/bench_n2_52q_nqs_v2_%j.log

echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "N2-CAS(10,26) 52Q HI-NQS v3 GPU-PT2: Config E/F/G"

source /home/leo07010/.bashrc 2>/dev/null || true
conda activate cudaq-env 2>/dev/null || source activate cudaq-env 2>/dev/null || true

cd /home/leo07010/HI-NQS/nqs-sqd
python scripts/bench_n2_52q_nqs_v2.py

echo "End: $(date)"
