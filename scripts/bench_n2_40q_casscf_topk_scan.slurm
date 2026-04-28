#!/bin/bash
#SBATCH --job-name=n2-40q-topk
#SBATCH --account=gov114009
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/leo07010/HI-NQS/nqs-sqd/logs/bench_n2_40q_casscf_topk_scan_%j.log

echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "N2-CAS(10,20) 40Q CASSCF top-k scan: n_samples=100K, top_k in {10,15,20,30}K"

source /home/leo07010/.bashrc 2>/dev/null || true
conda activate cudaq-env 2>/dev/null || source activate cudaq-env 2>/dev/null || true

cd /home/leo07010/HI-NQS/nqs-sqd
python -u scripts/bench_n2_40q_casscf_topk_scan.py

echo "End: $(date)"
