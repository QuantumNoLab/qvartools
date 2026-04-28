#!/bin/bash
#SBATCH --job-name=hinqs-cr2
#SBATCH --account=gov114009
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/leo07010/HI-NQS/nqs-sqd/logs/bench_cr2_pes_rerun_%j.log

echo "=== Job $SLURM_JOB_ID | $(hostname) | GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="
echo "Start: $(date)"
echo "Cr2 PES HI-NQS rerun: CAS(12,20) cc-pVDZ 40Q, 7 bond lengths"

source /home/leo07010/.bashrc 2>/dev/null || true
conda activate cudaq-env 2>/dev/null || source activate cudaq-env 2>/dev/null || true

cd /home/leo07010/HI-NQS/nqs-sqd
python -u scripts/bench_cr2_pes_rerun.py

echo "End: $(date)"
