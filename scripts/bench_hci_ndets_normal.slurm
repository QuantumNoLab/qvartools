#!/bin/bash
#SBATCH --job-name=hci-ndets
#SBATCH --account=gov114009
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/leo07010/HI-NQS/nqs-sqd/bench_hci_ndets_%j.log

echo "=== Job $SLURM_JOB_ID | $(hostname) ==="
echo "Start: $(date)"

source /home/leo07010/.bashrc 2>/dev/null || true
conda activate cudaq-env 2>/dev/null || source activate cudaq-env 2>/dev/null || true

cd /home/leo07010/HI-NQS/nqs-sqd
python scripts/bench_hci_ndets.py

echo "End: $(date)"
