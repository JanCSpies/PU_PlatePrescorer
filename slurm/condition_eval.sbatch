#!/bin/bash
#SBATCH --job-name=bhpu-cond-eval
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) start=$(date -Is)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 SLURM_CPUS_PER_TASK
srun .venv/bin/python scripts/17_condition_aware_eval.py --conditions-per-pair 10 --n-estimators 25
echo "done=$(date -Is)"
