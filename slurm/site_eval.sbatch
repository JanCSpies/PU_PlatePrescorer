#!/bin/bash
#SBATCH --job-name=bhpu-site-eval
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export SLURM_CPUS_PER_TASK
srun .venv/bin/python scripts/15_site_descriptor_eval.py --n-estimators 20
echo "done=$(date -Is)"
