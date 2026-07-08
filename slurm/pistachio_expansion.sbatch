#!/bin/bash
# Positive-corpus expansion eval: lit vs lit+ORD vs lit+ORD+Pistachio (ECFP, IW-PU).
#SBATCH --job-name=bhpu-pist-exp
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) start=$(date -Is)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 SLURM_CPUS_PER_TASK
srun .venv/bin/python scripts/16_pistachio_expansion.py --n-estimators 60
echo "done=$(date -Is)"
