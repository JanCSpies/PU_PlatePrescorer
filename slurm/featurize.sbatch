#!/bin/bash
# Precompute + cache DRFP/ECFP features for all datasets on a CPU node.
# Submit from the repo root:  sbatch slurm/featurize.sbatch
#
# Uses a VERY HIGH niceness so colleagues with more pressing jobs preempt us in the
# queue (see docs/slurm.md for the cluster + courtesy policy).
#SBATCH --job-name=bhpu-featurize
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Pin per-process math threads to 1 so N worker processes don't oversubscribe cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} mem=${SLURM_MEM_PER_NODE:-?} start=$(date -Is)"
srun .venv/bin/python scripts/03_precompute_features.py \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --bits 2048
echo "done=$(date -Is)"
