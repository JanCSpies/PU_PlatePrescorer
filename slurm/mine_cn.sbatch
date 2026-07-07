#!/bin/bash
# Mine the full ORD (ord.csv + ORDerly parquets) for product-confirmed BH C-N couplings.
# One heavy single-node job; RDKit detection parallelised via forkserver workers.
# Submit from repo root:  sbatch slurm/mine_cn.sbatch
#
# VERY HIGH niceness so colleagues preempt us (see docs/slurm.md). CPU_Express: 24 h cap, idle.
#SBATCH --job-name=bhpu-mine-cn
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"
# Pin BLAS/threads low so the 60 forkserver workers own the parallelism (no oversubscription).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun .venv/bin/python scripts/07_mine_cn_couplings.py \
    --sources ord,orderly \
    --workers 60 \
    --batch-size 2000 \
    --out cn_corpus_ord
echo "done=$(date -Is)"
