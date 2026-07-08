#!/bin/bash
# Mine the LICENSED Pistachio corpus (7.2M atom-mapped reactions) for product-confirmed BH C-N
# couplings. Output is written to datasets/processed/private/ (GITIGNORED) and must never be
# committed or published. Submit from repo root:  sbatch slurm/mine_pistachio.sbatch
#SBATCH --job-name=bhpu-mine-pistachio
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun .venv/bin/python scripts/07_mine_cn_couplings.py \
    --sources pistachio \
    --workers 60 \
    --batch-size 2000 \
    --out private/cn_corpus_pistachio
echo "done=$(date -Is)"
