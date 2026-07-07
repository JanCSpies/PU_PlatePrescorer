#!/bin/bash
# Honest literature->JnJ transfer experiments (spy[PAYN-parity] / naive / CORAL / importance-weight;
# PU vs supervised) under the frozen substrate-disjoint protocol. Three data/feature configs run
# CONCURRENTLY as an array (one per task):
#   task 0: DRFP  lit-only    (reagent-inclusive reaction fingerprint; historical strong feature)
#   task 1: ECFP  lit-only    (substrate fingerprint; common space for the mined corpus)
#   task 2: ECFP  lit + ORD   (data expansion: does the 35.9k mined C-N corpus help transfer?)
# Submit from repo root:  sbatch slurm/transfer.sbatch
#
# VERY HIGH niceness so colleagues preempt us (see docs/slurm.md).
#SBATCH --job-name=bhpu-transfer
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

N_EST=40
CORPUS=datasets/processed/cn_corpus_ord.parquet

case "${SLURM_ARRAY_TASK_ID}" in
  0) srun .venv/bin/python scripts/06_transfer_experiments.py \
        --featurizer drfp2048 --n-estimators ${N_EST} --tag lit_only ;;
  1) srun .venv/bin/python scripts/06_transfer_experiments.py \
        --featurizer ecfp2048 --n-estimators ${N_EST} --tag lit_only ;;
  2) srun .venv/bin/python scripts/06_transfer_experiments.py \
        --featurizer ecfp2048 --n-estimators ${N_EST} --tag lit_plus_ord \
        --extra-corpus ${CORPUS} ;;
esac
echo "done task=${SLURM_ARRAY_TASK_ID} at=$(date -Is)"
