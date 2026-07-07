#!/bin/bash
# Honest literature->JnJ transfer experiments (naive vs CORAL vs importance-weighting; PU vs
# supervised) under the frozen substrate-disjoint protocol. Three data/feature configs:
#   1. DRFP  lit-only      (reagent-inclusive reaction fingerprint; the historical strong feature)
#   2. ECFP  lit-only      (substrate fingerprint; the common space for the mined corpus)
#   3. ECFP  lit + ORD     (data expansion: does the 35.9k mined C-N corpus help transfer?)
# Submit from repo root:  sbatch slurm/transfer.sbatch
#
# VERY HIGH niceness so colleagues preempt us (see docs/slurm.md).
#SBATCH --job-name=bhpu-transfer
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"

N_EST=100
CORPUS=datasets/processed/cn_corpus_ord.parquet

echo "### [1/3] DRFP literature-only ###"
srun .venv/bin/python scripts/06_transfer_experiments.py \
    --featurizer drfp2048 --n-estimators ${N_EST} --tag lit_only

echo "### [2/3] ECFP literature-only ###"
srun .venv/bin/python scripts/06_transfer_experiments.py \
    --featurizer ecfp2048 --n-estimators ${N_EST} --tag lit_only

echo "### [3/3] ECFP literature + ORD (data expansion) ###"
srun .venv/bin/python scripts/06_transfer_experiments.py \
    --featurizer ecfp2048 --n-estimators ${N_EST} --tag lit_plus_ord \
    --extra-corpus ${CORPUS}

echo "done=$(date -Is)"
