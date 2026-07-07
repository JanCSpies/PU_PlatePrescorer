#!/bin/bash
# Benchmark sweep (feature x PU-strategy x seed) as a SLURM array job.
# One array task = one config; results merge via `--aggregate` afterwards.
# Submit from the repo root (needs cached features from featurize.sbatch first):
#   sbatch slurm/benchmark.sbatch
# Array size 0-19 must match `uv run python scripts/04_benchmark.py --list`.
#
# VERY HIGH niceness so colleagues preempt us (see docs/slurm.md).
#SBATCH --job-name=bhpu-benchmark
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --array=0-19%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"
srun .venv/bin/python scripts/04_benchmark.py --config-id "${SLURM_ARRAY_TASK_ID}"
echo "done=$(date -Is)"
