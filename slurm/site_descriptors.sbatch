#!/bin/bash
# Precompute reactive-site descriptors (morfeus sterics + g-xTB electronics) for every unique
# substrate. One single-node job; per-substrate work parallelised via forkserver workers.
# Submit from repo root:  sbatch slurm/site_descriptors.sbatch
#
# VERY HIGH niceness so colleagues preempt us (see docs/slurm.md).
#SBATCH --job-name=bhpu-site-desc
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date -Is)"
# Pin BLAS/threads low so the forkserver workers own the parallelism; each g-xTB subprocess
# runs single-threaded (OMP_NUM_THREADS=1) so N workers do not oversubscribe the node.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# g-xTB parameter files (.gxtb/.eeq/.basisq) live with the shared install.
export GXTBHOME=/mnt/share/public/software/g-xtb/parameters
export GXTB_BINARY=/mnt/share/public/software/g-xtb/binary/gxtb

srun .venv/bin/python scripts/12_precompute_site_descriptors.py \
    --method both \
    --workers "${SLURM_CPUS_PER_TASK}"
echo "done=$(date -Is)"
