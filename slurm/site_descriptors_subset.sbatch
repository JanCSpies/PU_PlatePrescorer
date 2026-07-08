#!/bin/bash
# Subset-first site-descriptor precompute (JnJ+literature+grid only; excludes the big ORD corpus).
# Morfeus sterics + g-xTB electronics, 20 conformers, all 3 aggregation modes.
#SBATCH --job-name=bhpu-site-subset
#SBATCH --partition=CPU
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GXTBHOME=/mnt/share/public/software/g-xtb/parameters
export GXTB_BINARY=/mnt/share/public/software/g-xtb/binary/gxtb
srun .venv/bin/python scripts/12_precompute_site_descriptors.py \
    --method both --datasets lit,jnj,grid --n-conformers 20 \
    --workers "${SLURM_CPUS_PER_TASK}"
echo "done=$(date -Is)"
