#!/bin/bash
# Run a pubench benchmark config on the cluster (courteous, low priority).
# Usage:  sbatch scripts_JSP/pubench_run.sbatch configs/pu_bench.e3.yaml

#SBATCH --job-name=pubench
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/pubench-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/pubench-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
CONFIG="${1:?usage: sbatch pubench_run.sbatch <config.yaml>}"
cd "$WORKTREE"

# Cap math-library threads; leave a few cores for CatBoost/HistGB internal parallelism.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} config=$CONFIG start=$(date -Is)"
"$PY" -m pubench.cli --config "$CONFIG"
echo "done=$(date -Is)"
