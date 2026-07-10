#!/bin/bash
# HistGB+L2 (pubench sup_histgb_l2) on Florian's data assembly — fills the reconciliation 2x2 cell.
# Usage:  sbatch scripts_JSP/reproduce_histgb_on_fb_data.sbatch

#SBATCH --job-name=histgb_fbdata
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/histgb_fbdata-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/histgb_fbdata-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
cd "$WORKTREE"

# Leave headroom for HistGB internal parallelism; don't oversubscribe cores.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} start=$(date -Is)"
"$PY" scripts_JSP/reproduce_histgb_on_fb_data.py
echo "done=$(date -Is)"
