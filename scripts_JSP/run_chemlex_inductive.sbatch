#!/bin/bash
# sup_histgb_l2 transfer on the switched datasets (Chemlex amide + HiTEA BH). Courteous, low priority.
# Usage:  sbatch scripts_JSP/run_dataset_switch.sbatch

#SBATCH --job-name=ds_induct
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/ds_induct-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/ds_induct-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
cd "$WORKTREE"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} start=$(date -Is)"
"$PY" -m scripts_JSP.run_chemlex_inductive
echo "done=$(date -Is)"
