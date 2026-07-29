#!/bin/bash
# Re-score the Chemlex amide plate grid with the beyond-SCAR prop_ips model for the PDF report.
# Writes multiseed_metrics_propips.json + eval_arrays_propips.npz (transductive+inductive x 2 feats
# x 5 seeds). Render the report afterwards with: uv run python -m scripts_JSP.make_report --model propips
#   sbatch --job-name=cx_msprop scripts_JSP/run_multiseed_eval_propips.sbatch

#SBATCH --job-name=cx_msprop
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/cx_msprop-%x-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/cx_msprop-%x-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
cd "$WORKTREE"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} start=$(date -Is)"
"$PY" -u -m scripts_JSP.run_multiseed_eval_propips
echo "done=$(date -Is)"
