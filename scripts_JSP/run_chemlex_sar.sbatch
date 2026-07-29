#!/bin/bash
# Score ONE beyond-SCAR / SAR PU method on the full Chemlex amide grid
# (transductive+inductive x 2 feats x 5 seeds). Submit once per method.
#   sbatch --job-name=cx_propips              scripts_JSP/run_chemlex_sar.sbatch prop_ips
#   sbatch --job-name=cx_sarem  --time=10:00:00 scripts_JSP/run_chemlex_sar.sbatch sar_em
# Smoke (single seed/cell, no file written):
#   sbatch --job-name=cx_smoke  --time=00:30:00 scripts_JSP/run_chemlex_sar.sbatch prop_ips --smoke

#SBATCH --job-name=cx_sar
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/cx_sar-%x-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/cx_sar-%x-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
MODEL="${1:?usage: sbatch scripts_JSP/run_chemlex_sar.sbatch <model_label> [--smoke]}"
shift
cd "$WORKTREE"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} model=$MODEL start=$(date -Is)"
"$PY" -u -m scripts_JSP.run_chemlex_sar --model "$MODEL" "$@"
echo "done=$(date -Is)"
