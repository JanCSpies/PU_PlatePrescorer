#!/bin/bash
# Second-reaction check for the beyond-SCAR arm: champion vs prop_ips vs sar_em on HiTEA BH.
# Writes hitea_sar_metrics.json (3 models x 2 feats x 5 seeds + bootstrap CI + shadow separability).
#   sbatch --job-name=hitea_sar                  scripts_JSP/run_hitea_sar.sbatch
# Smoke (raw_ecfp + seed0 for each model, no file written):
#   sbatch --job-name=hitea_sar_smoke --time=00:30:00 scripts_JSP/run_hitea_sar.sbatch --smoke

#SBATCH --job-name=hitea_sar
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/hitea_sar-%x-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/hitea_sar-%x-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
cd "$WORKTREE"

export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} start=$(date -Is)"
"$PY" -u -m scripts_JSP.run_hitea_sar "$@"
echo "done=$(date -Is)"
