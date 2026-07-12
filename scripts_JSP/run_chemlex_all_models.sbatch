#!/bin/bash
# Score ONE PU method on the full Chemlex amide grid (transductive+inductive x 2 feats x 5 seeds).
# Submit once per method; wall time is overridden on the command line for the heavy bagging job.
#   sbatch --job-name=cx_bagging --time=12:00:00 scripts_JSP/run_chemlex_all_models.sbatch bagging
#   sbatch --job-name=cx_elkan                   scripts_JSP/run_chemlex_all_models.sbatch elkan
#   sbatch --job-name=cx_spy                     scripts_JSP/run_chemlex_all_models.sbatch spy
#   sbatch --job-name=cx_nnpu   --time=04:00:00  scripts_JSP/run_chemlex_all_models.sbatch nnpu
#   sbatch --job-name=cx_iwpu                    scripts_JSP/run_chemlex_all_models.sbatch iw_pu

#SBATCH --job-name=cx_pu
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/cx_pu-%x-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/cx_pu-%x-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
MODEL="${1:?usage: sbatch scripts_JSP/run_chemlex_all_models.sbatch <model_label>}"
cd "$WORKTREE"

# Nodes are idle and we run one job per method, so let each HistGB/CatBoost fit use all cores.
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} model=$MODEL start=$(date -Is)"
"$PY" -u -m scripts_JSP.run_chemlex_all_models --model "$MODEL"
echo "done=$(date -Is)"
