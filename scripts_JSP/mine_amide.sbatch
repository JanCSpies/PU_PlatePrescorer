#!/bin/bash
# Mine amide-coupling positives from ORD/Pistachio (courteous, low priority).
# Array task 0 = open ORD+ORDerly; task 1 = licensed Pistachio (private, gitignored output).
# Usage:  sbatch scripts_JSP/mine_amide.sbatch

#SBATCH --job-name=mine_amide
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --array=0-1
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/mine_amide-%A_%a.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/mine_amide-%A_%a.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
cd "$WORKTREE"

# RDKit forkserver workers do the parallelism; pin math libs so they don't oversubscribe.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
  SOURCES="ord,orderly"
  OUT="amide_corpus_ord"
else
  SOURCES="pistachio"
  OUT="private/amide_corpus_pistachio"
fi

echo "host=$(hostname) jobid=${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA} sources=$SOURCES out=$OUT start=$(date -Is)"
"$PY" scripts_JSP/mine_amide.py --sources "$SOURCES" --out "$OUT" --workers 30 --batch-size 2000
echo "done=$(date -Is)"
