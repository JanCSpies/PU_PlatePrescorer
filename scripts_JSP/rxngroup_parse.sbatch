#!/bin/bash
# Parse ONE reaction source into <source>_parsed.parquet (class + conditions + dedup key).
# Submit once per source (pistachio is the heavy one -> more cpus + time on the command line):
#   sbatch --job-name=rg_pistachio --cpus-per-task=32 --time=08:00:00 scripts_JSP/rxngroup_parse.sbatch pistachio
#   sbatch --job-name=rg_uspto76                                       scripts_JSP/rxngroup_parse.sbatch uspto_76_16
#   sbatch --job-name=rg_uspto01                                       scripts_JSP/rxngroup_parse.sbatch uspto_01_16
#   sbatch --job-name=rg_orderly --time=02:00:00                       scripts_JSP/rxngroup_parse.sbatch orderly

#SBATCH --job-name=rg_parse
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/rg_parse-%x-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/rg_parse-%x-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
SOURCE="${1:?usage: sbatch scripts_JSP/rxngroup_parse.sbatch <pistachio|uspto_76_16|uspto_01_16|orderly>}"
cd "$WORKTREE"

# RDKit is single-threaded per molecule; parallelism comes from the worker Pool, so cap BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} source=$SOURCE start=$(date -Is)"
"$PY" -u -m scripts_JSP.rxngroup_parse --source "$SOURCE" --workers "${SLURM_CPUS_PER_TASK:-16}"
echo "done=$(date -Is)"
