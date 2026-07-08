#!/bin/bash
# Assemble TEST / P (ORD + Pistachio) / U condition-aware datasets on the cluster.
#
# Pistachio is a 2.9 GB reaction corpus, so mining it is offloaded here as a courteous,
# low-priority job (see docs/slurm.md). Detection is parallelised over CPUs with a
# forkserver pool (RDKit-safe). Outputs land in the worktree's final_datasets/ (shared FS);
# Pistachio-derived + merged corpora go to final_datasets/private/ (gitignored, licensed).
#
# Submit:  sbatch scripts_JSP/assemble_datasets.sbatch
# Watch:   squeue --me   ;   tail -f scripts_JSP/slurm_logs/assemble-<jobid>.out

#SBATCH --job-name=jsp-assemble
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/assemble-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/assemble-%j.out

set -euo pipefail

WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PISTACHIO=/mnt/datasets/pistachio/releases/latest/data/pistachio.smi.deduped.csv
PY="$WORKTREE/.venv/bin/python"

cd "$WORKTREE"

# Pin math-library threads so the many worker processes don't oversubscribe the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} start=$(date -Is)"

"$PY" scripts_JSP/assemble_datasets.py \
  --pistachio "$PISTACHIO" \
  --jobs "${SLURM_CPUS_PER_TASK:-16}" \
  --out-dir "$WORKTREE/final_datasets"

echo "done=$(date -Is)"
