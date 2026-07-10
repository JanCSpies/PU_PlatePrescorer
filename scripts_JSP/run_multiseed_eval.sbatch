#!/bin/bash
# Multi-seed eval (Chemlex transductive + inductive) for the plate-prescoring report. Low priority.
#SBATCH --job-name=ms_eval
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/ms_eval-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/ms_eval-%j.out
set -euo pipefail
cd /home/student/j_spie17/claude_hackathon_JSP2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} start=$(date -Is)"
.venv/bin/python -u -m scripts_JSP.run_multiseed_eval
echo "done=$(date -Is)"
