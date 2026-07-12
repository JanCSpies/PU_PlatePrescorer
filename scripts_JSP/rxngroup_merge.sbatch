#!/bin/bash
# Stage 2-3: merge the per-source parsed parquets, dedup, group, and render the HTML catalogue.
# Run AFTER all rxngroup_parse jobs finish.
#   sbatch scripts_JSP/rxngroup_merge.sbatch

#SBATCH --job-name=rg_merge
#SBATCH --partition=CPU
#SBATCH --nice=1000000
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/rg_merge-%j.out
#SBATCH --error=/home/student/j_spie17/claude_hackathon_JSP2/scripts_JSP/slurm_logs/rg_merge-%j.out

set -euo pipefail
WORKTREE=/home/student/j_spie17/claude_hackathon_JSP2
PY="$WORKTREE/.venv/bin/python"
cd "$WORKTREE"

echo "host=$(hostname) jobid=${SLURM_JOB_ID:-NA} start=$(date -Is)"
"$PY" -u -m scripts_JSP.rxngroup_merge_group --yield-threshold 20 --min-pairs 0 --min-other-reports 25
D=datasets/processed/private/rxn_groups
"$PY" -u -m scripts_JSP.rxngroup_report --top 1500 --in-parquet "$D/reaction_condition_groups_all.parquet"    --out "$D/groups_report_all.html"
"$PY" -u -m scripts_JSP.rxngroup_report --top 1500 --in-parquet "$D/reaction_condition_groups_public.parquet" --out "$D/groups_report_public.html"
echo "done=$(date -Is)"
