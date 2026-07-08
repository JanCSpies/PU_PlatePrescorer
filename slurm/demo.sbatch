#!/bin/bash
# Final "better PAYN" demo with the VAL-selected winning config: importance-weighted PU-bagging on
# ECFP substrate features, trained on literature+ORD positives vs the transductive JnJ grid. CORAL
# is OFF (it hurt on high-dim fingerprints). Produces the single final JnJ TEST metrics + a designed
# demo plate with hit-rate uplift.  Submit:  sbatch slurm/demo.sbatch
#SBATCH --job-name=bhpu-demo
#SBATCH --partition=CPU_Express
#SBATCH --nice=1000000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "host=$(hostname) start=$(date -Is)"
srun .venv/bin/python scripts/08_plate_design_demo.py \
    --no-coral --n-estimators 80 --plate-size 96
echo "done=$(date -Is)"
