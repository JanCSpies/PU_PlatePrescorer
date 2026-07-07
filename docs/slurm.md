# SLURM cluster — resources, policy, and how we use it

The group SLURM cluster (Slurm 23.02) is available to `fboser`. Compute is affordable for us,
so we offload heavy work (featurisation, sweeps, tuning) here — but always as **courteous,
low-priority jobs** so colleagues with pressing work preempt us in the queue.

## Courtesy policy (IMPORTANT)
Every job we submit uses a **very high niceness**: `#SBATCH --nice=1000000`. This drops our
scheduling priority to ~1 so anyone else's job jumps ahead; we still run immediately whenever
there is idle capacity. Do not remove it. Keep memory/CPU requests to what a job needs.

## Partitions (from `sinfo`)
| Partition | Time limit | Nodes | Notes |
|---|---|---|---|
| `CPU` (default) | 14 days | `worker.additive_2..9` | 2 idle nodes (`_8`,`_9`) with **384 threads / 1 TB RAM** each |
| `CPU_Long` | infinite | `worker.additive_2` | long-running CPU jobs |
| `CPU_Express` | **24 h** | `worker.additive_1` | 192 threads / 500 GB; fast turnaround, short jobs |
| `GPU` | infinite | `worker.product_1..2` | `_1`: 2× RTX **A6000** (48 GB); `_2`: 8× **A40** (48 GB) |

Nodes are dual-socket; `worker.additive_8/9` are 2×96×2 = 384 hardware threads, 1 TB RAM.

## Environment on compute nodes
`/home/employee/fboser` is shared, so the project's `uv` virtualenv runs directly:
`.venv/bin/python` (Python 3.13) with rdkit/catboost/etc. imports fine on the nodes (verified).
No `module load` needed. Notes:
- Multiprocessing: use a **`forkserver`** context (not the default fork) — forking after
  numpy/rdkit spawn threads can deadlock. See `scripts/03_precompute_features.py`.
- Pin math threads (`OMP_NUM_THREADS=1`, …) when running many worker processes so they don't
  oversubscribe cores (set in the sbatch files).

## Jobs in this repo
| Script | What | Submit |
|---|---|---|
| `slurm/featurize.sbatch` | Parallel DRFP+ECFP feature cache (CPU, 64 cores) | `sbatch slurm/featurize.sbatch` |
| `slurm/benchmark.sbatch` | Array job: data/feature × PU-strategy sweep (depends on features) | `sbatch --dependency=afterok:<featurize_jobid> slurm/benchmark.sbatch` |

Outputs land in `datasets/processed/features/` (cached features) and `artifacts/` (results).
Job logs: `slurm/logs/<name>-<jobid>.out` (git-ignored).

## Handy commands
```bash
squeue -u fboser                          # my jobs
sinfo -o "%P %.6a %.14l %.6t %N"          # partitions / node states
scontrol show job <jobid>                 # job detail
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqCPUS   # after completion
scancel <jobid>                           # cancel
```
