# Autonomous run plan (8 h) — READ FIRST after any compaction

This is the durable roadmap for an ~8-hour autonomous session. If context was just compacted,
read this + `CLAUDE.md` + the top of `LAB_JOURNAL.md`, then continue the workstreams below. Keep
this file's **Status checklist** updated as you go.

## Objective (frozen — decided by Florian)
Predict **feasibility of Buchwald–Hartwig C–N couplings on the JnJ (Neves) test set**, training on
**literature / external data only via PU learning** — **never use JnJ labels for training**
("purest thesis"). Feasibility = assay **yield ≥ 10 %** (verified; fixed).

## HARD RULES (do not violate)
1. **No cheating.** Do not modify the evaluation to look better. No metric hacking. No data leakage.
   The eval protocol is frozen in `src/bhpu/eval/protocol.py` — do **not** change it to raise a score.
2. **Substrate-disjoint holdout.** JnJ is split into VAL and TEST by *whole substrate groups*
   (val/test substrates are disjoint). Literature training must contain **no JnJ substrate**
   (aryl or amine) and **no JnJ (aryl,amine) pair**. TEST is touched **once**, at the very end;
   model selection uses VAL only.
3. **Don't modify `/mnt` files.** They are shared group data. To clean/alter, **copy into the repo**
   (`datasets/raw/<source>/` or `datasets/external/`) and edit the copy.
4. **Never `git push`.** Local commits are fine (and encouraged for checkpoints); only Florian pushes.
5. Use **SLURM** for heavy compute, always `--nice=1000000` (see `docs/slurm.md`). Don't touch the
   pre-existing `evo2` jobs.

## Frozen eval protocol (the anti-cheat core) — `src/bhpu/eval/protocol.py`
- Training data = external only (literature HTE minus Ahneman, Cernak, ORD, …) as **P** (label-1) +
  the unlabeled combinatorial grid as **U**. No JnJ labels ever enter training.
- JnJ split: deterministic substrate partition → VAL substrates vs TEST substrates (disjoint).
  VAL/TEST reactions are those whose aryl AND amine fall in the respective substrate set.
- Report on TEST: ROC-AUC, **AUPRC (primary, imbalanced)**, balanced acc, precision@high-conf, ECE,
  plus plate hit-rate uplift. Select models on VAL only.
- Invariants asserted in code: no JnJ pair in train; test substrates absent from val+train;
  threshold fixed at 10 %; test used once.

## Workstreams (ordered by Florian's priority)
- **WS0 (task #9)** — freeze the eval protocol (do this first; everything else builds on it).
- **WS1 (task #10)** — **C–N data expansion + cleaning** (primary focus). Mine C–N couplings from:
  - `/mnt/datasets/ORDerly/*.parquet` (cols: reactant_000/001, product_000, agent_*, solvent_*,
    yield_000, rxn_str) — open (CC-BY-SA). Extract aryl-halide + amine → aryl-amine BH reactions.
  - `/mnt/datasets/DynaMAIK/ord.csv` (reaction_id, reactants, products, reaction_smiles).
  - `datasets/raw/Cernak/50K_dataset_v2.csv` (Elec_SMILES=aryl, Nuc_SMILES=amine, Assay Yield;
    ~71 % zero-yield = negatives; only ~4 substrates → mostly condition info / negatives).
  - `datasets/processed/literature_hte.*` (already have; 6 HTE sources).
  - Pistachio (`/mnt/.../pistachio`) — LICENSE-flagged (validation/enrichment only; do NOT put in a
    publishable training corpus without confirming ToS). Prefer ORD as the open primary.
  Canonicalize (RDKit), dedupe (InChIKey + reaction hash), leakage-audit vs JnJ, keep provenance.
  Build a unified **positive corpus**; benchmark which sources actually help on VAL.
- **WS2 (task #11)** — **domain adaptation / propensity (SAR)**. Core problem: naive literature→JnJ
  ≈ random (0.52). Try SAR-EM propensity e(x), importance weighting, domain-invariant features,
  and PU-with-combinatorial-grid-U. Evaluate on the frozen protocol.
- **WS3 (task #12)** — **PU theory + PAYN baseline**. Deep-dive the problems with bagging (implicit
  SCAR, uncalibrated) and PAYN (SCAR); find better estimators (SAR/SNAR, nnPU, class-prior).
  **Clone `github.com/GloriusGroup/PAYN`** (read-only; never push) and run as a baseline.
- **WS4 (task #13)** — **better features**. `xtb` semiempirical descriptors (set up on SLURM;
  charges/HOMO-LUMO/Fukui/dipole for unique substrates) and **OMol-UMA** via the colleague's
  `gpuma/` package (Fairchem UMA/OMol25 on the GPU partition: relaxed geometries + energies/
  embeddings). Research whether QM/DFT descriptors beat DRFP/ECFP under the frozen protocol. Timebox
  env setup (torch/fairchem/model weights can be a rabbit hole).

## Key facts & paths
- Env: `uv` + Python 3.13, `.venv/bin/python` (works on compute nodes). Gates:
  `uv run ruff check . && uv run mypy src scripts && uv run pytest`.
- Data: `datasets/raw/Neves/BH_HTE_Curated_Dataset_v20260219.csv` → split in `datasets/processed/`.
  Cached features (DRFP2048/ECFP2048) in `datasets/processed/features/`.
- Coding standards (CLAUDE.md §3): uv, OOP, full typing, Google docstrings, ruff+mypy strict.
- `gpuma/` = colleague's MLIP toolkit (Fairchem UMA/OMol25, ORB-v3; GPU; morfeus conformers).
- Baselines so far (within-JnJ recovery, NOT the target): bagging PU ROC-AUC ~0.90 > spy ~0.86;
  supervised OOD bar (40 folds, DRFP2048) 0.837. **Real literature→JnJ transfer ≈ 0.52 (the number
  to beat honestly).**

## Status checklist (update as you go)
- [x] WS0 protocol frozen (`FrozenEvalProtocol`; val 2624 / test 1920, both-axis substrate-disjoint;
      leakage guard tested). TODO: re-run baselines under it for honest starting numbers.
- [ ] WS1 ORD C–N mining; Cernak adapter; unified positive corpus + leakage audit; source benchmark
      — CONFIRMED minable: ORDerly ~10.6% C–N (~153k), ord.csv ~22.9%; detector = aryl-halide + N–H
      amine → product. literature_hte (4,464 pos) is the clean in-domain seed; ORD adds substrate diversity.
- [ ] WS2 SAR/propensity + grid-U PU; transfer improved over 0.52
- [ ] WS3 PU-theory notes; PAYN cloned + run as baseline
- [ ] WS4 xtb features on SLURM; UMA via gpuma (stretch); QM-vs-fingerprint verdict
- [ ] Final: best honest literature→JnJ TEST metrics + plate-uplift demo; LAB_JOURNAL written up
