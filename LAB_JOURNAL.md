# Lab Journal

> **Append-only, reverse-chronological.** Newest entry at the **top**. One entry per work session
> or experiment. **Never edit or delete another author's entry.** Tag every entry with author +
> date + (agent/human). This journal is read by both humans *and* both Claude Code sessions — write
> so either audience can follow. For ML runs, record: **hypothesis · setup (data/seed/params) ·
> result (numbers) · interpretation · next step.**
>
> Entry template:
> ```
> ## YYYY-MM-DD — <short title>   —  <author> (<claude|human>)
> **Context/Hypothesis:** …
> **What I did:** …
> **Result:** …
> **Interpretation / decision:** …
> **Next:** …
> ```

---

## 2026-07-07 (later) — Deep-dive complete, split verified, data landscape mapped — Florian's session (claude)

**What I did:**
- Ran `scripts/01_split_raw_dataset.py`: **test 11,328 / literature 11,857 / dropped(Ahneman) 4,312**,
  27,497 conserved. Outputs in `datasets/processed/` (csv+parquet) + `split_report.json`. Code passes
  `ruff` + `mypy --strict`.
- **Verified feasibility definition:** `labels = (Yield ≥ 10 %)` — 99.9 % agreement dataset-wide;
  JnJ label-0 ≤ 9.9 %, label-1 ≥ 10.0 %. Recorded in `docs/datasets.md`.
- All **5 literature agents finished** → 22 notes in `docs/literature/notes/` + merged
  `docs/literature/SYNTHESIS.md` + `docs/decisions/002-pu-framing-and-benchmark.md`.
- **Data landscape discovered** (colleague reorganised `datasets/raw/` + added mounts): local
  `Cernak/50K_dataset_v2.csv` (50,687 wells, ~71 % zero-yield = negatives); external read-only mounts
  `/mnt/datasets/ORDerly` (1.9 GB), `/mnt/datasets/DynaMAIK/ord.csv` (2 GB),
  `/mnt/datasets/pistachio/releases/latest` (153 GB), `/mnt/datasets/blockspace/v5.0/...` (37 GB TACIT
  corpus). Catalogued in `docs/datasets.md`.

**Key findings (see SYNTHESIS.md):**
- Our data is **SAR/SNAR, not SCAR** → benchmark PU arms: SCAR-spy (PAYN) → PULSNAR → SAR-EM → PUe.
- Reliable-negative set = **union(TACIT implicit negatives, PAYN spies)**.
- Generative PU = stretch goal only.
- Test = Neves ChemRxiv 2025 (`10.26434/chemrxiv-2025-xcr46`, `github.com/schwallergroup/bh-hte-ood`);
  reuse their DRFP K-means OOD folds. Bar to beat: RF/DRFP >90 % precision@conf>0.85 OOD.
- Bundled literature sources are **substrate-disjoint from JnJ** (0 shared amines, ≤2 aryls).
- Demo = **hit-rate uplift** of a model-designed vs random JnJ plate (zero new experiments); oracle
  ceiling ≈ 2.56×, realistic ≈ 1.7–2.0×.

**Next (open):** (1) DRFP feature module (recompute from SMILES). (2) Leakage-audit module
(canonical SMILES → InChIKey → substrate/reaction/cluster checks). (3) Combinatorial U-space
generator per `bh-JnJ-testset-spec.md`. (4) PU baselines: reproduce PAYN spy-CatBoost, then PULSNAR.
(5) Evaluation harness on Neves folds. (6) hit-rate-uplift demo + figures (`dataviz` skill).

---

## 2026-07-07 — Project bootstrap: scaffolding, dataset split, literature deep-dive — Florian's session (claude)

**Context:** Kickoff of the hackathon build. Sharpened the project from the earlier broad
"TACIT+PAYN" idea to a concrete, testable target: **predict BH C–N coupling feasibility from
literature data via PU learning, evaluated on the JnJ HTE 2024 (Neves) set.**

**What I did:**
- Read `rules.md` (judging: Impact 25 / Claude use 25 / Depth 20 / Demo 30; submission Mon Jul 13
  9 PM ET; must be built during event & open-sourced).
- Inspected `datasets/raw/BH_HTE_Curated_Dataset_v20260219.csv` (27,497 rows, 8 sources). Confirmed
  by structural fingerprint that **`Source == "Dreher Doyle 2018"` is Ahneman et al. *Science* 2018**
  (15 aryl × 1 amine × 4 ligand × 3 base × 23 additives; the only additive-bearing source) and that
  the curated file has **no additive-identity column** → decided to **drop it** (see
  `docs/decisions/001-drop-ahneman.md`).
- Characterised the **JnJ HTE 2024** test space: 110 aryl × 68 amine (only 888/7480 pairs measured),
  32 catalysts / 8 bases / 6 solvents, 4 dominant P2Et "general-condition" ligand sets; labels 0/1 =
  6896/4432. This defines the combinatorial unlabeled space we must generate (minus test).
- Noted a data caveat: **`DRFP` and `QM` columns in the raw CSV are truncated NumPy reprs** and are
  **not usable**; features must be recomputed from SMILES (`drfp`, RDKit). Recorded in `docs/datasets.md`.
- Built project scaffolding: `pyproject.toml` (uv, Python 3.13, ruff Google-docstring + mypy strict),
  `src/bhpu/` package, `CLAUDE.md` (shared charter + collaboration protocol + coding standards),
  this journal, `docs/`.
- Wrote `src/bhpu/data/splitting.py` (`RawDatasetSplitter`, typed/OOP/validated) +
  `scripts/01_split_raw_dataset.py`.
- Launched a **5-territory literature deep-dive** (subagents): (A) PU theory beyond SCAR, (B)
  generative/deep PU, (C) BH C–N prediction + OOD + the JnJ test set, (D) reaction data sources &
  bias, (E) chemistry→biology / plate design / impact. Notes land in `docs/literature/notes/`.

**Result (so far):**
- Env built via `uv sync` (Python 3.13). Split script written; run pending env confirmation.
- Literature agent (B) returned: **generative negative-synthesis = stretch goal**; adopt **prior-free
  PU losses** (VPU/CoVPU, balanced-PU) and the "positive distribution pollution" framing (few +
  publication-biased + unknown prior are one problem). Other agents in progress.

**Interpretation / decision:** Data framing is settled (test=JnJ, drop Ahneman, literature=rest).
Method direction leaning to prior-free PU + reliable-negative recovery over generative approaches.

**Next:** confirm env → run split → generate combinatorial unlabeled space (needs JnJ spec from
agent C) → leakage-audit module → PU baselines (nnPU / spy-CatBoost / VPU) → hit-rate-uplift demo.
