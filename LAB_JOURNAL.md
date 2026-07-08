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

## 2026-07-08 — Site-specific descriptors (morfeus + g-xTB + conformers): a clean NEGATIVE — Florian's session (claude)

**Context/Hypothesis:** Whole-molecule GFN2-xTB didn't help (earlier entry). The hypothesis was
that the failure was *resolution*, not QM: proper **site-specific** descriptors at the reacting
atoms — sterics at the aryl C–X carbon and amine N, plus better electronics — might beat ECFP and,
crucially, help the **inductive** (novel-substrate) case. Also test the sub-hypothesis that physical
descriptors are more **domain-invariant** than fingerprints (lower domain-shortcut AUC).

**What I did (all this session, gated green):**
- `chem/sites.py`: BH-template reactive-site locator (aryl C–X carbon, amine N); multi-conformer
  ETKDG+MMFF embedding; boltzmann/lowest/mean aggregation.
- `features/morfeus_desc.py`: Sterimol L/B1/B5, %V_bur, SASA at the C–X carbon (along the N-approach
  axis); %V_bur, pyramidalisation, SASA at the amine N. `features/gxtb.py`: g-xTB (Grimme 2025,
  ≈ωB97M-V) gap/IP/EA/dipole + **per-atom EEQ_BC charge & Wiberg BO at the reacting atom**.
- **Reaction-center validation** (`scripts/13`): template site is 100% correct on single-site
  substrates (56% of ORD) but only **36.7%** on ambiguous multi-site (poly-halide/poly-N–H).
  → **Product-guided resolution** (`scripts/14`, `coupling_site_indices_for_product`): resolve the
  *observed* reacting atom from reactions with known products (57,020 reactions → 14,377 aryl +
  10,152 amine site overrides).
- **Subset-first** eval (`scripts/12`,`15`; JnJ+lit+grid, 1,476 substrates, 20 conformers, 3 modes):
  IW-PU on the frozen protocol, transductive + inductive, 26 arms.

**Result (subset, TEST ROC-AUC):**
| feature | transductive (best mode) | inductive (best mode) |
|---|---|---|
| **ECFP (baseline)** | **0.709** | **0.448** |
| morfeus (sterics) | 0.68 (mean) | 0.471 (mean) |
| g-xTB (electronics) | 0.60 (mean) | 0.38 |
| site (morfeus+g-xTB) | 0.63 (mean) | 0.42 |
| ecfp + site | 0.70 (lowest) | 0.38 |

- **Domain-shortcut AUC = 1.000 for ALL 26 arms** — physical descriptors do **not** reduce
  literature-vs-JnJ separability (sub-hypothesis refuted).
- **Conformer aggregation: mean ≥ boltzmann > lowest** (morfeus mean 0.68 vs boltzmann 0.59 vs
  lowest 0.58; same ordering for g-xTB and the combo). Lowest-energy is the worst.
- **No descriptor breaks the ~0.47 inductive ceiling**; ecfp+site *hurts* inductively (0.35–0.38).

**Interpretation / decision:** A **robust negative**. Even with correct reacting atoms
(product-resolved), site sterics + g-xTB electronics, 20-conformer ensembles, and 3 aggregation
strategies, **connectivity fingerprints (ECFP) remain the best feature** for this transfer, and
site descriptors do **not** solve novel-substrate generalisation. The domain shortcut is not a
representation artifact — it persists across every feature type. Per Florian's "don't compute every
molecule if the effect is negative" rule: **descriptors are NOT scaled to the full corpus.** The
open inductive problem needs a different lever (e.g. substrate-aware DA, or reaction-condition
context), not richer per-substrate physics.

**Next:** Pistachio **data expansion** (orthogonal to descriptors): mined **201,340** unique
JnJ-leak-free C–N pairs (0 exact test pairs, 54,937 with Pd — 20× ORD), gitignored per licence.
Running lit vs lit+ORD vs lit+ORD+Pistachio on ECFP IW-PU (`scripts/16`) — result pending.

## 2026-07-08 — RESULTS: better-PAYN transfer, xTB verdict, transductive limit — Florian's session (claude)

**Full transfer sweep** (`scripts/06`+`09`, frozen protocol, VAL-selected → TEST reported once).
Domain-shortcut discriminator AUC = **1.000** everywhere (literature vs JnJ trivially separable —
the diagnosis, confirmed). Headline TEST ROC-AUC:

| feature/data | spy (PAYN/SCAR) | naive PU | **IW-PU (better-PAYN)** |
|---|---|---|---|
| ECFP, lit-only | 0.554 | 0.561 | **0.710** |
| ECFP, lit+ORD | 0.607 | 0.604 | **0.720** |

- **Covariate-shift importance weighting is the win** (0.55→0.72), beating the PAYN-parity spy
  baseline by ~0.15 AUC. **DRFP is catastrophic** (PU arms ~0.26 — its condition bits *are* the
  domain shortcut → use substrate ECFP). **CORAL hurts** on 4096-d sparse fingerprints (honest
  negative; disabled by default). Data expansion (ORD) lifts VAL 0.669→0.773.
- **Final model + plate demo** (`scripts/08`): TEST ROC-AUC 0.722 / AUPRC 0.723; a model-ranked
  96-well plate is **100 % feasible vs 43 % random = 2.32× uplift**. Scores uncalibrated (ECE 0.42) —
  ranking strong, probabilities need calibration.

**Two decisive follow-ups** (`scripts/11`, `10`):
- **WS4 — xTB doesn't help.** GFN2-xTB electronics for all 24,525 substrates (45 s, `tblite`).
  ECFP 0.720 > ECFP⊕xTB 0.669 > xTB-only 0.614. Global QM descriptors dilute the fingerprint;
  connectivity features suffice. (Site-specific QM / UMA embeddings untested.)
- **⚠ Transductive limit (the key honesty finding).** Removing TEST substrates from U (inductive)
  collapses TEST **0.720 → 0.403** (below random) while VAL holds at 0.752. So the gain is
  **transductive**: the model ranks feasibility over a *known* substrate library (= the plate-design
  goal, legitimate) but does **not** generalize to novel substrates. Reported honestly in RESULTS.md;
  not oversold.

**Infra fix:** CatBoost was grabbing all 192 node cores per SLURM task → ~10–100× oversubscription
(array tasks took 62–101 min!). Capped `thread_count` to `SLURM_CPUS_PER_TASK` → the QM+inductive
sweep ran in ~4 min. See `models/boosting.py`.

**Bottom line:** literature-only, no JnJ labels — a domain-debiased PU model pre-scores a known
BH substrate matrix at ROC-AUC 0.72 (2.3× plate uplift), beating PAYN; novel-substrate
generalization remains open. Full writeup in `RESULTS.md`.

## 2026-07-08 — WS1 mining done + WS2/WS3 domain-adaptation core + "better PAYN" — Florian's session (claude)

**Context:** the research sub-agent's diagnosis reframed the whole problem (note
`docs/literature/notes/theory-crossdomain-pu.md`, ADR `docs/decisions/003-...`): the ≈0.52
literature→JnJ ROC-AUC is **not** class-prior shift (ROC-AUC is invariant to that) — it is a
**domain shortcut** (a P-vs-U classifier learns "was this reaction *reported*?" not "does it
work?"). Fix = transductive U (the JnJ grid we already have) + kill the shortcut (CORAL feature
alignment, covariate-shift importance weights, prior calibration). This *is* the "better PAYN"
Florian asked for (PAYN = spy-PU under SCAR, predicts yield, single-dataset CV — confirmed cloned &
running in `external/PAYN`, note `payn-baseline.md`; its in-domain SCAR framing is the baseline to beat).

**What I did:**
- **WS1 — C–N miner** (`src/bhpu/data/mining.py`, `scripts/07`, SLURM `slurm/mine_cn.sbatch`).
  High-precision detector: a reaction counts only if some (aryl-halide, N–H amine) pair's enumerated
  BH product matches one of the reaction's *recorded* products. Mined **ord.csv + ORDerly** on
  CPU_Express (60 workers): **scanned 3.80M reactions in 132 s (~29k/s)** → **124,518 confirmed C–N
  couplings** → **35,929 unique substrate pairs** after cross-source dedup. Fixed an atom-map leak in
  stored SMILES (some ORD reactants are mapped) → re-mined clean (0 mapped rows).
- **LEAKAGE / the user's explicit worry ("is the test set inside ORD/Pistachio?"):**
  **`n_exact_jnj_pair_hits = 0`** — ORD does **not** contain the JnJ (aryl,amine) test pairs.
  **2,221** mined pairs reused a JnJ *substrate* (common building blocks) → **dropped** to keep the
  frozen protocol's substrate-disjointness. Provenance: `datasets/processed/cn_corpus_ord_provenance.json`.
  (Pistachio deferred — license-flagged; ORD is the open primary. Only 2,724/35,929 have detectable
  Pd → most ORD C–N is SNAr/Cu; `has_pd` flag kept for a BH-only ablation.)
- **WS2/WS3 — domain-adaptation package** (`src/bhpu/adapt/`): `DomainDiscriminator`
  (cross-fitted source-vs-target AUC = the shortcut diagnostic + covariate-shift weights
  `w=d/(1−d)`, Kish ESS guard), `CoralAligner` (closed-form whiten→recolour). Productised
  **`DomainAdaptedPULearner`** ("better PAYN": transductive U + CORAL + importance-weighted PU-bagging).
- **Honest transfer harness** (`src/bhpu/eval/transfer.py`, `scripts/06`): 7 arms
  (naive/CORAL/IW × PU/supervised) under the frozen protocol; extended to fold in the mined ORD
  corpus (ECFP substrate space) for a data-expansion benchmark. Running on SLURM (job 55114):
  DRFP lit-only, ECFP lit-only, ECFP lit+ORD.
- **Bug the unit tests caught:** CORAL fit on positives *alone* inflates their covariance to the
  grid's bimodal spread and **destroys** the feasibility signal (synthetic AUC 0.46). Fix: fit the
  domain map on the *source marginal* (all literature, spans both classes), apply to positives.
  New test `test_domain_adaptation_beats_naive_under_shift` now demonstrates the full thesis
  (naive sinks under a domain shift; adaptation recovers). 20 tests green; ruff+mypy clean.

**Result (data so far):** positive corpus expanded ~9× (4.1k literature → ~40k with ORD), huge
substrate diversity, provably JnJ-leak-free. Transfer numbers pending (SLURM 55114).

**Next:** read 55114 results → pick the winning arm on VAL → single TEST report + plate-uplift demo;
then WS4 (xtb/UMA features) if the fingerprint verdict warrants it. **TEST touched once, at the end.**

## 2026-07-08 — Autonomous run START: frozen protocol + WS1 data recon — Florian's session (claude)

**Decisions locked (Florian, before an 8h autonomous run):** objective = predict JnJ BH C–N
feasibility **from literature/PU only, never using JnJ labels** (purest thesis). Held-out =
**substrate-disjoint**. PAYN = **clone & run the real repo** as baseline. Focus order: **C–N data
expansion+cleaning → domain adaptation/propensity (SAR) → PU-theory problems (bagging/PAYN/SCAR) →
better features (xtb / OMol-UMA via gpuma / DFT)**. Rules: **no cheating/leakage, don't modify /mnt
(copy in), never git push.** Master plan in `AUTONOMOUS_RUN.md`.

**WS0 done — frozen anti-cheat eval** (`src/bhpu/eval/protocol.py`, `FrozenEvalProtocol`): JnJ split
into VAL (2624) / TEST (1920) by whole substrate groups, **val/test share 0 aryls & 0 amines**;
`assert_training_clean()` rejects any training row reusing a JnJ substrate/pair (verified: raw
literature = 1384 leaks → filtered to 0). Base rates val 0.32 / test 0.44. Not to be changed to
inflate scores.

**WS1 recon — C–N mining is rich (structure detector = aryl (pseudo)halide + N–H amine → product):**
- ORDerly (`/mnt/datasets/ORDerly`, 1.44M condition rows): **~10.6% C–N ≈ 153k couplings**, 52% have
  yields. Open (CC-BY-SA).
- ord.csv (`/mnt/datasets/DynaMAIK`, 2GB): **~22.9% C–N** (sample shows XPhos/piperazine BH reactions).
- Cernak 50k: 50k wells but ~4 substrates → negatives/conditions, not substrate diversity.
- literature_hte (have): 4,464 positives, already substrate-disjoint from JnJ — cleanest in-domain seed.
→ Plan: SLURM-parallel miner (structure SMARTS) → canonicalize → dedupe (InChIKey/reaction-hash) →
leakage-audit vs JnJ (drop any JnJ substrate) → unified positive corpus. ORD's **substrate diversity**
is the key lever for cross-domain transfer.

**Running:** background research subagent on cross-domain PU (why 0.52, SAR/propensity/importance
weighting/class-prior-shift, ranked fixes) → will write `docs/literature/notes/theory-crossdomain-pu.md`.

**Next:** build ORD C–N miner (WS1) → re-run baselines under frozen protocol (honest 0.52 start) →
WS2 domain adaptation → WS3 PAYN clone + PU theory → WS4 xtb/UMA features.

---

## 2026-07-07 (night) — SLURM online + first benchmark: bagging PU wins decisively — Florian's session (claude)

**SLURM setup (see `docs/slurm.md`):** cluster mapped (CPU 14d / CPU_Express 24h / CPU_Long / GPU
2×A6000+8×A40; idle 384-thread/1TB nodes). All our jobs use **`--nice=1000000`** (priority ~1) so
colleagues preempt us. Verified the uv `.venv` runs on compute nodes; use `forkserver` for MP.

**Job 1 — parallel feature cache** (`scripts/03`, `slurm/featurize.sbatch`, 64 cores): cached DRFP+ECFP
(2048) for JnJ / clean-literature / unlabeled-grid → `datasets/processed/features/`. **DRFP JnJ 303s→7.2s
(42× speedup)**; whole job 41s.

**Job 2 — benchmark array** (`scripts/04`, `slurm/benchmark.sbatch`, 20 configs, 8 cores each, all
COMPLETED). Within-JnJ PU = reveal ½ true positives as P, recover the rest from U. Results
(`artifacts/benchmark_summary.json`):

| task | feat | strategy | ROC-AUC | AUPRC | uplift@96 |
|---|---|---|---|---|---|
| supervised OOD (40 folds) | drfp2048 | – | **0.837 ± 0.105** | 0.744 | – |
| supervised OOD (40 folds) | ecfp2048 | – | 0.846 | 0.741 | – |
| PU | drfp2048 | spy | 0.855–0.863 | 0.62–0.67 | ~3.0× |
| PU | drfp2048 | **bagging** | **0.897–0.900** | **0.82–0.83** | **~4.1×** |
| PU | ecfp2048 | spy | 0.80–0.835 | 0.46–0.58 (high var) | 1.8–4.1× |
| PU | ecfp2048 | bagging | 0.889–0.895 | 0.81–0.82 | ~4.0× |

**Findings:**
1. **Bagging PU ≫ spy** on every metric, and dramatically **more stable across seeds** (spy variance is
   large, worst on ECFP). Confirms the "many weak predictors" thesis (phenol paper / Mordelet–Vert) — make
   bagging the default PU arm.
2. **Subsample ratio k=1 ≈ k=2** (0.898 vs 0.900) → use k=1 (cheaper).
3. **DRFP ≥ ECFP** for PU (esp. spy); features roughly tied for the supervised bar.
4. Supervised OOD DRFP2048 across 40 folds = **0.837 ROC-AUC** — a credible Neves-level in-domain bar
   (vs the misleading 0.699 single-fold/1024-bit smoke number).

**Next:** (a) the real cross-domain problem (literature→JnJ, still the open challenge — Exp C was ~random);
try PU with **combinatorial-grid U** + propensity/importance weighting. (b) Optuna-tune CatBoost + bagging
(GPU partition for TabPFN/ChemProp arms). (c) precision@high-conf calibration (guard the empty-slice
warning in metrics). (d) apply the winning bagging model to score the unlabeled grid → plate design demo.

---

## 2026-07-07 (evening) — First end-to-end run: PU works within-domain, naive transfer fails — Florian's session (claude)

**Setup:** `scripts/02_run_pu_baseline.py`. DRFP (1024 bits, reagent-inclusive), CatBoost (250 iters,
depth 6, auto_class_weights=Balanced). Leakage: substrate-level filter removed 1,384 literature rows
(19 shared aryls, 1 amine via canonical InChIKey) → 10,473 clean. Artifacts:
`artifacts/pu_baseline_metrics.json`.

**Results:**
| Exp | Setup | Headline |
|---|---|---|
| A | Supervised CatBoost on 1 Neves OOD fold of JnJ | ROC-AUC 0.699, AUPRC 0.357, BalAcc 0.531, prec@0.85 = n/a (0 preds > 0.85 → underconfident) |
| B | **PU within JnJ** (reveal ½ positives, recover rest) | **Spy 0.874 / AUPRC 0.719; Bagging 0.899 / AUPRC 0.826** (base 0.243) |
| C | **Literature → JnJ transfer** (supervised) | ROC-AUC **0.524** (≈random); plate uplift ~0.93–1.01× (no uplift) |

**Interpretation (the important part):**
1. **PU machinery works, and bagging > spy** (0.899 vs 0.874 ROC-AUC) — matches the phenol-PU-bagging
   paper's finding. The within-domain PU experiment recovers hidden positives well. ✓ stack validated.
2. **Naive literature→JnJ transfer is ≈random (0.524).** This is the anticipated **cross-domain problem**:
   a model trained on literature-HTE *substrates* does not transfer to JnJ *substrates* with plain DRFP.
   This is the central research challenge for the week, now empirically confirmed — not a bug.
3. Exp A is well below the Neves >90%-precision bar: single fold, small DRFP (1024), no Optuna, and
   `auto_class_weights=Balanced` flattens probabilities (frac>0.85 = 0). Needs tuning, not alarm.

**Next / hypotheses to test:** (a) **domain adaptation** — use a shared *unlabeled* combinatorial grid as U
so P(lit) and U(JnJ-domain) overlap; importance weighting / propensity (SAR-EM). (b) Bigger DRFP (2048)
+ ECFP + Optuna; drop class-weighting for the supervised bar; average all 40 Neves folds. (c) Try the
PU *from literature positives + combinatorial-grid U* (the real design), predicting JnJ. (d) TabPFN /
ChemProp arms. (e) **Perf:** DRFP featurisation is slow (JnJ 303s, lit 213s via per-row assemble) →
cache features to disk + parallelise; this is why we move heavy runs to SLURM.

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
