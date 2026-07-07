# JnJ HTE 2024 — Test-Set Specification (BH C–N feasibility)

**Purpose:** precise, data-verified spec of the "JnJ HTE 2024" Buchwald–Hartwig HTE dataset that is our
held-out **test set**, so data-engineering can build the UNLABELED space and mine literature positives
**without leakage**. All numbers below are verified directly against the local file, not just the paper.

- **Source paper:** Neves, Hao, Aikonen, Diccianni, Wegner, Schwaller, Strambeanu, *Robust Out-of-Distribution
  Prediction of Buchwald–Hartwig Reactions*, ChemRxiv 2025, DOI 10.26434/chemrxiv-2025-xcr46 (preprint, CC-BY-NC-ND).
- **Data repo:** GitHub `schwallergroup/bh-hte-ood`; Zenodo https://zenodo.org/records/17634928.
  Canonical file in repo: `data/BH_HTE_Curated_Dataset_v2025-11.csv`.
- **Local file analyzed:** `datasets/raw/Neves/BH_HTE_Curated_Dataset_v20260219.csv`
  (a later v2026-02-19 build; 27,497 records total across 8 sources; the JnJ subset is our test set).

## 1. What "JnJ HTE 2024" is
The `Source == "JNJ HTE 2024"` subset of the curated file. It is the **newly generated J&J HTE campaign**
(labeled "JnJ24" in the paper). It is NOT the whole curated file — the curated file merges 8 sources
(see §5). The JnJ subset is the OOD test target; the other 7 sources are candidate literature/HTE positives.

## 2. Scope (verified counts, JnJ subset only)
| Quantity | Value |
|---|---|
| Reactions (rows) | **11,328** (paper: "11.3k") |
| Unique aryl halides | **110** |
| Unique amines | **68** |
| Possible aryl×amine pairs | 110×68 = **7,480** |
| Aryl–amine pairs actually measured | **888** (only ~11.9% of the 7,480 grid) |
| Unique catalysts (ligand·Pd precatalyst) | **32** |
| Unique bases | **8** |
| Unique solvents | **6** |
| Feasibility label balance | **label 0 = 6,896 (60.9%)**, label 1 = 4,432 (39.1%) |

Not a full combinatorial grid: 888 substrate pairs are each run under a subset of the 32×8×6 condition
space. Reactivity is sparse and substrate-pair-driven.

## 3. Feasibility definition (VERIFIED, exact threshold)
- Column `Yield` = UPLC-MS-CAD assay yield in **percent** (range 0.0–114.3; can exceed 100 due to assay/response artifacts).
- Column `labels` ∈ {0,1}. **label = 1 iff Yield ≥ 10.0 %**, else 0. Verified exactly: label-0 yields span 0.0–9.9,
  label-1 yields span 10.0–114.3. No overlap.
- So "feasible / reacts" = **≥10% assay yield**. This is the classification target we must reproduce.
- Of the 6,896 negatives, 1,774 have a nonzero-but-<10% yield (a "sub-threshold" grey zone); the rest are ~0%.

## 4. Conditions — the 4 dominant "general conditions" (P2Et sets)
Bases (by count): **P2Et 3,468**, K3PO4 2,016, Cs2CO3 1,920, LiHMDS 1,336, NaOtBu 807, NaBHT 735, BTMG 718, NaOTMS 328.
Solvents: t-AmOH 2,400, 1,4-Dioxane 2,294, DMSO 2,276, DMA 2,206, 2-Me-THF 1,152, MeCN 1,000.
Catalysts: 32 Buchwald-type Pd precatalysts (mostly Pd-G4/G3), top = RuPhos-Pd-G4, P(tBu)3-Pd-G4, XPhos-Pd-G4, BrettPhos-Pd-G4, …

The **4 dominant (catalyst, base, solvent) "general condition" sets** (each ~580–606 rows, run broadly across pairs):
1. **RuPhos-Pd-G4 / P2Et / t-AmOH** (606)
2. **BrettPhos-Pd-G4 / P2Et / t-AmOH** (605)
3. **XPhos-Pd-G4 / P2Et / DMA** (582)
4. **P(tBu)3-Pd-G4 / P2Et / DMSO** (581)

These 4 P2Et-based sets are the Doyle/Rinehart-style "substrate-adaptive general conditions" for Pd C–N
(Rinehart et al., *Science* 2023, 381, 965). They form a dense sub-block screened across many substrate pairs;
the remaining ~9,000 rows are sparser condition scouting (K3PO4/Cs2CO3/LiHMDS in dioxane/2-Me-THF/MeCN, etc.).

## 5. The other 7 sources in the curated file (candidate literature positives)
| Source | rows | label0 | label1 |
|---|---|---|---|
| JNJ HTE 2024 (**TEST**) | 11,328 | 6,896 | 4,432 |
| Dreher Doyle 2018 (Ahneman *Science* 360) | 4,312 | 1,243 | 3,069 |
| Merck 2025 (Ha et al. *JACS* 147, 19602) | 4,204 | 2,408 | 1,796 |
| Roche 2023 | 3,359 | 1,882 | 1,477 |
| HiTEA 2024 | 2,632 | 2,127 | 505 |
| Dreher 2014 | 768 | 714 | 54 |
| AZ ELN 2023 | 750 | 159 | 591 |
| Dreher 2016 | 144 | 103 | 41 |
| **Total** | **27,497** | 15,532 | 11,965 |

All share the same schema (Aryl SMILES, Amine SMILES, Catalyst/Base/Solvent Name+SMILES, Product SMILES,
DRFP, DRFP_subst_prod, QM, Yield, labels, Additive_* flags, and per-iteration OOD split columns).

## 6. OOD split (how Neves defined/evaluated OOD)
- **Method:** cluster ALL reactions with **DRFP** reaction fingerprints + **K-means**; a reaction is "test" only if
  it lies in a cluster that contains **no** training reactions (leave-cluster-out → genuinely unseen substrate/condition
  regions). Repeated many times (paper: "clustered 20 times"; file exposes 10 iterations).
- **File columns:** `iteration_0..9 cluster` (12 K-means clusters each) and `iteration_0..9 train_test`
  (each cell is a 4-element list of train/test assignments = 4 held-out folds per clustering iteration).
  → Reuse these columns directly to reproduce their exact OOD folds instead of inventing our own.
- **Metrics:** 10 metrics computed; 4 headline ones reported = **ROC-AUC, Balanced Accuracy, AU-PR-C (AUPRC)**,
  plus a precision/uncertainty-calibration metric. Predictions binned into 5 confidence buckets; calibration
  assessed vs a diagonal.
- **Headline result:** a plain **Random Forest** on curated DRFP features reached **>90% precision** on
  high-confidence (>0.85) OOD predictions. Diversity (their CRDS score), not dataset size or model complexity,
  drove OOD performance (Spearman 0.967 between CRDS and ROC-AUC).

## 7. Leakage analysis vs the other sources (VERIFIED)
Overlap of each other source's substrate PAIRS / substrates with the JnJ test subset:

| source | aryl∩ | amine∩ | **pair∩** | cat∩ | base∩ | solv∩ |
|---|---|---|---|---|---|---|
| Dreher Doyle 2018 | 2 | 0 | **0** | 0 | 2 | 0 |
| Dreher 2014 | 1 | 0 | **0** | 1 | 2 | 1 |
| Merck 2025 | 0 | 0 | **0** | 0 | 0 | 1 |
| Roche 2023 | 0 | 0 | **0** | 0 | 1 | 2 |
| HiTEA 2024 | 0 | 0 | **0** | 1 | 5 | 2 |
| AZ ELN 2023 | 0 | 0 | **0** | 0 | 4 | 4 |
| Dreher 2016 | 0 | 0 | **0** | 0 | 4 | 2 |

- **No other source shares a single aryl–amine PAIR with JnJ** → pair-level leakage from these literature/HTE
  sources is essentially nil. Individual-substrate overlap is tiny (≤2 aryls, 0 amines).
- **Conditions overlap heavily** (bases/solvents/some catalysts are shared) — expected; condition identity is
  NOT leakage for substrate-pair-OOD, but if we ever do condition-OOD we must track it.

## 8. Cernak 50k C–N (external literature source) vs JnJ
- Cernak `50K_dataset_v2.csv` (JACS 2026, see cernak note) is an **informer/ligand screen**: only **4 electrophiles
  × 2 nucleophiles**, ~163 ligands, 34 metal precatalysts, 17 bases, 4 solvents.
- Substrate overlap with JnJ: **morpholine and 4-phenylpiperidine are NOT among JnJ's 68 amines**; **3-bromopyridine
  (`Brc1cccnc1`) IS one of JnJ's 110 aryl halides**. Because Cernak has only 8 substrate pairs and none coincide
  with JnJ's 888 measured pairs, **Cernak → JnJ pair leakage = 0**; but the shared aryl (3-bromopyridine) means
  aryl-identity is not fully disjoint — track it if doing aryl-OOD.
- (NB: overlap check above used raw-string SMILES; re-run with RDKit-canonical SMILES before final leakage sign-off.)

## 9. Actionable rules for our pipeline
1. **Test set = the 11,328 JnJ rows** (or a defined OOD subset of them using the `iteration_* train_test` columns).
2. **UNLABELED space** = full combinatorial JnJ substrate grid (110 aryl × 68 amine × condition sets) **minus**
   the 888 measured pairs (or minus test rows). Everything unmeasured = unlabeled, per the PU premise.
3. **Positives** to mine = label-1 rows from the 7 non-JnJ sources + external (Cernak, ORD, USPTO, etc.), but
   **exclude any row whose (aryl, amine) canonical pair appears in the JnJ test pairs** (verified currently 0, but
   enforce programmatically after RDKit canonicalization).
4. **Feasibility threshold = 10% assay yield** — match it exactly when binarizing any yield-bearing source.
5. **Reproduce their OOD folds** from the `iteration_*` columns; report ROC-AUC, Balanced Acc, AUPRC, precision@high-conf.
6. Beatable baseline: RF on DRFP, **>90% precision @ conf>0.85** on OOD. That is the bar.
