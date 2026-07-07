# Neves et al. — Robust OOD Prediction of Buchwald–Hartwig Reactions (OUR TEST-SET SOURCE)
- **Citation:** P. Neves, B. Hao, S. Aikonen, J. B. Diccianni, J. K. Wegner, P. Schwaller, I. I. Strambeanu.
  *Robust Out-of-Distribution Prediction of Buchwald–Hartwig Reactions.* ChemRxiv 2025 (13 Oct 2025), preprint.
  DOI 10.26434/chemrxiv-2025-xcr46. Affiliations: Johnson & Johnson + EPFL (LIAC/Schwaller).
- **Type:** application + data (generates our test set)
- **File:** literature/chemrxiv-2025-xcr46.pdf
- **TL;DR:** Standardize+merge all public BH HTE datasets with a new 11.3k-reaction J&J HTE campaign, use active
  learning to expand chemical space, and show that data **diversity** (not size or model complexity) drives
  out-of-distribution binary-reactivity prediction — a plain Random Forest hits >90% precision on high-confidence OOD.

## Problem & contribution
BH C–N coupling outcome prediction fails OOD because ELN/literature data is noisy, positive-biased, and lacks
unified structure-per-entity representations. Contribution: (1) a unified curation pipeline → "unique-structure-per-entity"
BH dataset (24.5k→27.5k reactions across sources); (2) a new J&J HTE campaign ("JnJ24", 11.3k reactions) built in
two phases (baseline model → active-learning expansion over a 64M virtual BH space via BERT-Enriched-Embeddings +
Ranked Batch-Mode Active Learning using uncertainty + Tanimoto distance); (3) a data-centric **CRDS** (Compound-Reaction
Diversity Score) that correlates 0.967 (Spearman) with OOD ROC-AUC; (4) three wet-lab validations (relabel, rescue, prospect).

## Dataset(s) — scope, size, conditions, feasibility definition
- **JnJ24 (our test set):** 11,328 rxns; 110 aryl halides × 68 amines (888 measured pairs of 7,480); 32 Pd precatalysts,
  8 bases, 6 solvents; 60.9% negative. **Feasibility = assay yield ≥10%** (UPLC-MS-CAD; verified from data). Four
  dominant P2Et "general condition" sets (RuPhos/BrettPhos/XPhos/P(tBu)3-Pd-G4). See `bh-JnJ-testset-spec.md`.
- **Unified corpus:** 96- & 384-well plates, UPLC-MS-CAD quantification. Merges 7 public BH HTE sets + JnJ24; the
  Merck–MIT Pd C–N set (Ha et al. *JACS* 2025, 147, 19602; 4,204 rxns) added during writing.
- **Phase-2 virtual space:** 64M BH reactions explored by the recommender; prospective test space = 13 catalysts,
  12 bases, 7 solvents, 12,252 aryl halides, 18,364 amines ≈ 2.46×10¹¹ reactions.

## Method & features
- Features: **DRFP** reaction fingerprints (also DRFP_subst_prod, QM descriptors, additive flags in the released file).
- Models: 3 classical ML models + 1 SMILES-based "AI" model; **Random Forest** is the headline winner.
- Recommender: binary reactivity classifier + reagent dictionary + uncertainty → ranks conditions/substrates,
  acts as HTE plate designer. Confidence = normalized inverse uncertainty in [0.5, 1.0], binned into 5 buckets.

## OOD / splitting strategy & metrics
- **OOD split:** cluster ALL reactions with DRFP + K-means; test only reactions in clusters with no training reactions
  (leave-cluster-out). Repeated ~20× ("clustered 20 times"; released file exposes 10 iterations × 12 clusters, with a
  4-fold train/test list per iteration). Addresses substrate+condition novelty jointly.
- **Metrics:** 10 computed → 4 headline: **ROC-AUC, Balanced Accuracy, AU-PR-C (AUPRC)**, + precision/uncertainty
  calibration. Explicitly designed for class-imbalanced OOD (precision-focused, serviceable recall, calibrated uncertainty).

## Results & claims
- RF on curated data: **>90% precision on high-confidence (>0.85) OOD** predictions with <25k samples.
- **CRDS↔ROC-AUC Spearman = 0.967** → diversity beats size; "All w/ JnJ24" beats every single prior dataset on all metrics.
- Wet-lab: (a) relabel — flagged 19/256 "0% yield" legacy rxns as mislabeled; 18 confirmed >10% (95%). (b) rescue —
  11 historically failed pairs, top-4 recommended conditions each → 27/44 succeeded, viable routes for 10/11 (91%).
  (c) prospect — from ~2.46×10¹¹ space, high-conf (>0.85) predictions gave 33% success vs 19% baseline (1.64× enrichment).

## Relevance to our pipeline (esp. leakage risk vs JnJ test set)
- **This IS our test set** — reproduce their OOD folds from the `iteration_*` columns; match the 10% feasibility threshold.
- Bar to beat: RF/DRFP **>90% precision @ conf>0.85 OOD**, high ROC-AUC/AUPRC/Balanced-Acc.
- Their approach is fully-supervised (uses real negatives). **Our angle = PU/literature-positive-only** (no J&J negatives),
  which is genuinely different and defensible: we treat their measured negatives as *unavailable* and predict feasibility
  from literature positives + implicit/spy negatives.
- **Leakage:** the other 7 curated sources share **0 substrate pairs** with JnJ (verified) — safe to use as positives if
  we drop any coincident pair after RDKit canonicalization. Their curated file already encodes the "no-leak" clusters.

## Open questions for us
- Do we predict feasibility **per-condition** or **pair-level (any-condition-works)**? JnJ is condition-resolved; decide the target granularity.
- Should we adopt their **CRDS** to justify our unlabeled-space design and diversity of mined positives?
- Reuse their exact `iteration_*` OOD folds vs. build our own scaffold/cluster splits — reuse gives a head-to-head vs their RF.
- The sub-threshold grey zone (1,774 negatives with 0<yield<10%) — treat as hard negatives or unlabeled?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Rinehart, …, Doyle. ML tool for substrate-adaptive Pd C–N conditions. *Science* 2023, 381, 965. DOI 10.1126/science.adg2114 🔒 (origin of the 4 P2Et general conditions)
- Ha et al. Pharma-relevant Pd C–N reactivity models via HTE (Merck–MIT). *JACS* 2025, 147, 19602. DOI 10.1021/jacs.5c... 🔒 (the "Merck 2025" source)
- King-Smith et al. Chemical 'reactome' via HTE (39k). *Nat. Chem.* 2024, 16, 633. DOI 10.1038/s41557-023-01393-w 🔒
- Ahneman/Doyle. Predicting C–N performance with ML. *Science* 2018, 360, 186. DOI 10.1126/science.aar5169 🔒 ("Dreher Doyle 2018" source)
- Probst, Schwaller, Reymond. DRFP. *Digital Discovery* 2022, 1, 91. DOI 10.1039/D1DD00006C 🔓
- Zhong et al. Global feasibility+robustness (see bh-zhong note). *Nat. Commun.* 2025, 16, 4522. 🔓
