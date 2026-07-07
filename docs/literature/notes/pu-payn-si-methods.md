# PAYN Supporting Information — implementation, spy-hyperparameter, and separability details

- **Citation:** Boser, Spies, Glorius. Supporting Information for "Yield Prediction of Organic Reactions in Biased Data Sets via Positive-Unlabeled Learning." J. Am. Chem. Soc. 2026, 148, 15066–15075. DOI:10.1021/jacs.6c00127. Data/code: Zenodo 10.5281/zenodo.18609343; github.com/GloriusGroup/PAYN.
- **Type:** method (implementation detail / SI)
- **File:** literature/ja6c00127_si_001 (4).pdf
- **TL;DR:** The SI documents the PAYN software modules (SpySplitting, SpyModel, augmen_negative_identifier), the spy_rate × spy_tolerance grid searches, the smoothness/separability Tanimoto analysis, and the exact dataset featurizations — including the Neves/JnJ24 test-set setup.

## Problem & contribution
Operationalizes the PAYN method from the main text: reproducible config-driven pipeline (YAML/JSON), CatBoost classifier + regressor, Optuna optimization, and quantitative diagnostics for reliable-negative (RN) quality. Confirms the reporting/selection-bias motivation empirically on Reaxys (Suzuki) and four HTE datasets.

## Method (key technical detail)
- **payn.Splitting.SpySplitting:** simulates the PU scenario — take a user fraction (spy_rate) of known positives, hide labels, inject into unlabeled; discard excess negatives to build the "known positives + spy-containing unlabeled" split. Spies keep metadata y=1 to serve as latent-positive anchors.
- **payn.AugmentationModels.SpyModel:** wraps CatBoost classifier; trained to distinguish known positives (s=1) from unlabeled+spy mixture (s=0). Any classifier with a class-probability output is pluggable.
- **augmen_negative_identifier:** uses the spy score distribution + spy_tolerance (default 5%) to set threshold; unlabeled below threshold → reliable negatives; above but unlabeled → discarded "undecisives" (prevents noisy labels); outputs balanced set = true positives + recovered spies + RN.
- **Regression augmentation:** CatBoost regressor trained on positives (true yields) + RN (assigned artificial yield 0%). Baseline "positives-only" model for comparison.
- **RN evaluation module:** negative-specific precision/recall, MCC (best overall metric for RN quality), undecisive-volume tracking, class-separability assessment.
- **Featurization (SI §4.5):** per-component ECFP concatenated then bit-condensed; Ahneman→389 bits, Stevens→433, Perera→275. Neves/JnJ24 uses provided DFT features or DRFP **without** bit condensation.

## Assumptions & when they hold
- Same SCAR + separability + smoothness as main text. SI §5.4 (smoothness/separability) empirically checks the assumptions rather than assuming them.
- **Separability/smoothness check (Table S8):** average Tanimoto similarity of each unlabeled point to its nearest known-positive neighbor. Spies and latent positives show **nearly identical, higher** similarity to known positives than unlabeled negatives (Ahneman 0.85/0.85 vs 0.81; Stevens 0.68/0.68 vs 0.60; Perera 0.87/0.86 vs 0.78). The spies≈latent-positives equality is the empirical justification that spies validly stand in for latent positives (a smoothness/SCAR-plausibility diagnostic).

## Datasets / benchmarks
- Reaxys Suzuki (Oct 2024 export): 40% of points >80% yield, only 4% <20% — quantifies literature high-yield skew.
- Ahneman BH: 58.7% positive (>20%). Stevens borylation: 76.9% positive. Perera Suzuki: 66.3% positive. All combinatorial and **positive-rich** (selection bias baked in).
- **Neves JnJ24 (our test set):** 11,328 newly generated diverse points from 11.5M space; JnJ24+Literature = 23,185 (Ahneman dropped). Yield distribution strongly shifted to **low** yields; **majority below the 20% threshold** (<29% positive) — the negative-rich outlier. Featurized with DFT/DRFP. Final run: spy rate 0.01, spy tolerance 0.05 (low spy rate because few positives).

## Results & claims
- Spy hyperparameters (Figs S7–S8 style grid, 24 combos/dataset): best MCC at spy rate ~0.2 (need enough spies to sample latent-positive distribution) and spy tolerance ~0.025–0.05 (higher tolerance → lower RN fidelity).
- RN confusion matrices (Fig S9): Ahneman & Perera give high-precision RN; **Stevens RN ~50% false negatives** (CatBoost can't separate latent classes in that feature space) — yet augmentation still improves the regressor.
- Final MAEs (Table S9): Ahneman aug 6.78 vs pos-only 11.1 vs full 4.01 (61%); Stevens 13.5/15.9/10.8 (47%); Perera 10.6/12.2/7.20 (33%); **Neves 12.3/27.9/7.05 (75%)**. Scatter plots (S10–S13): positives-only systematically fails to predict <20% yields; augmented corrects this with slightly higher low-yield variance (RN forced to exactly 0% vs true graded low yields).

## Relevance to our BH C–N PU pipeline
Gives us the **exact reproducible knobs and diagnostics** we inherit: spy_rate/spy_tolerance ranges, MCC as RN-quality target, and — most useful for beyond-SCAR work — the **Tanimoto separability/smoothness diagnostic (Table S8)** as a pre-flight check on whether our literature-positive vs unlabeled featurization even supports PU learning. The Neves/JnJ24 setup (features, thresholds, low spy rate for negative-rich data) is the precise test-harness our feasibility models must plug into. Stevens's poor RN fidelity is a cautionary tale: on the real combinatorial BH space, if literature positives and latent negatives overlap in ECFP space, single-threshold SCAR-spy RN will be noisy — a concrete argument for subtype-aware (PULSNAR) or propensity-aware (SAR) thresholds.

## Caveats / limitations / leakage risks
- Neves JnJ24 uses a different featurization (DFT/DRFP, no bit condensation) than the ECFP combinatorial sets — cross-dataset comparisons of RN quality are confounded by featurization.
- RN assigned artificial yield 0% injects a systematic low-yield error (visible as higher variance); for a *feasibility* (classification) reframing this is less harmful.
- Table S8 shows HTE feature spaces are globally high-similarity (constrained space) — small Pos−Neg similarity gaps (0.04–0.08) mean separability is thin; over a broad literature space the gap and thus RN fidelity could differ substantially.
- **Leakage:** JnJ24+Literature explicitly drops Ahneman overlap — we must replicate this de-duplication so mined literature positives never coincide with JnJ test reactions.

## Open questions for us
- Can Table S8's Tanimoto-gap be turned into a quantitative go/no-go (or a propensity-attribute) for whether beyond-SCAR debiasing is even needed per reaction subtype?
- With real literature positives (not simulated masking), does the spies≈latent-positives equality still hold, or does reporting bias make spies unrepresentative of latent positives (a direct SCAR-violation test we can run)?

## Cited works worth chasing (DOIs; 🔒 = likely paywalled)
- Neves et al. 2025, BH OOD / JnJ HTE 2024, ChemRxiv. DOI:10.26434/chemrxiv-2025-xcr46 (open)
- Prokhorenkova et al. 2018, CatBoost, NeurIPS. arXiv:1706.09516 (open)
- Akiba et al. 2019, Optuna, KDD. DOI:10.1145/3292500.3330701 🔒
- Rogers & Hahn 2010, ECFP, JCIM 50:742. DOI:10.1021/ci100050t 🔒
