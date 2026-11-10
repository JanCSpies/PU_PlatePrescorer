> **Provenance.** Summary written by a Claude agent on 2026-10-09 from the full PDF (paper and SI), focused on this project's questions. Page references point to that PDF. Checked only where `docs/design/modeling.md` or ADR 003 relies on it; read the paper before citing a detail.
>
> **Reference.** Boser, Spies, Glorius, "Positivity is All You Need", *J. Am. Chem. Soc.* 2026, 148, 15066–15075, doi:10.1021/jacs.6c00127; code github.com/GloriusGroup/PAYN.

# PAYN ("Positivity is All You Need") — structured summary

Boser, Spies, Glorius. *Yield Prediction of Organic Reactions in Biased Data Sets via Positive-Unlabeled Learning.* J. Am. Chem. Soc. 2026, 148, 15066–15075 (published 1 April 2026; CC-BY 4.0). DOI 10.1021/jacs.6c00127.

Citation convention: "p. 150xx" = main paper page; "SI p. Sxx" = Supporting Information page label (the SI has an unnumbered cover and pages S1–S33).

---

## 1. Problem setting and datasets

**Goal.** The paper sets out to learn from success-biased literature data, where positives dominate, by using PU learning to recover "reliable negatives" (RN) from an unlabeled pool. The authors then train a **yield regression** model on positives plus RN (p. 15066 abstract; p. 15067). The end task is yield regression (MAE). It is **not** ranking or classification of untested pairs (p. 15071; SI p. S14).

**Motivating literature analysis.** The authors exported Suzuki–Miyaura reactions from Reaxys on 23–24 Oct 2024 (SI p. S3). They dropped entries with no yield, reaction SMILES or reagent, and kept reactions with exactly 2 reactants and 1 product (SI p. S3). In this set, 40% of reactions have yield >80% and only 4% have yield <20% (SI p. S3, Fig. S2 on p. S4). The literature data is used **only** to show the bias. It is never used to train or evaluate a model (implied by SI §2 and p. 15069; not stated explicitly).

**Simulation of reporting bias (how P and U were built).** All P and U sets are **synthetic and come from fully labeled HTE data** (p. 15069, Fig. 4):
- Binarisation: a reaction is positive if yield >20% (p. 15069; SI p. S10 `classify_binary`).
- Within each CV training fold, a portion set by the hyperparameter "PU ratio" has its negatives removed and becomes the known positives. The rest of the data (true positives and true negatives together) goes into U with labels hidden (p. 15069, Fig. 4).
- Default partitioning is "Controlled Ratio" (`split_data_with_controlled_PU_ratio`). It enforces a strict known-positive : unlabeled ratio by **discarding excess negatives** (SI p. S11). An "Original Ratio" mode is implemented but was not used as the default (SI p. S11).
- Ratio Positives/Unlabeled = **1.0** in all experiments (SI p. S16, Table S1).
- The masking is random, so this simulates reporting bias only. The authors say selection bias is **not** simulated, although the HTE sets themselves carry selection bias (p. 15069; SI p. S7).

**Datasets (all HTE, four reaction types/sources):**

| Dataset | Reaction | Components | N | % positive (>20%) | Feature length |
|---|---|---|---|---|---|
| Ahneman et al. | Buchwald–Hartwig | 15 aryl halides, 4 Pd cat., 3 bases, 22 additives (one amine: 4-methylaniline) | 3955 | 58.7% | 389 bits |
| Stevens et al. | Ni-cat. borylation (EtOH subset only) | 33 aryl (pseudo)halides, 23 ligands | 779 | 76.9% | 433 bits |
| Perera et al. | Suzuki–Miyaura | 7 indazoles, 4 quinolines, 12 ligands, 8 bases, 4 solvents | 5760 | 66.3% | 275 bits |
| Neves et al. (JNJ24) | Buchwald–Hartwig, non-combinatorial | 68 amines, 110 halides, 32 Pd cat., 8 bases, 6 solvents | 11 328 | 28.2% | DRFP / DFT (no condensation) |

Sources: p. 15069 Fig. 5; SI pp. S4–S8; Fig. 9 on p. 15072. Neves "JNJ24 + Literature" has 23 185 points. The authors used only the 11 328 new JNJ24 points and dropped the Ahneman points (SI p. S7, p. S25).

**Sizes of the labeled data actually used.** These are the training-set sizes averaged over the 5 folds (SI pp. S28–S29, Table S10):

| Dataset | Positives-only training set | Augmented (P + RN) set | Full labeled set |
|---|---|---|---|
| Ahneman | 1061 | 1381 | 2487 |
| Stevens | 245 | 285 | 560 |
| Perera | 1657 | 1962 | 4147 |
| Neves | 1809 | 2735 | 8155 |

Fig. 8 (p. 15072) gives slightly different circle counts for Ahneman: 2847 vs 1054, "37% of labeled datapoints". So the known positives are roughly 1–1.8k per set, which is the same order as our ~1k.

**Reaction classes.** Pd C–N coupling, Ni borylation and Pd Suzuki C–C coupling (p. 15069). **Amide coupling is not studied** (not stated anywhere).

---

## 2. PU method

- **Algorithm.** The authors use the spy technique from Liu et al. (ref 69) as a one-shot RN extraction (p. 15068, Fig. 3; SI p. S11):
  1. Move a random fraction of known positives into U as spies, with s = 0.
  2. Train a binary classifier on P vs U-plus-spies.
  3. Predict p(s=1|x) for all of U.
  4. Set the threshold t_spies so that (1 − spy_tolerance) of the spies score above it (p. 15069).
  5. Points in U with p ≤ t_spies become RN, i.e. RN = {x ∈ U | p(s=1|x) ≤ t_spies} (p. 15069).
  6. Points above the threshold are "undecisives" and are **discarded** (p. 15071; SI p. S14).
- **What is not used.** There is no bagging, no nnPU or uPU risk estimator, no Elkan–Noto calibration and no iterative second step (EM or retraining). The second stage is simply a CatBoost **regressor** trained on P (with their true yields) plus RN. Every RN gets a synthetic yield of 0.0% (SI p. S14). None of these alternatives is mentioned as tried, apart from the naive "U as negative" baseline below.
- **Base classifier.** CatBoost classifier (p. 15069; SI p. S11). The paper says it is interchangeable as long as it yields a class probability (p. 15070; SI p. S11).
- **Class prior.** It is **not estimated and not used**. The spy threshold replaces it. The authors state the SCAR assumption and the identity p(s=1|x) = c·p(y=1|x), with c taken from Elkan & Noto (p. 15068). However, they never estimate c or π (not stated). The text points to "SI Section 3" for the claim that p(y=1) is very low for random substrates (p. 15069). SI §3, however, only describes the HTE datasets and gives no derivation (SI pp. S4–S8).
- **Spy hyperparameters.**
  - Spy rate is "typically 15–20%" (SI p. S11).
  - Spy tolerance defaults to 5% (SI p. S13).
  - Grid searched: spy rate {0.01, 0.05, 0.10, 0.20} × tolerance {0.01, 0.025, 0.05, 0.075, 0.10, 0.15} (SI pp. S20–S23, Tables S5–S7).
  - Final settings: spy rate 0.2 and tolerance 0.05 for Ahneman, Stevens and Perera. For Neves: spy rate **0.01** and tolerance 0.05 (SI p. S25, Table S9). The reason for 0.01 on Neves is not stated.
- **Classifier tuning metric.** This was tuned using downstream regression MAE as the readout. MCC was chosen and used for all final models (SI pp. S17–S18, S25). Optimising the classifier for "precision" was worst on all three datasets (SI p. S18).
- **Model hyperparameters.**
  - Optuna TPE with 50 trials (SI p. S13).
  - Search space: depth and learning_rate. Max depth 12, min learning rate 1e-5 (SI p. S16).
  - Up to 1000 CatBoost iterations with early stopping (SI pp. S12–S13, S16).
  - Random seed 42 (SI p. S16).
- **How negatives are "sampled".** They are not sampled. RN are all U points below the spy threshold (p. 15069).
- **Naive baseline.** Treating all of U as negatives with 0% yield (`generate_unlabeled_as_negative_set`) did **worse than positives-only**, so the authors dropped it as a comparator. No numbers are given (SI p. S14).

---

## 3. Features and representation

- **Default featurisation.** For each component (substrates, ligands, bases, solvents, additives) the authors compute an ECFP with radius 2 and 2048 bits. They **concatenate** the per-component fingerprints and remove zero-variance bits ("condensed_bits") (SI p. S4; SI p. S10; SI p. S16 Table S1). The final lengths are 389, 433 and 275 bits (SI pp. S5–S6).
- **Why generic ECFP.** The authors chose it to give a universally accessible baseline. They concede that domain-tailored descriptors could improve separability (p. 15070) and might close more of the gap on Perera (p. 15072).
- **MFF (multiple fingerprint features).** This re-implements EasyChemML and concatenates RDKit FP, ECFP/FCFP, Layered, MACCS, Avalon, AtomPair and Torsion fingerprints (default 2048 bits × 22 components + MACCS). It is implemented but not used in the reported results (SI p. S10; p. 15070).
- **Neves.** Here the authors used the provided DFT features or DRFP without condensation (SI p. S8). SI p. S24 says DRFP was used for the benchmark.
- **Reaction-level encodings** such as CGR or a product fingerprint: not stated / not used, apart from DRFP on Neves.

---

## 4. Evaluation

- **Splits.** Stratified **random** 5-fold CV, with 10% of the training data held out for hyperparameter validation (p. 15069; SI p. S11, S16). Scaffold (leave-one-group-out) and Butina-cluster splits are **implemented** (SI p. S11). No results are reported for them (not stated). Substrate-disjoint evaluation: not reported.
- **Robustness.** The authors ran 5 seeds × 5 folds = 25 paired runs and a Wilcoxon signed-rank test. Augmented beats positives-only on all four datasets with p < 0.001 (SI p. S30, Table S11; pp. S30–S32, Table S12).
- **Metrics.**
  - RN quality: **negative precision** (share of RN that are true negatives) and **negative recall** (share of U negatives recovered), plus the average yield of RN and confusion matrices over U (p. 15070–15071; SI p. S14).
  - Downstream: **MAE** (primary), plus R² and RMSE in the scatter plots (SI pp. S26–S27).
  - ROC-AUC, PR-AUC, top-k and enrichment for the final classifier: **not reported**. AUC and PR-AUC appear only as candidate tuning targets (SI p. S17).
- **Baselines.**
  - Lower bound: "positives-only" regressor trained only on known positives.
  - Upper bound: "fully labeled" regressor.
  - Size-matched fully labeled regressors (SI pp. S27–S29).
  - The literature uses no other PU method as a comparator (not stated).

**Key numbers (MAE, yield %):**

| Dataset | Positives-only | PAYN | Fully labeled | Share of gap closed | Source |
|---|---|---|---|---|---|
| Ahneman | 11.1 ± 0.7 | 6.8 ± 0.4 | 4.0 ± 0.2 | 61% | p. 15071–15072, Fig. 8; SI Table S9 p. S25 |
| Stevens | 15.9 | 13.5 | 10.8 | 47% | same |
| Perera | 12.2 | 10.6 | 7.2 | 33% | same |
| Neves | 27.9 | 12.3 | 7.05 | 75% | p. 15072; SI p. S25 |

For Neves, the positives-only R² is −0.65 (SI p. S27, Fig. S13). PAYN raised the share of negatives in RN up to 97% (p. 15072).

**RN confusion matrices** (as % of U, mean over 5 folds):

| Dataset | RN that are true negatives | RN that are true positives | Undecisives that are negatives | Undecisives that are positives | Source |
|---|---|---|---|---|---|
| Ahneman | 27.8% | 2.6% | 14.2% | 55.4% | p. 15071, Fig. 7 |
| Perera | 15.1% | 3.3% | 18.8% | 62.7% | p. 15071; SI p. S24 |
| Stevens | 8.1% | 8.4% | 15.2% | 68.4% | SI p. S24, Fig. S9 |

- For Ahneman this gives negative precision **91%** and negative recall **66%** (p. 15071).
- For Stevens, about 50% of RN are false negatives. Training with them still improved MAE (SI p. S24).

**Size-matched upper bound.** When the fully labeled model is trained on the same number of points as the augmented one, PAYN closes more of the gap. For example, Ahneman goes from 61% to 77% (augmented size) or 85% (positives-only size) (SI p. S28, Table S10).

---

## 5. Findings relevant to us

**What worked**
- Spy-based RN extraction plus a regressor beats positives-only on all four HTE sets, and the result is statistically significant (p. 15071; SI p. S30).
- The largest gain came on the most negative-rich and most diverse set, Neves (28% positives) (p. 15072). There, the positives-only model fails badly because it rarely predicts yields below 20% (SI p. S25).
- RN with modest fidelity (Stevens, ~50% precision) still helped. The authors' explanation is that RN is still richer in negatives than U (SI p. S24).
- Classifier-tuning metric matters: MCC was best and precision worst (SI p. S18). Accuracy ranked best on Ahneman (rank 2.4), MCC on Perera (1.8), and AUC on Stevens (2.2) (SI pp. S17–S18).
- Spy rate 0.2 was best on all three main sets. Tolerance 0.025–0.05 was best, and both lower and higher values hurt (SI p. S20).
  - Ahneman best: rate 0.2 / tolerance 0.025, MAE 6.56.
  - At rate 0.01, results were often insensitive to tolerance, giving identical MAEs. This is probably because there were too few spies to set a threshold (SI pp. S20–S21; that interpretation is mine).
- Separability check: the authors compare the mean Tanimoto similarity to the nearest known positive for spies, latent positives and unlabeled negatives (SI p. S23, Table S8).
  - Spies and latent positives score almost the same: 0.85/0.85 (Ahneman), 0.68/0.68 (Stevens), 0.87/0.86 (Perera).
  - Negatives score lower: 0.81, 0.60, 0.78.
  - The authors take this as evidence for smoothness and separability, even though the gap is only Δ = 0.04–0.08.

**What failed or was weak**
- Treating all of U as negative was worse than positives-only (SI p. S14).
- The false-positive rate is high: 14.2% of U on Ahneman are true negatives left in the undecisives (p. 15071). Latent positives get widely spread, high-variance scores (p. 15070, Fig. 6A).
- Gains are smaller where positives and negatives overlap in feature space (Perera, 33%) (p. 15072).
- The synthetic 0% yield given to RN adds systematic error compared with the fully labeled model (SI pp. S25, S27).

**Sensitivity**
- To P size / P:U ratio: **not studied**. The ratio is fixed at 1.0 (SI p. S16).
- To label noise in P: not studied (not stated).
- To class prior: not studied. No prior is used (section 2).
- To dataset size: SI §5.8 covers this for the upper bound only (SI pp. S27–S29).

**Bias assumptions**
- SCAR is stated as an assumption (p. 15068). In their simulation, positives are removed at random, so SCAR holds by construction (p. 15069).
- Real literature bias violates SCAR, because reporting depends on the chemistry. The paper does **not** test this or discuss any departure from SCAR (not stated).
- Selection bias is acknowledged but not simulated (p. 15069).
- HTE sets have selection bias themselves: their conditions were chosen to give high yields (SI p. S7).
- **Domain shift** between training and test sources is not studied. Train and test always come from the same HTE set (p. 15069, 15071).
- Real literature data was never used as P (not stated otherwise).

---

## 6. Code

- **Repository:** https://github.com/GloriusGroup/PAYN. Docs at https://GloriusGroup.github.io/PAYN. Data and per-fold role assignments (labeled positive, spy, RN, undecisive, etc.) are on Zenodo, https://doi.org/10.5281/zenodo.18609343 (SI p. S3, S8, S24).
- **Language and dependencies:** Python 3 with pandas, RDKit, CatBoost, Optuna, numpy and scikit-learn. MLflow is used for logging (SI p. S8–S9).
- **Structure:** modules `ConfigLoader`, `Logging`, `DataLoader`, `DataSchema`, `Featurisation` (incl. `MFFFingerprinter`), `Splitting.DataSplitting`, `Splitting.SpySplitting`, `AugmentationModels.SpyModel` (with `augmen_negative_identifier`), `RegModel`, `Optimization` (Bayesian/Grid), `Recombination`, `Evaluation`, `Visualisation`, `Utility`. There are more than 125 tests (SI pp. S8–S16).
- **Running it:** experiments are driven by one YAML/JSON config file. Every key can be overridden from the command line through `ConfigArgParser` (e.g. `--spy_model_eval_metric`), and Slurm is supported (SI p. S8). Exact install and run commands: not stated. Software licence: not stated (only the article is CC-BY 4.0, p. 15066).

---

## 7. Recommendations for our setting, and where PAYN's conclusions may not carry over

**Differences that could break PAYN's conclusions**
1. **P is real literature data, not randomly masked HTE.** PAYN's P/U split satisfies SCAR by construction (p. 15069). Our ~1k literature HATU/DIPEA/DMF positives carry real selection and reporting bias, which correlates with chemistry. Under that bias, spies are a biased sample of all positives, so t_spies may be wrong. PAYN offers no evidence for this regime.
2. **U is huge and imbalanced: 529 × 538 ≈ 284.6k pairs against ~1k P.** PAYN always uses P:U = 1:1 with excess negatives thrown away (SI p. S11, S16). The spy threshold, the 0.2 spy rate and the MCC tuning were all tuned at 1:1. To mimic their regime we would have to subsample U. Otherwise the defaults need to be re-tuned.
3. **The class prior in U is unknown and probably very different.** PAYN's U is 58–77% positive for three of the four sets (SI pp. S4–S6), which is unusually positive-rich. An inventory cross-product of amides with fixed HATU conditions probably has a high success rate too. That is not established, and PAYN gives us nothing on estimating π.
4. **Task.** PAYN's end task is yield regression with RN set to 0% (SI p. S14). We rank or classify untested pairs. Their evidence of success is MAE, not ranking metrics (no ROC-AUC, PR-AUC or top-k; SI p. S17).
5. **Evaluation source.** PAYN trains and tests on the same HTE set with random 5-fold CV (p. 15069). We train on literature plus inventory and test on Chemlex HTE, which is a cross-source domain shift. PAYN does not report scaffold or Butina results even though both are implemented (SI p. S11). Its numbers are therefore optimistic relative to our setup.
6. **Variable components.** PAYN's sets vary catalysts, ligands, bases and so on (p. 15069). Our conditions are fixed, so only the acid and amine vary. Their concatenated per-component ECFP maps onto ours as [acid ECFP ‖ amine ECFP].
7. **Different reaction class:** amide coupling is not covered (p. 15069).

**Concrete recommendations**
- **Use PAYN as a baseline, matched to its published settings:**
  - CatBoost classifier tuned on MCC.
  - Spy rate 0.2 and spy tolerance 0.05.
  - Per-component ECFP4 with 2048 bits each, concatenated, zero-variance bits removed.
  - Optuna TPE with 50 trials over depth and learning rate.
  - Seed 42 (SI p. S16, S25).
- **Report both P:U regimes:** a 1:1 subsampled U (PAYN-faithful) and the full U.
- **Use Chemlex as the ground truth for U, as PAYN uses HTE:** for inventory pairs that also appear in Chemlex, report negative precision and negative recall of RN, plus the confusion matrix of RN vs undecisive (p. 15071; SI p. S14). This is the fairest like-for-like comparison with Fig. 7.
- **Ranking metrics.** Add ROC-AUC, PR-AUC and top-k / enrichment on Chemlex, since PAYN gives none. Say explicitly that the comparison is new rather than a reproduction.
- **Splits.** Use substrate-disjoint or Butina splits next to random splits. PAYN's code already supports them (SI p. S11).
- **Comparison set.** Include the naive "U = negative" baseline (it lost to positives-only in PAYN, SI p. S14) and at least one prior-based method (nnPU or Elkan–Noto with an estimated π). PAYN does not compare against these.
- **Spy diagnostics.**
  - Grid the spy rate and tolerance again, because the optimum moved to 0.01 on Neves (SI p. S25).
  - With ~1k positives, a 0.2 spy rate takes ~200 out of P. Check the cost to the classifier, a trade-off the paper notes (SI p. S19).
- **Separability.** Run PAYN's nearest-neighbour Tanimoto diagnostic (SI p. S23, Table S8): spies vs Chemlex positives vs Chemlex negatives. If spies don't look like the Chemlex positives, the literature bias is breaking SCAR, and the spy threshold should not be trusted.
- **Positive threshold.** For comparability with PAYN, binarise Chemlex at yield >20% (p. 15069), or report both thresholds.
- **Discussion.** Cite PAYN's own caveats, and then position our work as the real-literature, non-SCAR and cross-source test that PAYN does not do:
  - SCAR is assumed (p. 15068).
  - Selection bias is not simulated (p. 15069).
  - HTE sets have selection bias themselves (SI p. S7).
