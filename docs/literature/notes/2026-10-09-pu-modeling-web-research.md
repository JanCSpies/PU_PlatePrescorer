> **Provenance.** Summary written by a Claude agent on 2026-10-09 from the full PDF (web sources listed inline), focused on this project's questions. Page references point to that PDF. Checked only where `docs/design/modeling.md` or ADR 003 relies on it; read the paper before citing a detail.
>
> **Reference.** Web research on PU methods, reporting bias, domain shift, featurisation and evaluation (not one paper). Classic references marked "not re-checked" were cited from memory.

# PU pre-scoring of a HATU/DIPEA/DMF amide plate: modeling research notes

Date: 2026-10-09. Sources were checked by web search or fetch in this session unless a line is tagged **[not re-checked]**. Those are standard references cited from memory with full bibliographic details. Tagged **[unverified]** means I could not confirm the claim.

---

## 0. Two facts that shape everything below

**Chemlex is Zhong et al., Nat. Commun. 2025, 16:4522.** Full title: "Towards global reaction feasibility and robustness prediction with high throughput data and Bayesian deep learning", doi:10.1038/s41467-025-59812-0 (https://pmc.ncbi.nlm.nih.gov/articles/PMC12081921/).
- 11,669 acid–amine reactions: 272 acids, 231 amines, 6 condensation reagents (HATU among them), 2 bases, 1 solvent.
- Yield is an *uncalibrated* LC-MS UV ratio. Positive means ≥ 20%.
- Substrates are commercial compounds **down-sampled to match the category proportions of Pistachio patent reactions** (MaxMin within each category).
- About 5,600 reactions were **added by expert rules (nucleophilicity, steric hindrance) to create likely negatives**.
- Their supervised Bayesian NN reached AUC 0.94 on a random split, 0.87 with one reactant unseen, and **0.79 with both reactants unseen**.
- The main text does not say what input features the BNN used. They are in SI Fig. 6, which I did not see **[unverified]**.

Consequences:
1. A supervised model trained on thousands of in-domain HTE labels reached ~0.79 AUC both-unseen, across all reagents. Treat that as a rough ceiling, not a target, for a PU model trained on ~1k literature positives.
2. Chemlex substrates were sampled to resemble *patent* chemistry, so they are probably closer to P than your lab inventory is. Chemlex results will therefore **understate the P-vs-inventory shift** (Section 3). Rule-designed negatives also inflate "easy" negatives. Check whether the 1,725/1,313 HATU rows include them and report results with and without them.

**PAYN** is Boser, Spies, Glorius, "Yield Prediction of Organic Reactions in Biased Data Sets via Positive-Unlabeled Learning", JACS 2026, 148(14), 15066–15075, doi:10.1021/jacs.6c00127. Preprint: ChemRxiv 2025, doi:10.26434/chemrxiv-2025-hq4rx (https://www.cambridge.org/engage/chemrxiv/article-details/68f03168bc2ac3a0e031d182). Code and README: https://github.com/GloriusGroup/PAYN.

From the README and abstract:
- **PU flavour:** a two-step "spy" method. Known positives are injected into U as spies. A CatBoost "SpyModel" scores U, and reliable negatives (RN) are taken below a dynamic threshold set by a spy-tolerance hyperparameter. P+RN then trains a CatBoost *yield regressor*.
- **Features:** ECFP, an in-house "Multi-Feature Fingerprint (MFF)" with automated bit condensation, or custom features.
- **Class prior:** no explicit prior estimate is described. The spy threshold plays that role implicitly.
- **Evaluation:** literature bias was *simulated* on fully labeled HTE sets (Ni borylation, Buchwald–Hartwig, Suzuki–Miyaura). They report negative precision and recall for the RN set plus downstream yield metrics. Hyperparameters were tuned with Optuna.
- **Gap for you:** PAYN simulates bias inside one HTE dataset, so P and U share a substrate pool. It does not appear to test the case where P comes from literature and U from a *different* inventory. I have not read the full paper, so this is **[unverified]**. It is the main methodological gap your project can fill.

---

## 1. PU methods for this setting

**Notation.** s = "reported in literature under HATU/DIPEA/DMF". y = "would work (≥ 20% conversion)". c = P(s=1 | y=1) is the labeling propensity. π = P(y=1) in U.

| Method | Needs π or c? | Notes for your case |
|---|---|---|
| **Elkan & Noto 2008** ("non-traditional classifier"): train P vs U, then f(x) = g(x)/c | c, only for calibration | Under SCAR, g(x) = c·f(x). **Ranking by g equals ranking by the true posterior**, so for a top-96 plate the prior is irrelevant *if SCAR holds*. Elkan C, Noto K. *Learning classifiers from only positive and unlabeled data.* KDD 2008, 213–220 **[not re-checked]** |
| **Bagging PU** (Mordelet & Vert): many classifiers, each P vs a random subsample of U, with out-of-bag aggregation | No | Robust, cheap, and suits the 1k-P / 285k-U imbalance. Gives a ranking, not a probability. Mordelet F, Vert J-P. *A bagging SVM to learn from positive and unlabeled examples.* Pattern Recognit. Lett. 2014, 37, 201–209 **[not re-checked]** |
| **uPU / nnPU**: unbiased PU risk, with the negative part clipped at 0 in nnPU | **Yes, π** | Needed for flexible models like NNs, which overfit under uPU. Too much machinery for ~1k positives with GBMs. Mis-specified π mainly hurts calibration. du Plessis, Niu, Sugiyama NeurIPS 2014 / ICML 2015; Kiryo R, Niu G, du Plessis MC, Sugiyama M. *Positive-unlabeled learning with non-negative risk estimator.* NeurIPS 2017 **[not re-checked]**. Overview in the survey: https://arxiv.org/abs/1811.04820 |
| **Two-step / spy (PAYN)** | Implicit (spy threshold) | Gives a reliable-negative set that you can inspect and show to chemists. Sensitive to threshold, and RNs inherit any shift artefact (Section 3) |
| **SAR / propensity PU** (Bekker, Robberechts, Davis): model e(x) = P(s=1 \| y=1, x) with its own features | Propensity model | Fits your situation: popularity and "drug-likeness" drive reporting. ECML-PKDD 2019, LNCS, doi:10.1007/978-3-030-46147-8_5 (https://arxiv.org/abs/1809.03207). The survey (Bekker J, Davis J. *Learning from positive and unlabeled data: a survey.* Mach. Learn. 2020, 109, 719–760; https://arxiv.org/abs/1811.04820) classifies assumptions as SCAR, SAR and others |
| **Positive-shift PU / PU-AUC** | Some need no π | Section 3 |

**Class-prior estimation.**
- **TIcE** (Bekker & Davis, AAAI 2018, doi:10.1609/aaai.v32i1.11715; https://ojs.aaai.org/index.php/AAAI/article/view/11715): estimates c by decision-tree induction.
- **KM1/KM2** (Ramaswamy, Scott, Tewari, *Mixture proportion estimation via kernel embeddings of distributions*, ICML 2016 **[not re-checked]**).
- The Bekker–Davis survey reports that **KM2 and TIcE were the most accurate under SCAR, and TIcE was much faster** (https://arxiv.org/abs/1811.04820).
- Both estimators **assume SCAR and that P and U share support**. Neither holds here, so any π estimate on "lab inventory U" is suspect.

A better route to π: estimate it **empirically from Chemlex HATU train-side labels**, where the positive rate is 32.8% on validation. Then treat π as a sensitivity range (e.g. 0.2–0.6) rather than a fitted number.

**Do you need calibrated probabilities?**
- To pick 96 wells, you need a ranking. AUC, PR-AUC and top-k are all rank-based.
- Calibration matters only if (a) chemists use an absolute cut-off ("only run wells with p > 0.5"), or (b) you combine the score with cost or diversity terms.
- If you need it, recalibrate on Chemlex validation (isotonic or Platt). Do not trust PU-derived probabilities.

**PU for chemistry and synthesizability.**
- Crystal synthesizability: Jang J, Gu GH, Noh J, Kim J, Jung Y. *Structure-based synthesizability prediction of crystals using partially supervised learning.* JACS 2020, 142, 18836–18843 (https://koasas.kaist.ac.kr/handle/10203/278892). Uses transductive bagging PU with a GCN "CLscore" and was validated *temporally* (TPR 86.2% on materials reported 2015–19). The temporal holdout is a useful idea for your P.
- MXenes: Frey NC et al. ACS Nano 2019, 13, 3031 **[not re-checked]**. Overview: https://towardsdatascience.com/positive-and-unlabeled-materials-machine-learning-8b216edea899
- SynCoTrain, a dual-classifier co-training PU method: https://arxiv.org/html/2411.12011v1
- Ternary oxides PU: Digital Discovery 2025, https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00065c
- **"TACIT": I found no PU/synthesizability paper by that name [unverified]**. Ask whoever suggested it for the reference.

These materials-science works report *recall on held-out positives* because they have no true negatives. That is a weakness you avoid by having Chemlex.

**Recommendation (Q1).** Start with **bagging PU over gradient-boosted trees or RF** (prior-free, ranking-only) and an **Elkan–Noto P-vs-U classifier** as the simplest PU baseline. Add a **spy/RN variant** to compare with PAYN. Skip nnPU unless you move to neural encoders. Use the prior only for calibration and sensitivity analysis, taken from Chemlex.

---

## 2. Reporting bias / SCAR violation

**Evidence:**
- **Beker W, Roszak R, Wołos A, Angello NH, Rathore V, Burke MD, Grzybowski BA.** *Machine Learning May Sometimes Simply Capture Literature Popularity Trends: A Case Study of Heterocyclic Suzuki–Miyaura Coupling.* JACS 2022, 144, 4819–4827, doi:10.1021/jacs.1c12005 (https://pmc.ncbi.nlm.nih.gov/articles/PMC8949728). On over 10k literature examples, models from feed-forward nets to GCNs **did not beat frequency-based assignment of conditions**. The authors blame chemist preference, reagent availability and missing negatives.
- **Strieth-Kalthoff F, Sandfort F, Kühnemund M, Schäfer FR, Kuchen H, Glorius F.** *Machine Learning for Chemical Reactivity: The Importance of Failed Experiments.* Angew. Chem. Int. Ed. 2022, 61, e202204647, doi:10.1002/anie.202204647 (https://www.wi.uni-muenster.de/publication/81765382). Models are fairly robust to experimental noise but strongly hurt by human selection and reporting bias (https://www.chemistryworld.com/news/yield-predicting-ai-needs-chemists-to-stop-ignoring-failed-experiments/4015662.article).
- **Liu Z, Moroz YS, Isayev O.** *The challenge of balancing model sensitivity and robustness in predicting yields: a benchmarking study of amide coupling reactions.* Chem. Sci. 2023, 14, 10835–10846, doi:10.1039/d3sc03902a (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10566507/). On 41,239 **literature amide couplings**, the best model reached R² ≈ 0.395 (0.457 after removing "reactivity cliffs" and uncertain reactions), against R² ≈ 0.9 on HTE data. This is directly relevant: amide literature yields carry little signal.
- **Saebi M et al.** *On the use of real-world datasets for reaction yield prediction.* Chem. Sci. 2023, 14, 4997–5005, doi:10.1039/D2SC06041H (https://pubs.rsc.org/en/content/articlehtml/2023/sc/d2sc06041h). A GNN that worked on HTE benchmarks gave no predictive model on pharma ELN data.
- Raghavan et al. (ACS Cent. Sci. 2023, below) note that in one literature Ni C–O dataset the important features **encoded scale or publication type rather than chemistry** (https://pmc.ncbi.nlm.nih.gov/articles/PMC10755851/).

**What this means for P.**
- s=1 depends on x through things unrelated to y: the drug-likeness of products, how popular the building blocks are, whether a patent needed that amide, and whether HATU was the default chosen.
- This is **SAR, not SCAR**. Under SAR, the Elkan–Noto argument that "ranking by g is ranking by f" breaks.
- Also, a "reported" reaction with low yield may still have been reported. Your yield cut-off on P should loosely mirror the Chemlex ≥ 20% definition. The literature yield is isolated yield, while Chemlex reports LC-MS area ratios, so the two scales are not the same.

**Baselines that must be beaten** (all evaluated on Chemlex):
1. **Random**: the plate hit-rate baseline. The prevalence is 32.8% on validation and should be reported for test.
2. **Substrate popularity:** score(pair) = log(1 + n_acid) + log(1 + n_amine). The counts come from a *broad* literature amide corpus (all ORDerly/USPTO amide couplings, any condition), not only P. This is the Beker-style test.
3. **Nearest neighbour to P:** for example, score = max over P pairs of min(Tanimoto_acid, Tanimoto_amine), or the mean of the top-k. Also a substrate-level variant: max similarity of the acid to P acids, plus the same for amines.
4. **Additive substrate-marginal model:** logit = a(acid) + b(amine), learned from substrate features only. If the full model does not beat this, it has learned no pair interaction, which may be acceptable. Report it either way.
5. **Simple chemistry heuristics:** amine class (aliphatic primary/secondary vs aniline vs electron-poor heteroaryl amine), α-branching or sterics, and acid α-substitution. Zhong et al. used exactly such rules to *create* negatives, so a heuristic baseline may do suspiciously well on Chemlex. Report it.
6. **Supervised ceiling:** a model trained on Chemlex train-side HATU labels, evaluated on your validation fold. This tells you how far PU is from "having negatives".

**Modeling or correcting the bias:**
- (a) **SAR propensity model.** Give a separate propensity model "popularity" features (literature counts, MW, drug-likeness, building-block vendor counts) and the classifier reactivity features. This follows Bekker et al. 2019 (https://arxiv.org/abs/1809.03207). Identifiability is weak, so treat it as an experiment, not the main model.
- (b) **Match the unlabeled pool to P's reporting process** so popularity cancels out (Section 3, U_lit).
- (c) **Include popularity as an explicit covariate** in a diagnostic model to measure how much of the score it explains, then **exclude it** from the deployed model.
- (d) **Temporal holdout of P** (train on patents before year T, test recall on later ones), as Jang et al. 2020 did.

---

## 3. Domain shift between P (literature) and U (inventory)

**The risk is real.** With U = inventory × inventory and P = literature pairs, the easiest decision rule is "is this acid or amine a literature-type molecule?". Under SCAR the P-vs-U classifier is supposed to learn reactivity. Here it can learn provenance perfectly from single substrates, because most inventory substrates never occur in P. Chemlex will only partly expose this, since Chemlex was sampled to look like Pistachio patents (Section 0).

**What the literature offers:**
- **PU under covariate shift:** Sakai T, Shimizu N. AAAI 2019, 33, 4838–4845, doi:10.1609/aaai.v33i01.33014838 (https://ojs.aaai.org/index.php/AAAI/article/view/4411). Importance-weighted uPU, with weights from density-ratio estimation (RuLSIF) between training-U and test-U.
- **Arbitrary positive shift:** Hammoudeh Z, Lowd D. *Learning from positive and unlabeled data with arbitrary positive shift.* NeurIPS 2020 (https://arxiv.org/abs/2002.10261). The labeled positives may be arbitrarily unrepresentative of target positives, as long as **the negative class-conditional stays fixed**. It needs unlabeled data from both source and target. This is the closest formal match to "literature P, inventory target".
- **AUC under positive distribution shift:** Kumagai A et al. NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/hash/3f8c7eb848ffec848f3ed2b7ca44915d-Abstract.html). Directly optimises test-distribution AUC using train PU data plus test unlabeled data.
- **PU-AUC under covariate shift:** Kumagai A et al. ICML 2025, PMLR 267:31876–31891 (https://proceedings.mlr.press/v267/kumagai25a.html). The authors say it needs **no class prior**.

These methods are research code and need a reasonably large target-unlabeled set. You have one: the 285k inventory pairs.

**Practical design (most impact first):**
1. **Build U from the same distribution as P.** Let U_lit = random acid × amine combinations of acids and amines drawn from the *literature amide corpus* (any condition), minus the P pairs. Train P vs U_lit. Single-substrate provenance no longer separates the classes, so the model must learn something about pairs or reactivity. Then *apply* it to the inventory plate. Variant: U_lit+inv, a mixture, with an indicator you can test for.
2. **Re-pairing / "decoy" negatives:** draw acids and amines of P in shuffled pairs. This removes substrate-marginal popularity entirely. The catch: the model can then only learn interactions, while most amide failures are probably substrate-level (anilines, hindered acids). Use it as a diagnostic, not the main model.
3. **Substrate-level vs pair-level modeling.**
   - With ~1k positives and both-unseen evaluation, a **factorised model** (acid score + amine score + small interaction) is likely to generalise better than a 4k-bit concatenated pair fingerprint.
   - Bagging PU at substrate level is also possible: an acid is positive if it occurs in P. That learns "literature-ness", though, so be careful.
4. **Adversarial validation (shift measurement):**
   - (i) Train a classifier: literature acids vs inventory acids (and the same for amines), on the same features as the PU model. AUC ≈ 0.5 means little shift; AUC near 1 means trouble.
   - (ii) Run the same test on Chemlex substrates vs inventory substrates, to see whether Chemlex is representative of your plate.
   - (iii) Report the PU model's score distribution by group: P, U_lit, inventory, and Chemlex positives and negatives.
5. **Leakage/shortcut checks:**
   - Correlate PU score with the adversarial "literature-ness" score on the inventory. A high correlation (e.g. Spearman > 0.7) means the model is mostly measuring provenance.
   - Report Chemlex metrics stratified by **whether each substrate is in P** (as planned), and also by **adversarial-score bins**.
   - Permutation test: shuffle which acid goes with which amine in P (keeping the marginals). If the performance drop is small, the model is substrate-only.
6. **Importance weighting** (P(inventory | x) / P(lit | x) from the adversarial classifier): use it to reweight U_lit toward the inventory. Weights explode when support barely overlaps, so clip them and report the effective sample size. If the ESS is tiny, the shift cannot be corrected and must be stated as a limitation.

---

## 4. Featurisation for acid–amine pairs (~1k positives)

**Evidence:**
- **Zhong 2025:** feature choice not reported in the main text **[unverified]**. Code and data are on GitHub/Zenodo (PMC12081921).
- **Liu, Moroz, Isayev 2023:** benchmarked four method families on literature amide data. Nothing worked well on literature, and HTE was easy (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10566507/). Its lesson: features are not the bottleneck on literature data.
- **Zhang C, Lin Q, Yang C, Kong Y, Yu Z, Liao K.** *Intermediate knowledge enhanced the performance of the amide coupling yield prediction model.* Chem. Sci. 2025, 16, 11809–11822, doi:10.1039/D5SC03364K. HTE amide set with machine-sampled substrates. Featurising the **activated-ester intermediate** improved yields, and they reported R² 0.89 on a substrate-novelty split and 0.71 on external literature, according to search snippets. I could not open the full text (HTTP 403) **[partially unverified]**.
- **Raghavan P, Haas BC, Ruos ME, Schleinitz J, Doyle AG, Reisman SE, Sigman MS, Coley CW.** *Dataset Design for Building Models of Chemical Reactivity.* ACS Cent. Sci. 2023, 9, 2196–2204, doi:10.1021/acscentsci.3c01163 (https://pmc.ncbi.nlm.nih.gov/articles/PMC10755851/).
  - Representation decides what a model can generalise to: a one-hot base cannot predict an unseen base, but a pKa descriptor can.
  - Structured (substrate-held-out) splits are needed to show extrapolation.
  - Binarising outcomes reduces the impact of noise.
  - The paper has no amide-specific guidance beyond citing descriptor-based amide studies.
- **DRFP:** Probst D, Schwaller P, Reymond J-L. *Reaction classification and yield prediction using the differential reaction fingerprint DRFP.* Digital Discovery 2022, 1, 91–97 **[not re-checked]**. Here every reaction forms the same amide bond, so DRFP essentially reduces to substrate substructures minus the shared reaction centre. Little gain over concatenated substrate fingerprints is expected **[my reasoning, untested]**.

**Recommendation, in order of effort:**
1. **Concatenated Morgan counts** (r = 2, 1024–2048 bits per substrate) feeding GBM/RF. This is the baseline, and PAYN uses ECFP too.
2. **Reaction-centre descriptors.** These are about 20–40 hand-crafted, chemically interpretable features, which is the Raghavan argument:
   - amine: N class (1°/2°, aliphatic/aryl/heteroaryl), estimated pKaH or partial charge on N, steric descriptors (number of α-substituents, buried volume if cheap);
   - acid: α-substitution, electron-withdrawing groups, ortho-substitution for aryl acids;
   - competing nucleophiles or electrophiles: free OH, second NH, unprotected groups.
   These features generalise to both-unseen pairs and are least likely to encode "literature-ness".
3. **Local-environment fingerprints** (Morgan environment rooted at the reacting N and C(=O)). Cheap, and focused on reactivity.
4. **Pretrained embeddings** (e.g. CheMeleon, MolFormer, Uni-Mol). Try them only after 1–3. With 1k positives they may *amplify* provenance shortcuts, because pretraining corpora look like literature molecules **[my reasoning]**.
5. Run every feature set through the Section 3 adversarial check. Prefer the features with the best (Chemlex AUC) − (literature-ness leakage) trade-off.

---

## 5. Evaluation design

- **Splits.** Your both-unseen design follows Zhong's "stratified" split. Additionally:
  - (a) Remove from P any pair whose *product* (or acid+amine pair) appears in Chemlex.
  - (b) Report metrics in the four strata {acid in P?} × {amine in P?}, plus by Tanimoto distance to the nearest P substrate (binned).
- **Metrics.** ROC-AUC, PR-AUC (with prevalence stated), and **top-k hit rate and enrichment factor** (hit rate@k / prevalence) at k = 24, 48 and 96.
  - Chemlex validation has 1,725 rows, so top-96 is about 5.6% of the set. That is a different operating point from a 96-well plate chosen from 285k pairs, so also report precision@5%.
  - Add **per-acid or per-amine "rank within substrate" metrics** if the plate design fixes a substrate.
- **Uncertainty.** Use a **substrate-level (cluster) bootstrap**: resample acids and amines, keep all pairs among the resampled sets, and take ≥ 1000 replicates for the CI. Pair-level bootstrap understates variance because pairs share substrates. For model comparisons, use a paired bootstrap of the metric difference on the same replicates.
- **No tuning on test.**
  - Freeze features, PU method, hyperparameters, yield cut-off and baselines on validation.
  - Write the frozen config and commit hash to the lab journal *before* the single test run.
  - Report every model that was compared on validation in the test table, not only the winner.
  - Optuna-style tuning (as in PAYN) should use P-only internal criteria (e.g. PU-AUC or recall of held-out P) or Chemlex validation, never test.
- **PU-only metrics** (for reporting on the inventory, where there are no labels): recall on held-out P, and the Jain–White–Radivojac correction of ROC/PR from PU data given a class prior. Jain S, White M, Radivojac P. *Recovering true classifier performance in positive-unlabeled learning.* AAAI 2017, 31, 2066–2072, doi:10.1609/aaai.v31i1.10937 (https://arxiv.org/abs/1702.00518). This depends on SCAR and π, so treat it as secondary to Chemlex.
- **Label-noise caveat.** Chemlex yields are uncalibrated UV ratios, and Zhong et al. show that high-aleatoric reactions replicate poorly (PMC12081921). Some of the error near the 20% boundary is irreducible. Report a sensitivity analysis at 10% and 30% thresholds.

---

## Recommended modeling plan

1. **Data hygiene.**
   - Fix the P yield cut-off: I suggest isolated yield ≥ 20–30%, with a sensitivity check.
   - Build **U_lit**: literature acids × literature amines, minus P.
   - Deduplicate P against Chemlex at the pair/product level.
2. **Shift diagnostics first.** Run adversarial validation for literature vs inventory vs Chemlex substrates. If literature vs inventory AUC is ≈ 1 on Morgan bits, prioritise descriptor features and U_lit.
3. **Baselines on Chemlex validation:** random, popularity, nearest-neighbour-to-P, additive substrate model, chemistry heuristics, and the supervised Chemlex-train ceiling.
4. **First PU model:** bagging PU (GBM, ~100 bags, P vs random U_lit subsamples of size |P|–3|P|) on concatenated Morgan plus reaction-centre descriptors. Compare U = inventory, U = U_lit and U = mix.
5. **Second PU model:** a PAYN-style spy/RN pipeline (CatBoost) for direct comparison with the group's method.
6. **Checks:**
   - Correlation of score with literature-ness.
   - P-pair permutation test.
   - Stratified metrics by in-P status and similarity.
   - Substrate bootstrap CIs.
   - π sensitivity, only if calibrating.
7. **Optional:** a SAR propensity model, or Kumagai-style PU-AUC under shift using the inventory as target-unlabeled.
8. **Freeze, then test once.** Report all compared models with CIs.

## Most important papers

1. Zhong H, Liu Y, Sun H, et al. *Towards global reaction feasibility and robustness prediction with high throughput data and Bayesian deep learning.* Nat. Commun. 2025, 16, 4522. doi:10.1038/s41467-025-59812-0
2. Boser F, Spies JC, Glorius F. *Yield Prediction of Organic Reactions in Biased Data Sets via Positive-Unlabeled Learning.* J. Am. Chem. Soc. 2026, 148, 15066–15075. doi:10.1021/jacs.6c00127 (code: github.com/GloriusGroup/PAYN)
3. Bekker J, Davis J. *Learning from positive and unlabeled data: a survey.* Mach. Learn. 2020, 109, 719–760. arXiv:1811.04820
4. Bekker J, Robberechts P, Davis J. *Beyond the Selected Completely At Random Assumption for Learning from Positive and Unlabeled Data.* ECML-PKDD 2019, LNCS, 71–85. doi:10.1007/978-3-030-46147-8_5
5. Beker W, et al. *Machine Learning May Sometimes Simply Capture Literature Popularity Trends: A Case Study of Heterocyclic Suzuki–Miyaura Coupling.* J. Am. Chem. Soc. 2022, 144, 4819–4827. doi:10.1021/jacs.1c12005
6. Strieth-Kalthoff F, et al. *Machine Learning for Chemical Reactivity: The Importance of Failed Experiments.* Angew. Chem. Int. Ed. 2022, 61, e202204647. doi:10.1002/anie.202204647
7. Hammoudeh Z, Lowd D. *Learning from Positive and Unlabeled Data with Arbitrary Positive Shift.* NeurIPS 2020. arXiv:2002.10261. Read alongside Kumagai A, et al. *AUC Maximization under Positive Distribution Shift.* NeurIPS 2024.
8. Liu Z, Moroz YS, Isayev O. *The challenge of balancing model sensitivity and robustness in predicting yields: a benchmarking study of amide coupling reactions.* Chem. Sci. 2023, 14, 10835–10846. doi:10.1039/d3sc03902a

Also: Raghavan et al., ACS Cent. Sci. 2023, doi:10.1021/acscentsci.3c01163; and Elkan & Noto, KDD 2008.

## Open questions for the chemists

1. What should count as "success" in P: isolated yield ≥ 20%? ≥ 50%? Is a literature HATU reaction reported at 15% evidence of failure or of success?
2. Are some inventory substrates known to be problematic for HATU (anilines with EWGs, α,α-disubstituted acids, free OH/NH, salts like HCl amines needing extra DIPEA)? These could become features or hard negatives.
3. How will the plate be built: free choice of 96 pairs, or a fixed acid × amine grid (e.g. 8 × 12)? A grid changes the selection problem and the metric.
4. Should the plate include diversity or exploration wells (uncertain predictions, as Zhong showed for active learning), or pure exploitation?
5. Is LC-MS conversion ≥ 20% (the Chemlex definition) the right success criterion for your downstream use, or do you need isolable product?
6. Are salt forms, protecting groups, and stoichiometry or DIPEA equivalents standardised in the inventory, so that "same condition" really holds?
