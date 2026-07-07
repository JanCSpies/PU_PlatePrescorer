# PU-bagging (Mordelet–Vert) for phenol oxidative-homocoupling reactivity

- **Citation:** Takafumi Nishii, Kaname Ichizawa, Haruka Nagano, Hiroya Mukai, Daimon Sakaguchi, Hiroaki Gotoh*. "Predicting Substrate Reactivity in Oxidative Homocoupling of Phenols Using Positive and Unlabeled Machine Learning." ACS Omega 2025, 10, 49805–49815. DOI:10.1021/acsomega.5c05523 (CC-BY-NC-ND). Code: github.com/poclab-web/pulearn_phenol
- **Type:** application (of an off-the-shelf PU-bagging method to a new chemistry task)
- **File:** literature/predicting-substrate-reactivity-in-oxidative-homocoupling-of-phenols-using-positive-and-unlabeled-machine-learning.pdf (+ SI ao5c05523_si_001 (2).pdf)
- **TL;DR:** They rank phenol substrates for oxidative homocoupling with the Mordelet–Vert **transductive bagging PU classifier** (an ensemble of ~100 random forests, each trained on all positives vs. a small random negative-labeled subsample of the unlabeled set, aggregated by out-of-bag positive-vote fraction), showing it beats a positive/negative RF on ~14 positives against ~4,600 unlabeled — a textbook "combine many weak predictors to dodge class imbalance" recipe directly relevant to next-PAYN.

## Problem & contribution
Literature reports which substrates *work* (positives) but rarely which *fail* (negatives), and untested substrate space is huge. For the oxidative homocoupling of phenols (biphenol synthesis; single-substrate reactivity, atom-economical vs. Suzuki/Negishi) they want to rank which of 4,665 phenols will dimerize under a given condition, from only a handful of confirmed positives and no reliable negatives. Contribution is not a new algorithm but the **first application of PU learning to organic-synthesis reactivity**, using an existing PU-bagging ensemble, and a head-to-head demonstration that it beats a supervised positive/negative (PN) model on a real held-out experiment.

## Method — the ensemble/PU-bagging algorithm
This is the **Mordelet & Vert (2014) "bagging SVM"** scheme, implemented via the open `pulearn` package (its `BaggingPuClassifier`), but with a **random forest** as the base learner rather than an SVM. Precisely:

- **Inputs:** positives P, unlabeled U, #bagging iterations T, subsample size K.
- **Each of T iterations:** draw K samples from U at random and label them **negative** → U_n (subsample of the unlabeled set, drawn per-bag; the rest of U is out-of-bag). Train a **random forest** on P (positive) + U_n (negative), with the two classes **balanced by class weighting** (so K need not equal |P|). Apply the fitted RF to the **remaining (out-of-bag) unlabeled points** in U.
- **Aggregation:** repeat T times; each unlabeled point's score = **fraction of the bags (in which it was out-of-bag) that predicted it positive** = "dimer synthesis potential score" (DPS). This is out-of-bag PU-bagging scoring.
- **Base learner:** random forest (class-weight balanced). **#weak models:** T = 100 (tuned over {100,200,300,400,500}; T had negligible effect, so they used the cheapest, T=100).
- **U-subsampling to balance P:** K tuned over 1–20 × N_P by grid search. Optimum **K = 2·N_P** for the 28 descriptors (MCC 0.69) and **K = 10·N_P** for ECFPs (MCC 0.59). As K grows, TPR falls and TNR rises (bigger negative bags → more conservative). So each weak model sees a near-balanced (P vs. ~2–10·N_P) problem instead of the raw 14-vs-4,600 imbalance, and class weighting balances it further.
- **Prior/threshold:** **no class prior is estimated.** The decision boundary is a **fixed DPS = 0.5** (majority vote). Hyperparameters T, K are chosen to maximize **MCC** in a leave-one-out-style check: treat each positive in turn as unlabeled and see if it is recovered as positive (→ TPR), and check that known (unlabeled) negatives score negative (→ TNR). Note: negatives are used **only for T/K tuning**, never in training.

## Featurization, data, labeled/unlabeled ratio
- **Two descriptor sets** compared: (1) **28 physically-motivated descriptors** — heavy DFT pipeline (ETKDG 10,000 confs → MMFF94 → PM6 → B3LYP/6-31G(d) freq; then HOMO at M06-2X/6-311G(d,p), Mulliken charges at MP2/6-31G(d) on defined ortho/meta/para ring atoms), Sterimol L/B1/B5 per substituent (Paton–Jackson), plus MolLogP, MolWt, heteroatom counts (RDKit). (2) **ECFP** radius 2, 2,048 bits, constant columns dropped.
- **Data:** 4,665 phenols (SG = 199 literature-synthesized, BG = 97 derived from bioactive biphenols, RG = 4,494 catalogue reagents). Five conditions. Positives per condition are tiny: K₃[Fe(CN)₆] N_P = 14 (44 tested → 30 negatives), T(p-OMe)PPCo 9 (+2 neg), CuCl 19, MesAcr⁺BF₄⁻ 20, Diacetyl 12 (+1 neg).
- **Labeled/unlabeled ratio ≈ 14/4,665 ≈ 0.3 % positive** — i.e. exactly the "small-P, large-U" regime, and negatives are essentially absent.

## Validation & results
- **Tuning metric:** MCC from LOO-style TPR (on positives) + TNR (on the few known negatives).
- **Held-out test:** 30 *new* substrates drawn from the unlabeled set (bioactive-derived or expensive), then experimentally run under K₃[Fe(CN)₆]. Prediction–experiment agreement **83.3–86.7 %**. On this test, **PU beats PN**: 28-desc MCC 0.65 (PU) vs 0.50 (PN); ECFP MCC 0.75 (PU) vs 0.60 (PN); PU TPR 0.89 (8/9) vs PN 0.56 (5/9). PN over-predicts positives (771 vs PU's 205 across U) yet is less accurate — the PU ensemble is more conservative and more precise.
- **Transfer to 4 literature conditions:** TPR 0.85–1.0 (28 descriptors generally ≥ ECFP; ECFP misses electronic/positional effects like para-CN, ortho-Ac deactivation).
- Headline: negatives are not needed to train; the ensemble recovers latent positives that sit within the existing positive region.

## Relevance to next-PAYN (large-U/small-P; avoiding imbalance)
This paper is the **cleanest published instance of the design pattern the team flagged** — "avoid the class-imbalance problem by combining many weak predictors." Mechanism: (a) each bag subsamples U down to K ≈ 2·N_P and further class-weights, so **no single model ever faces the raw imbalance**; (b) averaging T out-of-bag votes reduces the variance that a tiny P set otherwise injects; (c) it needs **no class prior** and no reliable-negative commitment. Contrast with **PAYN** (single CatBoost + spy-based SCAR reliable-negative extraction, then one balanced retrain): PAYN commits *once* to a hard RN set, so a latent positive wrongly mined as a negative poisons the whole retrain; PU-bagging instead **dilutes** any latent positive that lands in a bag's negative subsample — an occasional mislabel is out-voted across bags. That robustness-to-negative-contamination is the ensemble's key advantage in our large-U/small-P setting. The trade-off: it is still implicitly **SCAR-flavoured** (a random U subsample is assumed to be representative negatives), it produces an **uncalibrated** vote with an arbitrary 0.5 cutoff, and it estimates **no α**, so it does not by itself address the SAR/SNAR selection bias that is our project's actual novelty.

## Concrete recommendation for our BH C–N pipeline
**Yes — add a PU-bagging arm** to the benchmark alongside spy-CatBoost (PAYN). It is cheap to implement (`pulearn.BaggingPuClassifier`) and gives an ensemble baseline that directly tests the "many weak predictors" hypothesis. Precise recipe I'd code:

- **Base learner:** keep **stack parity with PAYN — CatBoost/HistGBDT** as one variant, and a fast **RandomForest/ExtraTrees** as the paper's variant. RF is cheap enough to bag hundreds of times; use it for the sweep, CatBoost for the final comparison.
- **Reaction (not substrate) featurization:** BH is a *multi-component* coupling, so featurize the **whole reaction** (DRFP / concatenated ECFP of aryl-halide + amine + ligand + base, per PAYN), **not** a single substrate. The authors explicitly flag "two-or-more-substrate reactions like cross-couplings" as untested — treat that as an open risk, not a given.
- **Subsample scheme:** per bag draw K unlabeled **without replacement** and label them negative; sweep **K ∈ {1, 2, 3}·N_P** (their optimum was 2·N_P). Because our U is ~10⁶–10⁷ and P is moderate (hundreds–thousands of literature positives), the Mordelet–Vert P≪U regime holds; but N_P is large enough that K = 2·N_P bags become big — cap K or also subsample P per bag to keep the learners "weak."
- **#estimators:** T = 200–500 (larger than their 100) so every unlabeled point is scored out-of-bag many times over a huge U; verify per-point OOB coverage.
- **Aggregation:** average **predicted positive probabilities** (softer, better-calibrated) rather than hard votes.
- **Threshold / prior:** do **not** ship 0.5. Set the operating point from a **PULSCAR/PULSNAR α estimate** or a held-out labelled HTE fold (e.g. Ahneman/Cernak negatives) — never the JnJ test set. Report ranking metrics (AUPRC, precision@k) since well-plate design is a ranking task.

**Pros vs spy-CatBoost:** robust to latent-positive contamination of negatives; auto-balanced bags dodge imbalance without SMOTE/weight hacks; low-variance smooth score ideal for substrate/well-plate ranking.
**Cons/pitfalls:** (1) **cost** — T×folds models, and each bag's OOB pass over a multi-million U is expensive → subsample U for scoring or restrict the unlabeled pool. (2) Still **SCAR-ish** — random-U-as-negative injects the same selection bias PAYN's SCAR does; it does *not* debias SAR/SNAR (that's what SAR-EM / PULSNAR are for), so it's a strong *baseline*, not the beyond-SCAR novelty. (3) **No calibration / no prior** — bolt on α-estimation. (4) **Leakage discipline:** any JnJ test reaction (and any literature positive equal to one) must be excluded from U entirely; positives must never be scored inside their own training bag; draw bags only from train-fold U.

## Cited works worth chasing (DOIs; 🔒 = likely paywalled)
- **Mordelet & Vert 2014**, "A bagging SVM to learn from positive and unlabeled examples," Pattern Recognit. Lett. 37:201–209. DOI:10.1016/j.patrec.2013.06.010 🔒 — *the* core algorithm to reimplement.
- **pulearn** package (BaggingPuClassifier): github.com/pulearn/pulearn (open) — reference implementation.
- Mordelet & Vert 2011, ProDiGe (multitask bagging PU), BMC Bioinformatics 12:389. DOI:10.1186/1471-2105-12-389 (open).
- Frey et al. 2019, MXene synthesizability via PU-bagging, ACS Nano 13:3031. DOI:10.1021/acsnano.8b08014 🔒 — same ensemble, materials.
- Jang et al. 2020, crystal synthesizability (partially supervised/PU), JACS 142:18836. DOI:10.1021/jacs.0c07384 🔒.
- Sakai, du Plessis, Niu, Sugiyama 2016, PNU learning (uses negatives in training), arXiv:1605.06955 (open) — authors' own "future work" comparison.
- Chicco & Jurman 2020, MCC vs F1/accuracy on imbalanced data, BMC Genomics 21:6. DOI:10.1186/s12864-019-6413-7 (open) — justifies MCC as the tuning metric.
- Bekker & Davis 2020, PU-learning survey, Mach. Learn. 109:719. DOI:10.1007/s10994-020-05877-5 🔒.
- Rana, Pflüger, Hölter, Tan, Glorius 2024, standardizing substrate selection, ACS Cent. Sci. 10:899. DOI:10.1021/acscentsci.3c01638 (open) — Glorius-group generality context for substrate ranking.
