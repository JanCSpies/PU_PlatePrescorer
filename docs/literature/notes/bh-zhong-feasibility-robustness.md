# Zhong et al. — Global Reaction Feasibility & Robustness via HTE + Bayesian Deep Learning
- **Citation:** H. Zhong, Y. Liu, H. Sun, …, T. Yu, P. Wang, X. Wang. *Towards global reaction feasibility and
  robustness prediction with high throughput data and Bayesian deep learning.* *Nat. Commun.* 2025, 16, 4522.
  DOI 10.1038/s41467-025-59812-0. (ChemLex, Shanghai — **NOT a Glorius paper**, despite an earlier guess.)
- **Type:** application + data + method
- **File:** literature/s41467-025-59812-0.pdf
- **TL;DR:** Build the largest single-reaction-type HTE feasibility dataset (**11,669 acid–amine amide couplings**,
  8,095 products) and train a **Bayesian neural network** that predicts feasibility at ~89% accuracy with calibrated
  uncertainty, using **epistemic** uncertainty for active learning (−80% data) and **aleatoric** uncertainty as a
  reaction-**robustness/reproducibility** proxy.

## Problem & contribution
Feasibility ("will this reaction work?") + robustness ("is it reproducible / scale-up-safe?") are both unsolved and
data-starved (no negatives in literature). Contribution: (1) a broad, negative-rich HTE dataset for amide coupling built
by diversity-guided down-sampling of Pistachio substrate space + **chemist-designed "expert-rule" negatives**; (2) a BNN
that is well-calibrated; (3) **uncertainty disentanglement** — epistemic → active-learning/OOD detection, aleatoric →
robustness (validated against mg-scale discovery vs kg-scale process literature).

## Dataset(s) — scope, size, conditions, feasibility definition
- **NOT Buchwald–Hartwig — amide (acid+amine) condensation.** 11,669 reactions, 8,095 target products, 200–300 µL scale,
  156 instrument-hours on an automated platform (CASL-V1.1 / MegaRobo).
- Space: **272 acids × 231 amines**, **6 coupling reagents** (HATU, TCFH, BOP, PyBrOP, PyBOP, EDCI/HOBt), **2 bases**
  (DIEA, NMI), **1 solvent** (DMF), rt, 0.5 h. All substrates single-reactive-group; all products novel vs Pistachio.
- ~5,600 of the reactions are deliberately-designed likely-**negatives** (steric/nucleophilicity expert rules).
- **Feasibility threshold = 20% yield** (Fig 5a; uncalibrated LC-MS UV ratio, per Smith et al.). (Contrast: JnJ BH uses 10%.)

## Method & features
- **Bayesian Neural Network** (2 FC layers, weights+biases as distributions); posterior via **NUTS-MCMC** (also SVI);
  composite Laplace+Gaussian prior fit by MAP. Baselines: MC-dropout, Deep-Kernel GP, deep ensembles.
- Predictive uncertainty = entropy of induced Bernoulli; **disentangled** into aleatoric (avg entropy at fixed weights)
  vs epistemic (total − aleatoric) following Depeweg et al.

## OOD / splitting strategy & metrics
- Three splits (following Xu/Roadmap 2022): **Random**, **Stratified one-reactant-unseen**, **Stratified both-reactants-unseen**
  (70/30). The stratified splits are the OOD/domain-shift tests (t-SNE shows increasing domain shift).
- **Metrics:** Accuracy, F1-macro, AUC-ROC; calibration (ECE), wrong-prediction curve.
- **Results (BNN+NUTS):** random 89.48% / F1 0.86 / AUC 0.94; one-unseen 80.84 / 0.74 / 0.87; both-unseen 72.47 / 0.65 / 0.79.
  Best-calibrated of all methods; top-30% highest-uncertainty samples cover >70% of errors. **Active learning: ~1/5 of data
  reaches full performance (−80%).** Aleatoric uncertainty separates reproducible (ICC 0.97) from irreproducible (low ICC) rxns;
  process-phase (kg) reactions have lower aleatoric uncertainty than discovery-phase (mg).

## Relevance to our pipeline (esp. leakage risk vs JnJ test set)
- **The closest prior "feasibility" benchmark to match/beat**, but on a DIFFERENT reaction (amide, not BH) → good methods
  transfer, **no substrate leakage** into JnJ BH.
- Directly comparable framings we should adopt/report:
  - **Metrics**: Accuracy / F1-macro / AUC-ROC + calibration (ECE) + wrong-prediction curve — a richer, uncertainty-aware
    scorecard than Neves' RF numbers; report both to be credible.
  - **Splits**: their one-/both-reactant-unseen stratified splits are cleaner "OOD" definitions than K-means clusters and
    map naturally onto BH (one-substrate-unseen / both-unseen) — consider adding these alongside Neves' folds.
  - **PU connection**: they manufacture negatives from **expert rules**; we manufacture them from **PU (spy/implicit)** —
    a cleaner, learnable alternative. Their "surprising" reactions (expert-predicted-negative but high-yield) are exactly the
    false-negatives a PU model avoids by not hard-coding chemistry.
  - **Robustness idea (novel bonus)**: aleatoric uncertainty ≈ reproducibility. If our PU model is Bayesian, we could add a
    robustness readout to our HTE-plate recommendation — a differentiator for the hackathon.
- Data at Zenodo 10.5281/zenodo.12920294; code github.com/Chemlex-AI/bayesian-reactivity-prediction (usable baseline).

## Open questions for us
- Do we want a Bayesian PU head (calibrated uncertainty) so we can rank plate wells by confidence like Neves/Zhong?
- Threshold harmonization: Zhong 20% vs JnJ 10% — keep 10% for our BH task; note the choice explicitly.
- Could their expert-rule negatives be a sanity check on our PU-derived negatives for BH amines?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Raghavan et al. Suzuki yield / synthetic accessibility (AbbVie 15-yr). *JACS* 2024, 146, 15070. DOI 10.1021/jacs.4c00098 🔒
- Schleinitz et al. NiCOlit literature yield. *JACS* 2022, 144, 14722. DOI 10.1021/jacs.2c05041 🔒
- Xu et al. Roadmap to pharma reactivity models (splits). ChemRxiv 2022, DOI 10.26434/chemrxiv-2022-x694w 🔓
- Haas, Sigman et al. amide-bond efficiency regression. *PNAS* 2022, 119, e2118451119 🔓
- Voinarovska et al. "When yield prediction doesn't yield prediction." *JCIM* 2024, 64, 42. DOI 10.1021/acs.jcim.3c... 🔓
