# PAYN — Positivity is All You Need: yield prediction from biased data via spy-based PU learning

- **Citation:** Florian Boser†, Jan C. Spies†, Frank Glorius*. "Yield Prediction of Organic Reactions in Biased Data Sets via Positive-Unlabeled Learning." J. Am. Chem. Soc. 2026, 148, 15066–15075. DOI:10.1021/jacs.6c00127 (CC-BY). †co-first authors.
- **Type:** method + application (our own group's paper)
- **File:** literature/yield-prediction-of-organic-reactions-in-biased-data-sets-via-positive-unlabeled-learning (1).pdf
- **TL;DR:** PAYN treats reported high-yielding reactions as positives and the unexplored reaction space as unlabeled, uses a **spy-based** PU technique under the **SCAR** assumption to extract reliable negatives, and augments a CatBoost yield regressor — recovering much of the accuracy of a fully-labeled model without any new experiments.

## Problem & contribution
Literature reaction data suffer selection bias (chemists favor familiar/successful conditions) and reporting bias (failed/low-yield reactions unpublished), yielding positive-only, high-yield-skewed datasets. Supervised yield models on such data over-estimate yields and can't flag failures. PAYN reconstructs the missing negatives via PU learning: build a balanced training set by mining **reliable negatives (RN)** from unlabeled reaction space, then train yield models on positives + RN. Open-source Python framework (github.com/GloriusGroup/PAYN).

## Method (key technical detail)
- **PU framing:** observe label s (s=1 reported, s=0 unlabeled), not true feasibility y. Unlabeled = mix of true negatives + latent positives. **SCAR assumption** (Elkan-Noto): Pr(s=1|x,y=1)=Pr(s=1|y=1)=c ⇒ Pr(s=1|x) ∝ Pr(y=1|x), so a classifier trained P-vs-U ranks reactions by true feasibility. Also invokes separability + smoothness.
- **Spy technique (Liu et al. 2002):** (1) inject a random fraction (spy_rate, ~15–20%) of known positives into U with labels hidden; (2) train CatBoost to separate remaining positives (s=1) from spy-containing U (s=0); (3) score all U; spies (true positives) get relatively high scores; (4) set threshold t_spies at the spy-tolerance quantile (e.g. 5% ⇒ 95% of spies land above threshold); (5) unlabeled points scoring below t_spies = reliable negatives; those above but unlabeled are discarded as "undecisive"; (6) positives + RN = balanced augmented set.
- **Framework:** CatBoost base (interchangeable), Optuna Bayesian hyperopt (depth, LR, iterations), ECFP / MFF / DFT features. Three hyperparameters: PU ratio (positive scarcity), spy rate, spy tolerance.
- **Benchmark design:** binary feasibility at yield >20%; simulate reporting bias by stripping negatives from fully-labeled HTE data to make "known positives," pool the rest as U. Compare Augmented vs Positives-only (lower bound) vs Fully-labeled (upper bound), identical splits.

## Assumptions & when they hold
- **SCAR** is the explicit assumption — reported positives assumed a representative random sample of true positives given feasibility. Acknowledged in-text that literature is *also* shaped by selection bias (feature-dependent), i.e. real data are SAR/SNAR, but PAYN's simulation only injects **reporting bias** (removing negatives at random ≈ SCAR-consistent removal of the negative side), sidestepping the harder SAR case.
- Chemistry argument for why PU works here: in a *random* substrate selection, P(y=1|x) is very low (feasible reactions are "rare islands"), so the unlabeled set is negative-rich and the spy classifier can separate.
- Needs **separability** (latent positives vs negatives distinguishable in feature space) and **smoothness** (similar reactions → similar outcome) for spies to be scored like latent positives.

## Datasets / benchmarks
HTE ground-truth sets: Ahneman BH C–N (3955), Stevens Ni-borylation (779, EtOH only), Perera Suzuki (5760). Applied out-of-distribution to **Neves et al. BH "JnJ HTE 2024"** (11,328 new diverse points from an 11.5M space; <29% positive — negative-rich, unlike combinatorial sets). ECFP featurization (DRFP/DFT for Neves).

## Results & claims
Augmented > Positives-only everywhere. Ahneman: MAE 11.1→6.8% (bridges 61% of gap to fully-labeled 4.0%) using only 37% of labels. Stevens 47%, Perera 33% of gap bridged. RN quality: Ahneman negative precision 91%, 66% of negatives recovered (FPR 14.2%). Neves/JnJ (using only positives + 22% of data): positives-only MAE 27.9% → augmented 12.3% (fully-labeled 7.05%), bridging 75% of gap; RN purity up to 97% negative. Spy tolerance ~0.025–0.05 and spy rate 0.2 optimal (Neves used spy rate 0.01 because it's negative-rich). Stevens RN only ~50% pure (poor separability in that feature space) yet still helps.

## Relevance to our BH C–N PU pipeline
This is the **anchor method and the direct predecessor** of the hackathon plan — same reaction class (BH C–N), same test set (Neves JnJ HTE 2024), same spy machinery and CatBoost stack. Our task reuses PAYN's framework but (a) shifts target from *yield* to *feasibility*, (b) mines positives from *real literature* (not simulated masking), and (c) treats the whole combinatorial space as unlabeled. **PAYN's limitation is exactly our project's opening:** it assumes SCAR, but real literature positives are selected feature-dependently (SAR/SNAR). Moving beyond SCAR (propensity/SAR-EM, PULSNAR per-subtype α, PUe) is the methodological novelty we can benchmark against PAYN's SCAR-spy baseline.

## Caveats / limitations / leakage risks
- **SCAR is the load-bearing assumption and is violated by real literature** (the whole point of "beyond SCAR"). The paper's own simulation removes negatives, which mimics reporting bias but not the feature-dependent selection of *which* positives get reported.
- Single global spy threshold ignores that different reaction subtypes are reported at different rates (PULSNAR's critique).
- RN quality collapses when featurization gives poor separability (Stevens) → RN carry label noise. On the real combinatorial BH space, separability of literature-positives from latent negatives is unknown.
- **Leakage discipline:** identical train/val/test splits, spies drawn only from train positives, hyperopt on a val fold — we must preserve this and additionally ensure no literature positive that is actually a JnJ test reaction leaks into training.

## Open questions for us
- What does replacing SCAR-spy with a propensity/SAR debiasing do to RN precision/recall on the *real* (not simulated) literature→JnJ pipeline?
- Can per-subtype spy thresholds (PULSNAR-style) beat the single global t_spies for chemically heterogeneous positives?
- Is the "feasible reactions are rare islands" claim (P(y=1|x) low under random selection) actually true for BH C–N over the JnJ 11.5M space, and does it justify SCAR-ish behavior in practice?

## Cited works worth chasing (DOIs; 🔒 = likely paywalled)
- Elkan & Noto 2008, KDD (SCAR). DOI:10.1145/1401890.1401920 🔒
- Bekker & Davis 2020, "PU learning: a survey," Mach. Learn. 109:719. DOI:10.1007/s10994-020-05877-5 🔒
- Liu, Lee, Yu, Li 2002, spy technique, ICML. (open)
- Neves et al. 2025, "Robust Out-of-Distribution Prediction of Buchwald–Hartwig Reactions," ChemRxiv. DOI:10.26434/chemrxiv-2025-xcr46 (open) — the JnJ HTE 2024 test set
- Ahneman, Estrada, Lin, Dreher, Doyle 2018, BH C–N ML, Science 360:186. DOI:10.1126/science.aar5169 🔒
- Nishii et al. 2025, PU learning for oxidative homocoupling reactivity, ACS Omega. DOI:10.1021/acsomega.5c05523 (open)
- Gao, Raghavan, Shprints, Coley 2025, publication bias vs reactivity via contrastive learning, JACS 147:8959. DOI:10.1021/jacs.4c15660 🔒
- Saunders & Freitas 2022/2024, evaluating PU classifiers / AutoML-PU. arXiv:2206.02423, arXiv:2401.06452 (open)
