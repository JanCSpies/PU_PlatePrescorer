# Beyond the SCAR Assumption for Learning from PU Data (SAR / propensity-weighted ERM + SAR-EM)

- **Citation:** Jessa Bekker, Pieter Robberechts, Jesse Davis. "Beyond the Selected Completely At Random Assumption for Learning from Positive and Unlabeled Data." ECML-PKDD 2019 (arXiv:1809.03207v4, 28 Jun 2019). DOI: 10.1007/978-3-030-46147-8_5
- **Type:** method (theory + algorithm)
- **File:** literature/Beyond_SCAR/1809.03207v4.pdf
- **TL;DR:** Introduces the SCAR/SAR/SNAR taxonomy of PU labeling mechanisms and shows that if the labeling probability is an instance-dependent *propensity score* e(x)=Pr(s=1|y=1,x), you can debias PU learning by propensity-weighting the risk (unbiased ERM) and learn e(x) jointly with the classifier via EM.

## Problem & contribution
Standard PU learning assumes SCAR: labeled positives are a uniform random subset of all positives, so Pr(s=1|y=1,x)=c is constant and Pr(y=1|x)=Pr(s=1|x)/c. Real data violate this (labels are collected preferentially — easier/more-obvious/more-severe positives get labeled). Contributions: (1) a taxonomy of labeling mechanisms borrowed from the missing-data literature — SCAR↔MCAR, SAR↔MAR, SNAR↔MNAR; (2) a propensity-weighted risk estimator that is provably unbiased for the true supervised risk given true e(x); (3) generalization bounds (Hoeffding-based); (4) analysis of the bias when e(x) is estimated; (5) a practical EM method (SAR-EM) that jointly fits classifier f and propensity model e when e is unknown but depends on a *known subset* of attributes.

## Method (key technical detail)
- **Propensity score:** e(x)=Pr(s=1|y=1,x). Unlike causal-inference propensity, it is conditioned on y=1, and it is exactly 0 for negatives (you never label a negative), so classic inverse-propensity weighting (IPW) cannot be applied directly.
- **Propensity-weighted estimator (their fix for zero-propensity negatives):** for each labeled example (weight 1/e_i as positive) also add a *negative-weighted* pseudo-negative (weight 1 − 1/e_i as class-0). Formally R̂(ŷ|e,s) = (1/n) Σ [ s_i ( (1/e_i)δ1(ŷ_i) + (1−1/e_i)δ0(ŷ_i) ) + (1−s_i)δ0(ŷ_i) ]. Expectation equals the true supervised risk R(ŷ|y) ⇒ unbiased. Works for MAE/MSE/log-loss.
- **Bounds:** |R̂ − R| ≤ sqrt(δ_max² ln(2/η)/(2n)); ERM generalization bound adds ln|H| term.
- **Bias with estimated ê:** bias = (1/n)Σ y_i(1 − e_i/ê_i)(δ1−δ0). Key takeaway: propensity only needs to be accurate *on positives*; underestimating e inflates the predicted positive rate.
- **They warn against "expected risk":** minimizing E_y[R] is a *biased* surrogate (degenerate — predicting all-positive gives 0 expected risk). Use the propensity-weighted risk instead.
- **SAR-EM (unknown e):** assume e depends only on propensity attributes x_e ⊂ x (fewer than the classifier uses — identifiability requirement). EM: E-step computes ŷ_i = Pr(y=1|s,x) = s + (1−s) f(x)(1−e(x))/(1−f(x)e(x)); M-step fits f by weighted log-loss (each unlabeled point entered twice, as pos weight ŷ and neg weight 1−ŷ) and fits e on labels s weighted by ŷ. Init: f fit treating unlabeled as negative; e init to constant c.

## Assumptions & when they hold
- **Propensity depends only on x_e (a known, lower-dimensional subset).** Needed for identifiability: otherwise a missing label is unattributable between "low class prob" vs "low labeling prob." Realistic when the labeler didn't have all features, can't interpret some, or uses only known class-correlated features.
- **Still needs a class-prior identifiability condition** (e.g. local certainty / irreducibility / a positive subdomain) in *each* propensity stratum — SAR does not remove the need for the standard PU anchor assumption; it generalizes it.
- **Smoothness** of f and e over x_e helps when strata assumptions are locally violated.

## Datasets / benchmarks
8 UCI/text/tabular sets (20 Newsgroups, Adult, Breast Cancer, Cover Type, Diabetes, Image Segmentation, Mushroom, Splice). SAR labelings synthesized by clustering into 5 groups and assigning 2 or 4 artificial binary propensity attributes with e∈[0.2,0.8]. Baselines: SCAR class-prior methods KM2, TIcE; Naive (unlabeled=negative); Supervised upper bound. Logistic regression base learner.

## Results & claims
SAR-EM (jointly learning e and f) significantly beats SCAR methods (KM2, TIcE) and Naive on ROC-AUC and MSE (Friedman p<0.001, Nemenyi p<0.01) on SAR data. Propensity scores recovered accurately (MSE_e <0.1 except Mushroom). No. of propensity attributes (2 vs 4) didn't materially change performance. Notably Naive sometimes beats SCAR methods (SCAR methods pick a wrong subspace to hit the target positive ratio).

## Relevance to our BH C–N PU pipeline
This is the theoretical spine of "beyond SCAR." Our literature reactions are labeled because they were *reported*, and reporting is emphatically feature-dependent (novelty, high yield, popular substrate classes, common ligands) → SAR/SNAR, not SCAR. Two concrete uses: (1) If we can define **propensity attributes** for "probability a feasible reaction gets reported" — e.g. substrate commercial availability, literature popularity of the aryl-halide/amine class, publication year, ligand commonness — we can either propensity-weight the PU risk (unbiased ERM) or run SAR-EM to jointly de-bias. (2) Even without trusting the exact e, the estimator shows *why* SCAR (PAYN's assumption) systematically over/under-estimates the positive rate when reporting is biased, motivating a benchmark axis: SCAR-spy (PAYN-style) vs SAR-propensity.

## Caveats / limitations / leakage risks
- Requires choosing x_e that is (a) predictive of *labeling/reporting* but (b) lower-dimensional than the classifier's features and ideally not fully redundant with reactivity features — hard in chemistry where "reported-ness" and "reactivity" are entangled (high-yield → reported). If e(x) and f(x) share the same drivers, identifiability breaks — this is exactly the SNAR danger.
- Base learner is logistic regression with well-calibrated probabilities; with CatBoost/XGBoost we'd need probability calibration for the EM E-step to be sound.
- **Leakage risk:** propensity attributes derived from the *test* well-plate's identity (e.g. year, source) must not leak the label. Fit e only on train positives + train unlabeled; never use JnJ-HTE membership as a propensity feature.

## Open questions for us
- What is a defensible, chemically meaningful set of propensity attributes for "reported feasibility" that is *not* the reactivity signal itself?
- Does propensity-weighted ERM beat PAYN's spy-thresholding on a controlled SAR simulation of literature bias (not just SCAR masking)?
- Can we stratify by data source (Reaxys vs each HTE campaign) as a discrete propensity variable (SAR-as-multi-SCAR) to get a first-order debiasing cheaply?

## Cited works worth chasing (DOIs; 🔒 = likely paywalled)
- Elkan & Noto 2008, "Learning classifiers from only positive and unlabeled data," KDD. DOI:10.1145/1401890.1401920 🔒 (the SCAR foundation)
- Bekker & Davis 2020, "Learning from PU data: a survey," Mach. Learn. 109:719. DOI:10.1007/s10994-020-05877-5 🔒
- Kiryo, Niu, du Plessis, Sugiyama 2017, nnPU, NeurIPS. arXiv:1703.00593 (open)
- du Plessis, Niu, Sugiyama 2015, "Class-prior estimation for PU," Mach. Learn. DOI:10.1007/s10994-016-5604-6 🔒
- Ramaswamy, Scott, Tewari 2016, KM1/KM2 mixture proportion, ICML. arXiv:1603.02501 (open)
- Bekker & Davis 2018, TIcE (class prior via decision trees), AAAI. (open)
- He, Liu, Webb, Tao 2018, instance-dependent PU via probabilistic gap (a special case of SAR). arXiv:1808.02180 (open)
- Schnabel et al. 2016, "Recommendations as treatments," ICML (propensity/IPW debiasing). (open)
