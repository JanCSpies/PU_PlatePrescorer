# PULSNAR / PULSCAR: class-proportion estimation without the SCAR assumption

- **Citation:** Praveen Kumar, Christophe G. Lambert. "Positive Unlabeled Learning Selected Not At Random (PULSNAR): class proportion estimation without the selected completely at random assumption." PeerJ Comput. Sci. 10:e2451 (2024). DOI:10.7717/peerj-cs.2451. (preprint arXiv:2303.08269)
- **Type:** method (class-prior estimation + calibration + classification)
- **File:** literature/Beyond_SCAR/peerj-cs-2451-2.pdf
- **TL;DR:** Two algorithms — PULSCAR estimates the positive fraction α in the unlabeled set under SCAR via a beta-kernel density objective, and PULSNAR extends it to SNAR by clustering labeled positives into subtypes, running PULSCAR per cluster, and summing the per-subtype α's.

## Problem & contribution
Focus is **α = fraction of positives among the unlabeled** (not the class prior; class prior is derivable from α), plus **probability calibration** and downstream classification. Motivated by SNAR settings (severe disease over-diagnosed, confirmed fraud, etc.) where SCAR α-estimators (KM1/KM2, TIcE, DEDPUL) are biased. Contributions: (1) PULSCAR α-estimator; (2) PULSNAR divide-and-conquer for SNAR; (3) a label-flipping calibration algorithm for PU probabilities; all fast and scalable (n>1M, 150k+ features).

## Method (key technical detail)
- **PULSCAR:** Train a probabilistic classifier (XGBoost, 5-fold CV) with labeled positives=1, unlabeled=0. Get class-1 scores; estimate score densities Dp (positives) and Du (unlabeled) with a **beta kernel** (well-behaved on [0,1], no boundary bias, no negative-density artifacts of Gaussian KDE). Mixture identity Du = αDp + (1−α)Dn with Dn≥0 ⇒ αDp ≤ Du pointwise. α_max = min_i(Du_i/Dp_i). Objective f(α)=log(|min(Du−αDp)|+ε); find α at the point of **maximum change in slope** (finite differences, step 1e-4), taking the first sharp change (closest to 0) to keep Du−αDp ≥ 0. Bandwidth optimized by differential evolution to match a histogram-derived density.
- **PULSNAR (SNAR):** (1) XGBoost on all P∪U to get feature gain scores; (2) scale labeled-positive features by their gains; (3) cluster labeled positives with a GMM, #clusters chosen by BIC "knee-point detection" (iterate 1..25); (4) for each cluster c, run PULSCAR on {positives in c} ∪ {all unlabeled} → α_c; (5) α = Σ α_c. Rationale: under SNAR, subclass proportions differ between P and U, so a single α can't scale Dp; within a homogeneous cluster, selection bias is ~uniform ⇒ locally SCAR.
- **Calibration (Algorithm 4):** once α known, probabilistically flip α|U| unlabeled labels 0→1 to match the positive-score PDF across 100 bins, then fit isotonic/sigmoid regression → calibrated probabilities. For SNAR, combine per-cluster probabilities via p = 1 − Π_c(1−p_c).
- **Classification boost:** relabel top α|U| unlabeled as positive, rest as negative, retrain.

## Assumptions & when they hold
- **PULSCAR assumes SCAR + separability** and that unlabeled positives have **support** in the labeled positives (Pr(x|y=1,s=1)=Pr(x|y=1,s=0)). Gives a trustworthy *upper bound* on α (empirically slightly over-estimates, more so at small α<5%).
- **PULSNAR (SNAR):** each positive **subclass in U must have support in the labeled set**; if a subtype is entirely missing from labeled P, α is under-estimated. Selection bias assumed uniform within a cluster.
- Requires knowing (or guessing) whether data are SCAR vs SNAR to pick the algorithm — a stated limitation; PULSNAR on true-SCAR data can over-estimate α (near-identical clusters double-counted).

## Datasets / benchmarks
Synthetic (sklearn make_classification, class_sep=0.3 hard) with α∈{1,5,10,20,30,40,50}%; SCAR real: KDD Cup 2004 physics, CDC Diabetes, MAGIC, Electrical grid, Wilt, Mice protein. SNAR real: Anuran calls, Dry bean, Room occupancy, Smartphone HAR, Letter recognition, Statlog Shuttle (multi-subclass positives). Baselines KM1, KM2, TIcE, DEDPUL.

## Results & claims
On SCAR, PULSCAR ≈ DEDPUL and both beat KM/TIcE (DEDPUL over-estimates at 1%). On SNAR, **only PULSNAR** tracks true α; all SCAR-based methods (including PULSCAR and DEDPUL) over/under-estimate — SCAR methods degrade as #positive-subclasses grows. PULSNAR far faster than KM/TIcE/DEDPUL on large data (KM often fails to run). Calibration curves closer to y=x for SCAR than SNAR; both improve with larger α.

## Relevance to our BH C–N PU pipeline
Two direct uses. (1) **α-estimation as an independent benchmark/anchor:** we need P(feasible) over our unlabeled combinatorial BH space; PULSCAR gives a fast SCAR upper bound and PULSNAR a SNAR-aware estimate — cross-check against SAR-EM. (2) **Conceptual match to our bias:** literature BH positives are a *mixture of reaction subtypes* (different amine/aryl-halide/ligand families) that are reported at very different rates — this is precisely PULSNAR's "subclass proportions differ between P and U." Clustering our known-positive reactions by chemistry-family (or by gain-weighted features) and estimating α per cluster is a principled beyond-SCAR alternative to PAYN's single global spy threshold. Their calibration algorithm could also give us calibrated per-reaction feasibility probabilities on the JnJ test set.

## Caveats / limitations / leakage risks
- Must pre-decide SCAR vs SNAR; BIC knee always returns ≥2 clusters, so running PULSNAR on genuinely-SCAR data risks α over-estimation.
- SNAR upper-bound guarantee is lost (only PULSCAR/SCAR case is a clean upper bound).
- Support assumption is the leakage-adjacent risk: if the JnJ test reactions include a positive *subtype not present* in our mined literature positives, α (hence recovered feasibility) is under-estimated for that region — an OOD failure mode exactly relevant to "Robust OOD prediction of BH" (Neves).
- Density-estimation error inflates α at small α (our regime if feasible reactions are rare) — watch the <5% region.

## Open questions for us
- Do chemistry-family clusters of our literature positives yield stable per-cluster α that sum to a sensible global feasibility rate?
- Can PULSNAR's per-cluster α replace a single SCAR label-frequency c inside a PAYN-style reliable-negative extractor (i.e. cluster-specific spy thresholds)?
- Does PULSCAR's beta-kernel density separation diagnose whether our featurization even *supports* PU learning (separability) before we commit to a model?

## Cited works worth chasing (DOIs; 🔒 = likely paywalled)
- Ivanov 2020, DEDPUL (EM on densities, uncalibrated probs). arXiv:1902.06965 (open)
- Ramaswamy, Scott, Tewari 2016, KM1/KM2. arXiv:1603.02501 (open)
- Bekker & Davis 2018, TIcE. (open)
- Elkan & Noto 2008, KDD. DOI:10.1145/1401890.1401920 🔒
- Garg et al. 2021, "Mixture proportion estimation and PU classification" (scalability of KM/TIcE), NeurIPS. (open)
- Bekker, Robberechts, Davis 2019, SAR propensity. DOI:10.1007/978-3-030-46147-8_5 🔒 (their SNAR complement)
- Chen 1999, beta kernel density estimation. 🔒
