# Learning from PU Data under the Selected At Random Assumption (SAR-EM, LIDTA 2018)

- **Citation:** Jessa Bekker, Jesse Davis. "Learning from Positive and Unlabeled Data under the Selected At Random Assumption." Proc. MLR (LIDTA 2018) 94:8–22. (companion/earlier version of arXiv:1808.08755)
- **Type:** method (algorithm-focused)
- **File:** literature/Beyond_SCAR/bekker18a.pdf
- **TL;DR:** The original LIDTA paper that proposes the SAR assumption and the SAR-EM algorithm — an EM method that jointly learns a classifier and a propensity model, with a "local-certainty decay" trick to avoid the degenerate e(x)≡1 solution; a special case (x_e=∅) also gives a strong SCAR class-prior estimator.

## Problem & contribution
Precursor to the ECML 2019 paper; more algorithm/pseudocode oriented. Frames three established PU assumption families: (1) unlabeled = negative (naive); (2) separability (two-step: find reliable negatives, then supervised); (3) SCAR (label-frequency scaling). Argues all are too strong, proposes SAR (labeling probability = propensity score e(x), a function of attributes). Contributions: formulate SAR; give SAR-EM; show the SCAR special case of SAR-EM beats state-of-the-art class-prior estimators; show that wrongly assuming SCAR hurts; show SAR-EM recovers propensity functions.

## Method (key technical detail — Algorithm 1 + 2)
- **Data model:** (x,y,s) ~ Pr(x)Pr(y|x,θ)Pr(s|x_e,y,φ); s=1⇒y=1. Fit θ (classifier), φ (propensity) by maximizing log-likelihood of observed (x,s) via EM.
- **E-step:** ŷ = Pr(y=1|x,s). For s=1, ŷ=1. For s=0, ŷ = f(x)(1−e(x_e)) / (1 − f(x)e(x_e)) — but with a **decay** ŝ = d·e(x_e[s=0]), d=0.9, deliberately shrinking predicted propensity to push toward local certainty (more positively-inclined classifier).
- **M-step:** refit f on a tripled dataset (labeled positives as y=1 weight 1; each unlabeled entered twice, y=1 weight ŷ and y=0 weight 1−ŷ); refit e on (positives s=1 vs unlabeled s=0) weighted by ŷ.
- **Local-certainty problem:** pure EM has a degenerate optimum — e≡1 plus a classifier that just predicts Pr(s=1|x) explains the data perfectly but is useless. The decay of predicted propensities (line 6) biases the search toward classifiers that attain Pr(y=1|x)=1 somewhere (local certainty), the standard PU anchor.
- **Init (Algorithm 2):** fit f to predict s; set c=1/max(f̂) (Elkan-Noto estimator 3); reweight to predict y; init e to return constant c.
- **Convergence:** measured by slope of propensity predictions over last n=10 iters < ε=1e-4.
- **Special cases:** x_e=∅ ⇒ SCAR label-frequency estimator; known e ⇒ classifier-only optimization (more stable than the Pr(s=1|x)/c scaling because it downweights unexpectedly-labeled low-probability positives).

## Assumptions & when they hold
- **Propensity attributes x_e known and lower-dimensional** (identifiability, same as the 2019 paper).
- **Local certainty in every propensity stratum** (some region where Pr(y=1|x)=1). Relaxable if f and e are *smooth* over x_e (similarity of the classifier across propensity configurations). This is the anchor that makes the class prior identifiable.
- Base classifier should give **calibrated probabilities** (they use logistic regression precisely for this).

## Datasets / benchmarks
Class-prior estimation on Breast Cancer, Mushroom, Adult, IJCNN, Cover Type, and six 20-Newsgroups pairings; c∈{0.1,…,0.9}. SAR experiments simulate propensity from 1 or 3 binary word-presence attributes on 20NG. Baselines: EN, PE, pen-L1, KM1, KM2, AlphaMax, AlphaMax-N, TIcE.

## Results & claims
- SCAR class-prior estimation: SAR-EM (x_e=∅) has the **best average rank and lowest |α̂−α|** (0.08) vs KM2 (0.10), TIcE (0.09), AlphaMax (0.12), EN (0.27).
- SAR: as data deviate from SCAR (Δc grows), assuming SCAR degrades F1 sharply; SAR-EM with *unknown* e performs almost as well as with *known* e, and clearly beats multi-SCAR and SCAR. Unbalanced classes are harder in PU; knowing the true prior helps most there.

## Relevance to our BH C–N PU pipeline
Same core message as the 2019 paper but gives us the **concrete algorithm + pseudocode** to reimplement SAR-EM as a beyond-SCAR benchmark, and shows the SCAR special case is itself a strong **class-prior (α) estimator** — useful because knowing α = P(feasible) in our unlabeled combinatorial space calibrates every downstream PU model. The decay/local-certainty trick is a practical knob analogous to PAYN's spy tolerance. Reimplementing SAR-EM with x_e = {data-source, substrate-popularity, year} gives us a principled beyond-SCAR arm to benchmark against PAYN's SCAR-spy.

## Caveats / limitations / leakage risks
- Logistic regression base; with gradient-boosted trees we must calibrate for the EM E-step (they flag this as future work). CatBoost from PAYN would need Platt/isotonic calibration to slot in.
- Local certainty is a strong requirement in chemistry: is there a region of reaction space where feasibility is *certain*? Probably only for canonical, heavily-reported couplings — which is exactly where reporting bias is worst.
- **Leakage:** never let test-set membership or the yield-derived label enter x_e or the propensity model.

## Open questions for us
- Does SAR-EM's α estimate for our full combinatorial BH space agree with PULSCAR/DEDPUL estimates? Cross-checking α across methods is a cheap sanity benchmark.
- Can decay parameter d be tuned to control the precision/recall of reliable negatives the way PAYN's spy tolerance does?

## Cited works worth chasing (DOIs; 🔒 = likely paywalled)
- Elkan & Noto 2008, KDD (SCAR + label-frequency estimators). DOI:10.1145/1401890.1401920 🔒
- Jain, White, Radivojac 2016, AlphaMax, NIPS. (open)
- Ramaswamy, Scott, Tewari 2016, KM1/KM2. arXiv:1603.02501 (open)
- Bekker & Davis 2018, TIcE, AAAI. (open)
- Niculescu-Mizil & Caruana 2005, "Predicting good probabilities," ICML (why calibration matters for the E-step). (open)
