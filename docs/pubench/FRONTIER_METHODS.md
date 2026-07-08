# Frontier PU methods — status, assumptions, and why (not yet) implemented

*Author: Jan's session (Claude). This documents the shift/bias-aware methods from the original
brief. Rather than ship one-night, possibly-incorrect implementations of research papers (which
would violate "good ML practices"), the tractable-and-correct ones are implemented and the rest are
specified here with their assumptions, relevance to our data, and an implementation sketch — so they
can be implemented properly with review.*

## Implemented (see `pubench.models` / `pubench.density_ratio` / `pubench.priors`)
| method | file | notes |
|---|---|---|
| SupervisedBaseline | `models/supervised.py` | naive lower bound (U as negative) |
| NonNegativePU (nnPU) | `models/nnpu.py` | Kiryo 2017 risk, numpy linear scorer, weighted-P capable |
| ImportanceWeightedPU | `models/iw_pu.py` | **our representative of the density-ratio-reweighting family** (covers the ShiftAware-IW idea): RuLSIF p_U/p_P weights (with random-projection to make the ratio informative in high dim) reweight the nnPU risk |
| SpyTwoStep | `models/spy.py` | Liu 2002 reliable-negative extraction (PAYN baseline) |
| BaggingPU | `models/bagging_pu.py` | Mordelet & Vert 2014; HistGB ensemble over U subsamples |
| ElkanNotoClassifier | `models/elkan_noto.py` | 2008 scaled classifier g/c (HistGB); nonlinear, prior-free |
| ULSIF / RuLSIF | `density_ratio/ulsif.py` | kernel density-ratio for the IW family |
| ElkanNotoPrior | `priors/estimators.py` | SCAR class-prior (degenerates under shift — see E1) |

## Deferred — specification + why

### ShiftAwareIWPU (Kumagai et al., AISTATS 2025)
- **Idea:** importance-weight between train and test distributions using only PU data on *both*
  sides, without assuming the shift type; estimate p_test(x)/p_train(x) from the unlabeled samples.
- **Relevance:** directly our setting (P vs target/U shift). **Largely realised by our
  `ImportanceWeightedPU`** (RuLSIF p_U/p_P reweighting). The exact Kumagai estimator (their specific
  train/test PU density-ratio objective) is the delta; implement by swapping the density-ratio step
  for their estimator and validating against the paper's synthetic benchmark.
- **Status:** the *concept* is covered; the exact estimator is future work (needs the paper).

### DETM — double exponential tilting model (Liu et al., JASA 2025)
- **Idea:** two exponential-tilt density ratios link source positives to target-positive and
  target-negative populations; fit by EM / empirical likelihood; estimates the prior internally;
  ships a **goodness-of-fit test** for SCAR/common-distribution vs distribution-shift (DD).
- **Relevance:** the most complete treatment of our *double* shift (target-pos AND target-neg differ
  from source). High value.
- **Why deferred:** the EM over latent target labels + empirical-likelihood tilt fitting is a
  substantial, error-prone implementation; correctness matters more than coverage. **Partial
  substitute shipped:** the SCAR-vs-shift *diagnostic* (P-vs-U domain-discriminator AUC, see
  EXPERIMENTS) captures the goodness-of-fit intuition (AUC ≫ 0.5 ⇒ the positive covariate
  distribution shifted, so SCAR is violated).
- **Sketch:** exponential tilt p_t(x)=p_s(x)exp(α+βᵀx) is fit by logistic regression discriminating
  the two samples (known equivalence); the two tilts + latent target labels give an EM loop
  alternating (E) posterior target labels and (M) two logistic tilts + prior update.

### PropensityIPS_DR (Schnabel 2016 / Saito 2020 style, SAR propensity)
- **Idea:** treat "reported (∈ P)" as an exposure; fit a reportability propensity e(x); inverse-
  propensity-weight (clipped) + a doubly-robust outcome correction.
- **Why deferred / caveat:** identifiability of a *feature-dependent* propensity needs a **shadow
  variable** (a feature affecting labeling but not the outcome). Our features (substrate + condition)
  are exactly the outcome features, so without distinct reportability metadata the propensity is not
  separately identifiable and the estimator **collapses to the P-vs-U classifier** (≈ ElkanNoto).
  Implement properly only once reportability metadata (e.g. publication/source/recency covariates)
  is available.

### SAR-EM (Bekker et al., "Beyond SCAR")
- Feature-dependent propensity via EM. Same shadow-variable identifiability caveat as above; deferred
  for the same reason.

### aPU / OSLS (cautionary comparisons)
- `aPU_PURR` assumes an **invariant negative** distribution; `OSLS_PULSE` assumes invariant
  class-conditionals. Our setting shifts *both* populations, so these assumptions are violated — they
  are of interest only as cautionary baselines. Implement with `requires.assumes_fixed_negatives=True`
  clearly flagged; low priority given the violation.

## Class-prior estimators (KM2 / TIcE)
- Stubbed. **Lower priority for this problem** because we *have* labeled validation data, so the
  legitimate prior is the **val base rate (0.32)** — a better estimate than any label-free PU prior.
  KM2/TIcE matter only if val labels were unavailable.
