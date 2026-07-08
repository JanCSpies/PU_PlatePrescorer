# `pubench` — experiment log

*Running log of every benchmark run: config, result, interpretation. Newest at the bottom.
Author: Jan's session (Claude). Metric of record: **test ROC-AUC** (held-out JnJ HTE 2024,
substrate-disjoint from val). Goal: best feasibility classification on test using **only**
literature positives (P) + unlabeled pool (U); val (labeled) used for selection/thresholds only.*

Data (unless noted): P = `positives_merged_bh` (64,061 Pd-BH, ORD+Pistachio), U = `unlabeled_grid`
(52,736), val = 2,624 / test = 1,920 (JnJ). Features: ECFP (Morgan, radius 2, 2048 bits/block);
`subs_only` = aryl+amine; `subs_cond` = + ligand/base/solvent fps + metal one-hot. Seed 0.

---

## E1 — Phase-1 headline (supervised, nnPU × subs_only/subs_cond)
- **Config:** `configs/pu_bench.example.yaml` (early 2-featurizer form). `out_dir` artifacts/pubench/phase1.
- **Run:** local, ~3.7 min. Models: SupervisedBaseline (CatBoost), NonNegativePU (linear, prior via
  Elkan-Noto).
- **Test ROC-AUC:** supervised subs_only **0.594**; nnpu subs_only 0.505; nnpu subs_cond 0.340;
  supervised subs_cond 0.322.
- **Interpretation:**
  1. **Substrate-only ≫ substrate⊕condition** (0.59 vs 0.32) — reproduces ADR-005's condition-FP
     transfer *crash* at merged scale: structural condition fingerprints act as a domain tag and
     push transfer below random. → the *featurizer* is a problem, not just the positives.
  2. **Elkan-Noto prior degenerated to 0.01** (clip floor) — SCAR-based prior estimation fails under
     the reporting-bias + shift. A wrong prior cripples nnPU. → motivates shift-robust priors and
     supplying a sane prior.
  3. **Optimistic val, weak test** for nnpu (val 0.72 → test 0.505) vs consistent supervised
     (0.59→0.59) — the substrate-disjoint test is doing its job; val alone would mislead.

## E2 — Phase-2 (supervised, spy, nnPU, IW-PU × subs_only/subs_cond)
- **Config:** `configs/pu_bench.phase2.yaml`. Risk models use fixed **prior = 0.32 (val base rate)**
  to avoid the degenerate Elkan-Noto estimate. `out_dir` artifacts/pubench/phase2. Local, ~6.5 min.
- **Test ROC-AUC (sorted):**
  | model | subs_only | subs_cond |
  |---|---|---|
  | **supervised** | **0.594** | 0.322 |
  | spy | 0.566 | 0.319 |
  | iw_pu | 0.389 | 0.271 |
  | nnpu | 0.389 | 0.271 |
- **Interpretation:**
  1. **The naive supervised baseline (CatBoost, U-as-negative) + substrate-only wins (0.594).** spy
     (PAYN-style reliable negatives) is a close second (0.566). The risk-based PU methods
     **underperform random** (0.389).
  2. **nnPU == IW-PU to 3 dp** → the RuLSIF density-ratio weights collapsed to ~uniform on the
     10,243-dim sparse-binary ECFP (Gaussian-kernel LSIF is degenerate in high dim), so IW-PU reduced
     to nnPU. **Density ratio needs dimensionality reduction to be informative here.**
  3. **Model capacity dominates the PU correction:** the *linear* nnPU/IW-PU scorer (test 0.389) is
     far weaker than the nonlinear CatBoost baseline (0.594) on identical features. On this severe
     cross-domain shift, a strong base learner matters more than the risk correction.
  4. Conditions-as-fingerprints crash every model again (all subs_cond < subs_only).
- **Implications / next levers:**
  - Add **nonlinear PU methods** that use a strong base learner: an **Elkan-Noto scaled classifier**
    (CatBoost P-vs-U, calibrate by `c`) and **Bagging-PU** (bootstrap U as negatives) — likely to
    beat 0.594.
  - **Fix IW density ratio**: reduce dimensionality (PCA / random projection) before RuLSIF, or weight
    a strong classifier instead of a linear risk.
  - **Conditions need descriptors, not fingerprints** (pending Florian's gpuma/xTB features).
  - Best-so-far target to beat: **test ROC-AUC 0.594**.

## E-DIAG — P-vs-U domain-discriminator AUC (the core obstacle) ⭐
- **Setup:** train a HistGradientBoosting classifier to tell P (merged literature positives) from U
  (JnJ-space unlabeled), 3-fold CV, 8k+8k subsample. High AUC ⇒ the two live in disjoint feature
  regions (covariate shift / domain tag); 0.5 ⇒ indistinguishable.
- **Result:** **AUC = 1.000** for *both* `ecfp_subs_only` and `ecfp_subs_cond`.
- **Interpretation (headline):** P and U/test are **essentially non-overlapping in substrate space** —
  literature (Pistachio/ORD) substrates and the JnJ substrate matrix are different molecules. This is
  the fundamental cause of everything above:
  - Transfer is capped near 0.59 because the model must **extrapolate** from P's region into the
    disjoint U/test region; there is almost no covariate overlap to carry a decision boundary across.
  - PU / importance-weighting / density-ratio methods are structurally handicapped: a P-vs-U
    classifier learns *"is this a literature molecule?"* (AUC 1.0), not *"does it react?"*; and
    `p_U/p_P` is ill-posed when the supports barely overlap (→ the RuLSIF degeneracy in E2).
  - The naive supervised baseline still reaches ~0.59 by picking up a weak, scaffold-invariant
    substructure signal — likely near the ceiling for this representation + positive set.
- **Implication:** the only lever that can genuinely shrink this gap is a **shift-invariant, physical
  representation** (ligand/base/solvent + substrate *descriptors*: %Vbur, Sterimol, gap/IP/EA, pKa,
  polarity) rather than scaffold fingerprints — the ADR-005/006 thesis, now strongly re-motivated.
  Florian's g-xTB/morfeus substrate descriptors exist but cover **100% of test yet only ~1% of the
  merged-P (Pistachio) substrates**, so they cannot train on P until computed for the P vocabulary
  (a GPU job). This is the recommended next investment; it is the bottleneck, not the algorithm.

<!-- New experiments appended below. -->
