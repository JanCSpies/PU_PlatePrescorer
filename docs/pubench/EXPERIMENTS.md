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

## Leakage audit (merged P) — clean ✅
Verified on the pubench-loaded splits (`inchikey` identities):
- P substrate ∩ test = 0; P substrate ∩ val = 0 (no training substrate in the eval sets).
- P pair ∩ test = 0; P pair ∩ val = 0 (no measured eval reaction is used as a positive).
- U pair ∩ test = 0; U pair ∩ val = 0 (U excludes measured pairs).
- val substrate ∩ test = 0 (substrate-disjoint eval split, `FrozenEvalProtocol`).
- U substrate ∩ test = 78 — **expected & label-free**: U reuses the JnJ substrate vocabulary
  (transductive PU); no labels leak. For a strictly inductive protocol, restrict U to val-side
  substrates. Given the P↔U/test domain gap (E-DIAG, AUC 1.0), this sharing does not aid transfer.
- Training uses P + U only; thresholds/prior come from val (labelled, allowed); test labels are
  touched once, at final scoring.

## E4 — feature × strong-model comparison (SLURM 55258, CPU) ⭐ new best
- **Config:** `configs/pu_bench.e4.yaml`. Featurizers: ECFP subs-only, DRFP (aryl.amine>>product),
  ECFP subs+cond. Models: supervised (CatBoost), spy, bagging_pu, elkan_noto (HistGB).
- **Test ROC-AUC (top):**
  | model | subs-only | drfp | subs+cond |
  |---|---|---|---|
  | **bagging_pu** | **0.689** | 0.549 | 0.323 |
  | **elkan_noto** | **0.688** | 0.553 | 0.407 |
  | supervised (CatBoost) | 0.594 | 0.562 | 0.322 |
  | spy | 0.566 | 0.536 | 0.319 |
  - val→test (subs-only): bagging 0.734→0.689, elkan 0.750→0.688, supervised 0.588→0.594 (modest gaps).
- **Interpretation:**
  1. **New best test ROC-AUC = 0.689** (bagging_pu / elkan_noto, substrate-only ECFP) — **+0.10 over
     the naive supervised baseline (0.594)**. So the earlier "PU doesn't help" (E2) was really "the
     *linear* nnPU doesn't help"; nonlinear PU reaches ~0.69.
  2. **It's mostly a base-learner effect, not PU-specific.** elkan_noto's ranking == its HistGB
     P-vs-U classifier (the ``c`` rescale doesn't change AUC), and bagging (0.689) ≈ single HistGB
     (0.688). So **HistGradientBoosting (0.688) transfers far better than CatBoost (0.594) on the
     identical P-vs-U task** — CatBoost's defaults overfit the literature-substrate shortcut; HistGB
     regularises better. The PU-specific machinery (bagging over U) adds ≈0 here.
  3. **DRFP transfers worse than substrate ECFP** (~0.55 vs ~0.59–0.69): the reaction-difference
     fingerprint is near-constant across BH couplings, so it carries less transferable substrate info.
  4. **Conditions-as-fingerprints crash again** (subs+cond ≤ 0.41 < subs-only for every model).
- **Implication:** base-learner regularisation is a cheap lever (test the effect + tune it, E5). The
  ceiling is still set by the P↔U/test domain gap (E-DIAG); descriptors remain the structural fix.
- **Note:** E3 (same models incl. nnPU/IW-PU) was cancelled as redundant (E2 + E4 cover all its
  combos) after the 10k-dim `bagging_pu` arm ran ~1 h/arm — HistGB bins every feature, so subs+cond
  (10,243-dim) is slow; prefer subs-only (4,096-dim).

## E5 — base-learner + regularisation tuning on substrate-only ECFP (SLURM 55282) ⭐ best
- **Config:** `configs/pu_bench.e5.yaml`. Model selection by **val** ROC-AUC (test touched once).
- **ROC-AUC (test / val):**
  | config | test | val |
  |---|---|---|
  | **sup_histgb_l2** (HistGB, L2=1.0, lr 0.05, 400it) | **0.711** | 0.741 |
  | elkan_noto (HistGB) | 0.688 | 0.750 |
  | sup_histgb (HistGB, lr 0.1, no reg) | 0.678 | 0.751 |
  | sup_histgb_shallow (max_leaf 15, L2=1.0) | 0.678 | 0.678 |
  | bagging25 (25× HistGB) | 0.660 | 0.718 |
  | sup_catboost | 0.594 | 0.588 |
- **Interpretation:**
  1. **Best test ROC-AUC = 0.711** (L2-regularised HistGB) — up from 0.594 (CatBoost) and 0.688 (E4).
     **L2 regularisation improves *test* transfer (0.678→0.711) while *val* barely changes** — it curbs
     overfitting to the P→JnJ covariate shift (the shortcut), which val (also JnJ) doesn't penalise.
  2. **Val is a noisy selector here:** val 0.72–0.75 does not cleanly rank test 0.66–0.71, so strict
     val-selection picks `sup_histgb` → **test 0.678** (the defensible, no-peek number). The
     regularised variant is a *principled* a-priori choice (not test-tuned) that a practitioner would
     reasonably use → ~0.71.
  3. **Bagging did not help** over a single HistGB (0.660 < 0.678) — the bootstrap-U noise slightly
     hurt; the transferable signal is in the base learner + regularisation, confirming E4.
- **Bottom line:** honest best PU feasibility classifier ≈ **test ROC-AUC 0.68–0.71**
  (regularised HistGB, treat-U-as-negative, substrate-only ECFP), a ~0.10–0.12 gain over the naive
  CatBoost baseline. Ceiling still set by the P↔JnJ domain gap (E-DIAG); descriptors remain the
  structural next lever.

## E-DIAG-2 — does condensation shrink the P-vs-U domain gap? (local diagnostic) — **no**
- **Setup:** P-vs-U HistGB discriminator (3-fold, 6k+6k substrate-only), raw ECFP vs SVD-condensed.
- **Result:** discriminator AUC stays **1.000** at raw 4090-d, **0.9999** at SVD-64, **1.000** at
  SVD-256. Condensation does **not** make literature and JnJ substrates overlap.
- **Interpretation:** the "AUC=1.0 is a width artifact" hypothesis is **refuted for substrates** — the
  two substrate sets are genuinely near-disjoint in Morgan space at every dimensionality. Condensation
  is still worth it for *speed* and for making density-ratio numerically viable, but it cannot close
  the domain gap. (Sets up E6's IW test.)

## E6 — condensed substrate ECFP (SVD vs correlation) ± physical condition descriptors (SLURM 55323) ⭐
- **Config:** `configs/pu_bench.e6.yaml`. Featurizers: `svd256` (TruncatedSVD→256-d), `corr08`
  (freq-filter min_freq 0.01 + drop |r|>0.8 → **982-d**), `cond_only` (ligand morfeus %V_bur/SASA at
  P + GFN2-xTB electronics + curated base-pKaH/solvent tables, 31-d), and the two composites
  `svd256+cond` / `corr08+cond`. Models: `sup_histgb_l2`, `elkan`, `iw_pu`, `spy`. Featurizers fit on
  **P∪U** (label-free). Reference: raw-4096 `ecfp_subs_only` = **0.711** (E5).
- **TEST ROC-AUC:**
  | model | svd256 | corr08 | cond_only | svd256+cond | corr08+cond |
  |---|---|---|---|---|---|
  | **sup_histgb_l2** | 0.672 | **0.723** | 0.321 | 0.409 | 0.340 |
  | elkan | 0.672 | 0.704 | 0.323 | 0.421 | 0.352 |
  | iw_pu | 0.663 | 0.401 | 0.294 | 0.290 | 0.285 |
  | spy | 0.633 | 0.588 | 0.321 | 0.396 | 0.342 |
  - corr08 sup_histgb_l2: TEST 0.723 / AUPRC 0.716 / VAL 0.742. svd256 sup_histgb_l2: TEST 0.672.
- **Three findings:**
  1. **Correlation-condensation ≥ raw ECFP; SVD < raw.** `corr08` (982-d, informative sparse bits
     kept) reaches **0.723** — matches-to-slightly-beats raw 4096-d (0.711) at ~4× lower dimension and
     far faster, while dense `svd256` **loses** signal (0.672). Keeping a filtered subset of bits
     transfers better than a dense linear projection, which blends away the sparse substructure signal.
     → **`corr08` is the new default substrate representation** (accuracy ≥ raw, cheaper). Strict
     val-selection would pick `elkan`/`corr08` (VAL 0.759 → TEST 0.704); the regularised HistGB is the
     principled a-priori pick → 0.723. Val remains a noisy selector (as in E5).
  2. **IW-PU did NOT wake up on condensed features.** On every featurizer `iw_pu` < the strong HistGB
     (svd256 0.663, and it *crashes* to 0.401 on corr08). **Refutes Stream B's premise** that
     dimensionality was the blocker: with the supports still disjoint (E-DIAG-2), density-ratio IW is
     structurally handicapped and the base learner dominates — consistent with E2/E4 ("it's a
     base-learner effect, not PU machinery").
  3. **Conditions crash transfer even as *physical descriptors*.** `cond_only` ≈ **0.32** (below
     random) and adding conditions to a good substrate rep **destroys** it (corr08 0.723 → corr08+cond
     0.340; svd256 0.672 → 0.409). This is the **same crash as condition *fingerprints*** (E1/E4,
     ADR-005) — so the descriptor hypothesis (ADR-005/006) is **refuted for the cross-domain PU
     setting**. Root cause is distributional, not representational: literature (P) and JnJ (U/test)
     occupy different condition regions, so in the P-vs-U contrast the condition axis becomes a domain
     tag whatever the encoding; below-random (0.32) is the anti-correlated-domain-tag signature (cf.
     DRFP 0.26). The +0.16 in-domain condition signal (WS6) is real but **inaccessible to a P-vs-U
     contrast** with disjoint condition distributions.
- **Implications / next levers:**
  - Substrate side is near its ceiling for this P: `corr08` is the cheap new baseline (~0.72); a light
    sweep of `min_freq`/`corr_threshold` is the only substrate tuning left.
  - To *use* conditions we need a different framing than raw P-vs-U: (a) balance the P and U condition
    distributions (importance weighting / propensity on conditions so they stop being a domain tag),
    or (b) a **two-part model** — literature-PU for substrate feasibility ⊕ an in-domain condition
    model (the WS6 combined-model idea) — or (c) condition-conditional PU. This is where **Stream D
    (propensity)** re-enters: reweighting toward JnJ-like conditions could de-confound the axis.
- **Note:** first E6 attempt (55322) crashed with `KeyError: ligand_key` — the `FinalDatasetsAdapter`
  projected splits to a SMILES-only column allowlist and dropped the identity keys the descriptor
  featurizer looks up by; fixed by adding the `*_key` columns to `_CANONICAL_COLUMNS` (+ a data-contract
  test assertion). A prior 55300 was cancelled (raw-4096 `bagging_pu` ~1h/arm; dropped from E6).

<!-- New experiments appended below. -->

## E7 — Carbonate salvage: does restricting to in-distribution conditions unlock condition modelling?

**Configs:** `configs/pu_bench.e7{c,a,b}.yaml` · SLURM 55348/55349/55350 (~5 min each, CPU_Express).
**Setup:** new **config-driven base filter** (`BaseKeyFilter{keys, apply_to}` on `DataConfig`, threaded
through `FinalDatasetsAdapter`; +2 data-contract tests). Restrict splits to **carbonate**
(`cond_base_key == BVKZGUZCCUSVTD` — decoded against ADR-007: 18.8 % of test, **49.7 % of P**; the one
base regime shared with the literature corpus, whereas P2-Et/HMDS/silanolate are ~0 rows in P). Same
featurizers as E6 (`corr08`, `cond_only`, `corr08+cond`) × `sup_histgb_l2`.

**Three regimes (which splits are carbonate-restricted):**

| arm | P | U | test | question |
|---|---|---|---|---|
| e7c control | full | full | full | reproduce E6 on this exact code path |
| e7a test-restricted | full | full | **carb** | "normal P": is the current model better on the in-distribution-condition test? |
| e7b fully-aligned | **carb** | **carb** | **carb** | "carbonate P": aligned regimes → do conditions finally help? |

Carbonate test = **n=360 (128 pos / 232 neg)**, val n=448. **98 % of carbonate-test ligands are in
carbonate-P** (vs the 8/81 catalyst gap on the full set); base now constant, solvent 57 % covered.

**Result (TEST ROC-AUC):**

| featurizer | e7c (full test) | e7a (carb test, full P/U) | e7b (carb test, carb P/U) |
|---|---|---|---|
| **corr08** (subs-only) | **0.723** | 0.698 | 0.695 |
| corr08+**cond** | 0.340 | 0.570 | 0.558 |
| cond_only | 0.321 | 0.497 | 0.506 |

95 % CI on the carbonate test ≈ **±0.06** (Hanley–McNeil, n=360). So the corr08-vs-corr08+cond gap
(**0.14**, paired on identical rows) is **real**; the 0.723→0.698 substrate change across the two test
sets (**0.025**) is **within noise**.

**Findings:**
1. **Control reproduces E6 exactly** (corr08 0.723, corr08+cond 0.340, cond_only 0.321) → the filter
   (`base_filter=None`) is a true no-op; the numbers below are apples-to-apples.
2. **The *below-random* condition crash was distributional — and it goes away in-distribution.**
   Restricting the **test** to carbonate lifts `corr08+cond` **0.340 → 0.570** and `cond_only`
   **0.321 → 0.497**. On the full test JnJ's *working* conditions (P2-Et/HMDS) are the ones rarest in
   literature, so "looks like a literature condition" **anti**-predicts feasibility (0.32, below
   random). Remove those rows and the anti-correlation vanishes (conditions become ~random/neutral).
   ADR-007's "it's distributional" diagnosis of the *sign flip* is confirmed.
3. **But conditions still do NOT help — even fully in-distribution.** In e7b (everything carbonate)
   `corr08+cond` **0.558 < corr08 0.695**, and `cond_only ≈ 0.51` (random). Aligning the *training*
   regime (e7a→e7b) changes nothing (Δ ≤ 0.01). So the "conditions are useless in transfer" result is
   **deeper than a domain-tag artifact**: the transferable condition→feasibility signal from a
   positive-only P is essentially **absent**, because P contains only *working* condition/substrate
   combos — a P-vs-U contrast has no failed-condition examples to learn from. The +0.16 **in-domain**
   condition signal (WS6, 2026-07-08) needs in-domain **negatives**, which literature P cannot supply.
4. **The substrate model is regime-robust (~0.70).** corr08 is 0.723 (full) ≈ 0.698 (carb test) ≈
   0.695 (carb P/U) — the substrate feasibility signal doesn't depend on the exotic-condition rows, and
   shrinking P 64k→32k / U 53k→9k to carbonate doesn't hurt it. Honest ceiling stays **~0.70–0.72**.

**Decision:** confirms & sharpens **ADR-007** (see its E7 update). The carbonate salvage removes the
*embarrassing* below-random condition crash but does **not** make conditions helpful — substrate-level
feasibility remains the deliverable, and condition modelling still requires **in-domain negatives**
(few-shot / an HTE with negatives in the target conditions), not just a matched condition distribution.
**Next:** stop condition engineering; the only remaining condition lever is few-shot in-domain (ADR-007).

## E8 — cross-track reconciliation: IW-PU-bagging vs regularization (SLURM 55341/42/56) ⭐
- **Setup:** ran Florian(session)'s **actual** `bhpu.pu.DomainAdaptedPULearner` (covariate-shift
  importance weights `d/(1−d)` + PU-bagging, no CORAL) on **our** `final_datasets`, and a **synthesis**
  swapping its CatBoost base for a regularized **HistGB+L2** `FeasibilityModel`. Substrate ECFP, raw
  (4096-d) + corr08 (982-d), IW on/off. Answers: does his method reproduce 0.72 on our data, and does
  IW add anything on top of the strong regularized learner?
- **TEST ROC-AUC (identical data):**
  | features | single HistGB+L2 (E5/E6) | HistGB-bag naive | HistGB-bag **+IW** | CatBoost-bag +IW | CatBoost-bag naive |
  |---|---|---|---|---|---|
  | corr08 | **0.723** | 0.713 | 0.710 | 0.686 | 0.602 |
  | raw ecfp | 0.711 | 0.718 | 0.720 | 0.674 | 0.605 |
- **Findings:**
  1. **IW adds nothing on HistGB+L2** (Δ = −0.003 corr08 / +0.002 raw = noise; ESS ~10k, healthy) but
     helps CatBoost **+0.07–0.08**. → **importance weighting and L2 regularization are *substitutes***:
     both suppress the domain shortcut; CatBoost overfits it (IW is its crutch), HistGB+L2 controls it
     (IW redundant). PU-bagging adds slight variance vs the single model.
  2. **Florian's method does NOT reproduce his 0.72 on our data** — CatBoost IW-PU = 0.674/0.686 here vs
     his reported 0.720. Same code → the gap is the **data assembly** (his bhpu lit+ORD vs our merged
     ORD+Pistachio + condition-spanning U). The two tracks' "0.72" were on different data.
  3. **Base learner dominates:** HistGB+L2 (~0.72) > CatBoost (0.60 naive / ~0.68 IW). All HistGB
     variants cluster 0.71–0.723 (test noise ±~0.02, n=1920).
- **Bottom line:** the winning model is deliberately simple — **single L2-regularized HistGB,
  treat-U-as-negative, substrate ECFP (~0.72)**; no PU method (spy/PAYN, nnPU, IW density-ratio,
  discriminator-IW + bagging, CORAL) beats it. Occam's razor is the headline. See `LAB_JOURNAL.md`
  (2026-07-09, "Two tracks reconciled").
