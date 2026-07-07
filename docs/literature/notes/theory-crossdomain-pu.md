# Cross-domain PU learning: why literature→JnJ collapses to random, and how to fix it

*Theory + implementation note. Compiled 2026-07-07. Complements `SYNTHESIS.md`, `pu-payn-boser-2026.md`,
`pu-sar-beyond-scar-bekker-2019.md`, `pu-sar-em-lidta-2018.md`, `pu-pulsnar-kumar-lambert-2024.md`.*

**One-line thesis:** ROC-AUC 0.52 (≈ random) on literature→JnJ is **not** a symptom of a bad prior or of SCAR
violation *per se* — ROC-AUC is invariant to both. It is the signature of a **ranking that has been captured by
the domain / reporting axis instead of the feasibility axis**. The fix is to (i) remove the domain shortcut and
(ii) make the *unlabeled* set live in the JnJ covariate region, and only then worry about prior/calibration.

---

## 1. Failure diagnosis — decompose the shift, then rank what bites

Write the training joint as `p_tr(x,y,s)` (s = "reported in literature", the only label we observe) and the test
joint as `p_te(x,y)` (JnJ). The gap between them factors into four distinct shifts. **Critically, they do not all
hurt the same metric**, which is the key to diagnosing 0.52.

**(b) The domain shortcut — the killer, and the reason it is exactly 0.52.**
Naive pipeline: P = literature positives, U = JnJ combinatorial grid. But literature substrates and JnJ substrates
are *nearly disjoint* (verified: 0 shared amines, ≤2 shared aryls, 0 shared pairs — `bh-JnJ-testset-spec.md §7`).
A P-vs-U classifier minimizes error, and the single cheapest error-minimizing direction is **"was this molecule
drawn from the literature manifold or the JnJ manifold?"** — a direction with near-zero Bayes error for *domain*
but ~orthogonal to *feasibility*. The classifier ranks every JnJ reaction by "how literature-like" → all JnJ points
score similarly low/uninformative w.r.t. their true label → **ROC-AUC ≈ 0.5 on feasibility**. This is the smoking
gun. **Diagnostic to run first:** train a domain discriminator (lit vs JnJ) on DRFP; if its held-out AUC ≈ 1.0
(it will be), the naive PU is provably learning domain, not feasibility. Report that number — it *is* the
explanation of 0.52.

**(a) Covariate shift, `p_tr(x) ≠ p_te(x)`.** Even without an explicit P/U domain split, the feasibility decision
boundary learned in the dense literature region is *extrapolated* into the JnJ region where the model has no
support. Reactivity classifiers are notoriously non-smooth across substrate classes, so extrapolation ≈ noise.
This compounds (b): the two together destroy ranking. **Bites hardest #1 (with b).**

**(c) SCAR violated → SAR/SNAR.** Reporting propensity `e(x)=P(reported|feasible,x)` is feature-dependent (yield,
novelty, substrate popularity, ligand commonness; Jia 2019: reagent popularity is *uncorrelated* with success,
p=0.26). This matters for AUC **only insofar as the reporting features overlap the reactivity features** — then the
P-vs-U classifier ranks by "reportability" (a confound of b). If `e(x)` were driven by features orthogonal to
reactivity, SCAR-violation alone would *not* wreck AUC, only the recovered prior. So SAR is a **second-order
amplifier of the domain shortcut**, not an independent cause of 0.52. **Bites #2.**

**(d) Class-prior / label shift, `π_tr ≠ π_te`.** Literature is positive-rich; JnJ is 39.1% feasible.
**This does NOT lower ROC-AUC at all** — AUC is invariant to the class prior and to any monotone score transform.
Prior shift corrupts *thresholded* metrics (accuracy, F1), *absolute* probabilities, and therefore **plate ranking
and precision@conf** — but it cannot explain 0.52 AUC. Treat (d) purely as a **calibration** problem (§4).
**Bites the demo metric, not the AUC.**

**Verdict:** the 0.52 is (b)+(a). SAR (c) makes it worse; prior shift (d) is orthogonal to AUC and belongs in
calibration. Spend week-1 effort on killing the domain shortcut and moving U into the JnJ region; treat SAR and
prior as refinements.

---

## 2. Why PU-bagging (Mordelet–Vert) and PAYN-spy do not transfer

Both are **in-domain** methods that implicitly assume P and U share `p(x|y=1)` and that `π_te = π_tr`.

- **PU-bagging** (Mordelet & Vert 2014, `10.1016/j.neunet.2013.06.003` 🔒; `pulearn.BaggingPuClassifier`):
  repeatedly subsamples U as pseudo-negatives and averages P-vs-subsample scores. In-domain (our ~0.90) the
  averaged score ranks by feasibility. **Cross-domain the averaged score ranks by domain** — every bag learns the
  same lit-vs-JnJ direction, so bagging *stabilizes the wrong signal*. More bags ≠ more feasibility.
- **PAYN spy / reliable negatives** (`pu-payn-boser-2026.md`): spies are *literature* positives; the spy-tolerance
  quantile `t_spies` is calibrated in the *literature* score/feature density. Reliable negatives mined below
  `t_spies` are RN *in the literature region*; mapping that threshold onto JnJ reactions is pure extrapolation, so
  RN precision on the JnJ manifold is unknown/likely poor (recall PAYN's own Stevens case: RN collapse to ~50%
  purity when featurization gives poor separability — cross-domain is that case by construction).
- **Prior dependence.** PAYN's spy machinery and any Elkan–Noto `c = P(s=1|y=1)` rescaling assume `π` is stable.
  Our `π_lit ≫ π_jnj`, so even a perfectly-ranked score gets mis-thresholded and mis-calibrated.
- **Calibration.** CatBoost/spy scores are calibrated to the literature base rate; on JnJ they are systematically
  over-confident-positive.

PAYN remains the right **in-domain baseline and test harness** — but as the SCAR/single-domain reference it is
exactly the arm we expect to fail cross-domain, which is the story.

---

## 3. Methods to fix cross-domain PU — ranked by (payoff / week-1 cost)

Each: key idea in words → Python path → caveat.

### R1. Transductive U = JnJ-domain grid, in an aligned/shortcut-free representation  *(highest payoff)*
The most important single change: the **unlabeled set must live where the test lives**. Use `U = JnJ combinatorial
grid (110 aryl × 68 amine × condition sets) minus measured/test pairs` (already the plan, `bh-JnJ-testset-spec §9`).
But naively pairing P(lit) with U(JnJ) *is* the domain shortcut (§1b). So U-in-target must be combined with
**shortcut removal** (R4). The clean framing: P from literature tells us where positives are; U from JnJ tells us
the target covariate region; alignment prevents the classifier from cheating on the domain gap between them.
- **Path:** build U from the grid; then feed R4. Evaluate whether RN mined *inside* the JnJ-U (not lit-U) transfer.
- **Caveat:** if a JnJ positive *subtype* has no literature analogue (support failure), it is unrecoverable — this
  is the irreducible OOD ceiling (PULSNAR support assumption; Neves' "diversity drives OOD").

### R2. SAR debiasing of the positives: propensity `e(x_e)` / SAR-EM  *(medium payoff, medium cost)*
Model `e(x)=P(reported|feasible,x)` from **reporting-predictive, reactivity-orthogonal** features
`x_e = {data source (Reaxys vs each HTE campaign as a discrete stratum), aryl/amine literature-popularity counts
(from USPTO/ORD frequency), publication year, ligand/base commonness}`. Then either propensity-weight the PU risk
(unbiased ERM: label i enters as pos weight `1/e_i` **and** neg weight `1−1/e_i`; Bekker 2019) or run **SAR-EM**
(joint EM over classifier f and propensity e; `pu-sar-em-lidta-2018.md` has the pseudocode). Cheapest first-order
version: **stratify by data source** (SAR-as-multi-SCAR) — one `c_source` per campaign.
- **Path:** logistic-regression e-head + CatBoost f with `sample_weight`; or reimplement SAR-EM (~150 lines). Use
  `pywsl` for reference risk estimators. Calibrate CatBoost (isotonic) before the EM E-step (E-step needs
  probabilities).
- **Identifiability caveat (the hard part):** `x_e` must be **predictive of reporting but lower-dimensional /
  not collinear with reactivity** — else "unlabeled" is unattributable between low-feasibility and low-reporting and
  EM degenerates (e≡1). In chemistry reporting and reactivity are entangled (high-yield → reported), so choose
  `x_e` deliberately from *bibliometric* metadata, never from DFT/DRFP reactivity descriptors. **Never** put JnJ-set
  membership, source=JnJ, or the yield-derived label into `x_e` (leakage).

### R3. Importance weighting / density-ratio for covariate shift  *(medium payoff, LOW cost; but fragile)*
Reweight literature training risk by `w(x) = p_te(x)/p_tr(x)` so the empirical risk approximates the JnJ test risk
(Sakai & Shimizu, "Covariate Shift Adaptation on PU Data", AAAI 2019, `10.1609/aaai.v33i01.33014838` — the paper
that *proves when IW is necessary for PU* and gives the importance-weighted PU risk; and "Importance-weighted PU
for Distribution Shift Adaptation", OpenReview `CTYgrczjj2`). Three ways to get `w`:
- **Classifier-based (cheapest):** train domain discriminator `d(x)=P(JnJ|x)`; `w(x) = d(x)/(1−d(x)) · (n_lit/n_jnj)`.
  Feed as `sample_weight` to the existing PAYN CatBoost — a ~20-line change to the current stack.
- **uLSIF / RuLSIF / KLIEP:** `densratio_py` (`from densratio import densratio`; PyPI `densratio`) — direct density
  ratio, no discriminator. RuLSIF's α-relative ratio bounds the weights (use α≈0.1) → variance control.
- **KMM:** `adapt.instance_based.KMM` (matches means in RKHS).
- **Caveat — this is the fragile one.** Literature and JnJ supports barely overlap, so honest weights explode
  (a few points carry all mass; effective sample size collapses). **Always clip/self-normalize weights** and report
  ESS. Under near-disjoint support, IW alone underperforms feature alignment (R4) — prefer R4, keep R3 as the
  cheap fallback that plugs straight into PAYN.

### R4. Domain-invariant representation: CORAL / DANN  *(high payoff; CORAL low cost, DANN medium)*
Directly removes the §1b shortcut at the representation level, and (unlike IW) **works under non-overlapping
support** because it *builds* an overlapping representation.
- **CORAL / Deep CORAL (start here):** align second-order statistics — whiten literature features by their
  covariance and recolor to the JnJ covariance (closed-form, no training). `adapt.feature_based.CORAL` /
  `DeepCORAL`. ~10 lines; run PU in the aligned space.
- **DANN:** adversarial gradient-reversal head forces the representation so a domain discriminator can't tell lit
  from JnJ, while the feasibility head still predicts. `adapt.feature_based.DANN` (torch/TF). Combine the label
  head with a **PU loss (nnPU)** rather than PN.
- **Caveat:** if feasibility genuinely correlates with domain, aggressive alignment can *erase feasibility signal*
  → monitor that the label head's val-AUC (on a held-out lit fold) doesn't collapse as domain-AUC → 0.5. Tune the
  adversarial weight λ. CORAL is safer (only 2nd-order) but weaker.

### R5. Class-prior shift handling — prior-invariant PU + label-free JnJ prior  *(needed for calibration, low cost)*
- **Estimate `π_jnj` on the JnJ-U WITHOUT labels:** run DEDPUL (`github.com/dimonenka/DEDPUL`, arXiv:1902.06965),
  KM2 (`arXiv:1603.02501`), and TIcE (Bekker & Davis AAAI 2018) on {lit-P vs JnJ-U}; cross-check the three (and
  against PULSCAR/SAR-EM α from the existing notes) as a sanity anchor. Newer: prior-shift PU via kernel embedding
  (`arXiv:2502.21194`, 2025) for label-shift-robust α̂.
- **Prior-invariant classifier:** Nakajima & Sugiyama 2023, "PU classification under class-prior shift: a
  prior-invariant approach based on density-ratio estimation" (Machine Learning, `10.1007/s10994-022-06190-z` 🔒;
  arXiv:2107.05045 open) — trains **without any prior**, and the JnJ prior is injected **only at test time**.
  Ideal because our train prior is unknown/biased and the test prior differs. `pywsl` has related PU-under-shift
  code.
- **PUe (NeurIPS 2023, OpenReview `6vtZIoxZoJ`):** normalized propensity + normalized IPW (NIPW) — a modern,
  unbiased "beyond-SCAR" estimator that also *estimates* propensity via regularization when the mechanism is
  unknown. A strong alternative to hand-designed `x_e` in R2.

### R6. nnPU / uPU with JnJ-domain U + estimated prior  *(medium payoff, low-medium cost)*
Kiryo et al. 2017 non-negative PU risk (`arXiv:1703.00593`, `github.com/kiryor/nnPUlearning`; also `pywsl`): unbiased
PU risk with a non-negativity clamp that prevents the flexible-model overfitting uPU suffers. Needs the prior `π`
(from R5) and a U set (use JnJ-U). Best used **as the label loss inside DANN (R4)** so you get shortcut-free +
prior-correct in one model. Caveat: needs `π_jnj`; sensitive to π mis-estimation.

**Ranked shortlist:** R1+R4 (transductive-U + alignment, kills the 0.52 cause) ≫ R2 (SAR debias) ≈ R5 (prior/calib)
> R3 (IW, cheap fallback) ≈ R6 (nnPU).

---

## 4. Calibration across domains (for plate ranking / precision@conf)

ROC-AUC needs none of this; **plate ranking and precision@conf do**. Two composable steps:

1. **Score→prob on a held-out literature fold:** CatBoost is roughly calibrated; refine with isotonic or Platt
   (`sklearn.calibration.CalibratedClassifierCV`, `cv="prefit"`). PULSNAR's label-flipping calibration
   (`pu-pulsnar-kumar-lambert-2024.md`) is a PU-native alternative.
2. **Prior-shift correction to `π_jnj`:** apply the Saerens–Latinne–Decaestecker EM prior-adjustment (2002) or the
   closed-form odds rescale `odds' = odds · (π_te/(1−π_te)) · ((1−π_tr)/π_tr)` using the label-free `π_jnj` from R5.
   Lipton BBSL/RLLS (`arXiv:1802.03916`) if label shift is severe.
3. **Under covariate shift, calibrate with importance weights:** reweight the calibration set by `w(x)` (R3) —
   "Unsupervised Calibration under Covariate Shift" (arXiv:2006.16405), "Calibrated Prediction with Covariate Shift
   via UDA" (arXiv:2003.00343). Cheap add-on if R3 weights already exist.
- **Report:** ECE on Neves' 5 confidence buckets + **precision@conf>0.85** (the operational metric; Neves' RF hit
  >90%). Precision@conf is prior-sensitive → do the calibration before quoting it.

---

## 5. Single recommended pipeline (+ 3 fallbacks)

**RECOMMENDED — "Domain-aligned transductive PU with SAR debiasing and prior-corrected calibration":**
0. **DRFP features** (recompute from SMILES; not the truncated column). Leakage-audit per `SYNTHESIS §3`.
1. **Diagnostic gate:** train lit-vs-JnJ domain discriminator on DRFP. Its AUC (~1.0 expected) is your published
   explanation of the 0.52 baseline. Keep `d(x)` — reused in steps 3 and 6.
2. **Transductive U:** U = JnJ combinatorial grid − measured/test pairs. P = leakage-audited literature positives.
3. **Kill the shortcut:** align features with **CORAL** (fast) → if val holds, escalate to **DANN** with an
   **nnPU** label loss. Target: domain-AUC → ~0.5 while a held-out-lit feasibility val-AUC stays high.
4. **SAR debias P:** propensity `e(x_e)`, `x_e` = {source stratum, aryl/amine lit-popularity, year, ligand
   commonness}; propensity-weight or SAR-EM. (Skip in v0 if time-boxed; add in v1.)
5. **Prior + calibration:** estimate `π_jnj` label-free (DEDPUL ∥ KM2 ∥ TIcE, cross-checked); isotonic-calibrate on
   held-out lit; prior-shift-correct to `π_jnj`; optionally IW-calibrate with `w(x)`.
6. **Rank & evaluate** JnJ candidates by calibrated feasibility on Neves' `iteration_*` OOD folds →
   ROC-AUC, AUPRC, Balanced-Acc, ECE, **precision@conf**, + the hit-rate-uplift demo.

**Fallback 1 (lowest cost, drops into the existing PAYN CatBoost):** **Importance-weighted PAYN** — classifier-based
`w(x)=d(x)/(1−d(x))`, self-normalized + clipped, passed as `sample_weight`; retrain spy-PU. ~20 lines. Fragile
under support mismatch (report ESS), but the cheapest thing that could move the needle.

**Fallback 2 (most principled for the prior/label-shift axis):** **Nakajima–Sugiyama prior-invariant density-ratio
PU** — no prior at train, inject `π_jnj` at test. Robust when train prior is unknown/biased.

**Fallback 3 (representation-only, if SAR+prior are too much):** **DANN + nnPU** alone (steps 1–3 + 6, skip 4–5),
calibrate post-hoc. Directly targets the dominant failure (§1b) with minimal moving parts.

---

## 6. DOIs / links (🔒 = paywalled; Florian has Münster access)

**Cross-domain PU / covariate shift**
- Sakai & Shimizu, *Covariate Shift Adaptation on Learning from PU Data*, AAAI 2019 — `10.1609/aaai.v33i01.33014838` (open, ojs.aaai.org).
- *Importance-weighted PU Learning for Distribution Shift Adaptation*, OpenReview `CTYgrczjj2` (open).
- Kumagai et al., *PU AUC Maximization under Covariate Shift*, PMLR 2025 — proceedings.mlr.press/v267/kumagai25a (open).
- Nakajima & Sugiyama 2023, *PU classification under class-prior shift (prior-invariant, density-ratio)*, Machine Learning — `10.1007/s10994-022-06190-z` 🔒 · arXiv:2107.05045 (open).

**Beyond-SCAR / propensity**
- Bekker, Robberechts, Davis 2019 (SAR/SNAR taxonomy, propensity-weighted ERM) — `10.1007/978-3-030-46147-8_5` 🔒 · arXiv:1809.03207 (open). *(local note)*
- Bekker & Davis 2018 (SAR-EM) — PMLR LIDTA 94:8 · arXiv:1808.08755 (open). *(local note)*
- PUe (NIPS 2023, normalized IPW) — OpenReview `6vtZIoxZoJ` (open).
- Kumar & Lambert 2024 (PULSNAR/PULSCAR) — `10.7717/peerj-cs.2451` (open) · arXiv:2303.08269. *(local note)*

**Core PU risk / prior estimation**
- Kiryo et al. 2017 (nnPU/uPU) — NeurIPS · arXiv:1703.00593 (open) · code `github.com/kiryor/nnPUlearning`.
- Elkan & Noto 2008 (SCAR) — `10.1145/1401890.1401920` 🔒.
- Mordelet & Vert 2014 (PU-bagging) — `10.1016/j.neunet.2013.06.003` 🔒.
- DEDPUL — Ivanov, ICMLA 2020 · arXiv:1902.06965 (open) · code `github.com/dimonenka/DEDPUL`.
- KM1/KM2 — Ramaswamy et al. ICML 2016 · arXiv:1603.02501 (open).
- TIcE — Bekker & Davis AAAI 2018 (open).
- Prior-shift PU via kernel embedding, 2025 — arXiv:2502.21194 (open).

**Domain adaptation / feature alignment**
- ADAPT toolbox (KMM, KLIEP, uLSIF, CORAL, DeepCORAL, DANN, TrAdaBoost) — de Mathelin et al., arXiv:2107.03049 (open) · `github.com/adapt-python/adapt` · docs adapt-python.github.io.
- DANN — Ganin et al., JMLR 2016 · arXiv:1505.07818 (open).
- Deep CORAL — Sun & Saenko, ECCVW 2016 · arXiv:1607.01719 (open); CORAL — Sun et al., AAAI 2016 · arXiv:1511.05547.
- KMM — Huang et al., NeurIPS 2006 (open).
- densratio_py (uLSIF/RuLSIF/KLIEP) — `github.com/hoxo-m/densratio_py` · PyPI `densratio`.
- pywsl (uPU, nnPU, PU-under-covariate-shift) — `github.com/t-sakai-kure/pywsl`.

**Calibration under shift**
- Saerens, Latinne, Decaestecker 2002 (EM prior adjustment) — `10.1162/089976602753284446` 🔒.
- Lipton et al. 2018 (BBSL/RLLS label shift) — arXiv:1802.03916 (open).
- Unsupervised Calibration under Covariate Shift — arXiv:2006.16405 (open).
- Calibrated Prediction with Covariate Shift via UDA — arXiv:2003.00343 (open).

**Chemistry priors (already in local notes)**
- Jia et al. 2019 (reagent popularity ⟂ success, p=0.26) — Nature 573:251. *(`data-jia-anthropogenic-bias.md`)*
- PAYN — Boser, Spies, Glorius, JACS 2026 — `10.1021/jacs.6c00127` (CC-BY). *(local note)*
- Neves 2025 (JnJ test set, OOD protocol) — `10.26434/chemrxiv-2025-xcr46` (open). *(local note)*
