# Literature synthesis — BH C–N feasibility from literature data via PU learning

*Compiled 2026-07-07 from a 5-territory subagent deep dive (22 per-paper notes in
`docs/literature/notes/`). This is the actionable top-level summary; open a note for detail.*

---

## 0. The settled problem framing

- **Target:** binary **feasibility** of Buchwald–Hartwig (BH) C–N couplings, defined as
  **assay yield ≥ 10 %** (verified exactly against the data: JnJ label-0 spans 0–9.9 %, label-1
  spans 10.0–114.3 %). This 10 % floor ≈ the direct-to-biology assay-readability floor (~12 %,
  Cernak) — so *predicting the label literally predicts D2B-assayability*.
- **Test set:** `JNJ HTE 2024` from **Neves et al., "Robust OOD Prediction of Buchwald–Hartwig
  Reactions," ChemRxiv 2025** (DOI `10.26434/chemrxiv-2025-xcr46`; code
  `github.com/schwallergroup/bh-hte-ood`; Zenodo 17634928). 11,328 reactions, 110 aryl × 68 amine
  (888/7,480 pairs measured), 32 Pd precatalysts / 8 bases / 6 solvents, **39.1 % feasible**.
- **OOD protocol to reproduce:** Neves split by **DRFP + K-means leave-cluster-out** (the file's
  `iteration_{0..9} cluster` / `train_test` columns expose 10 iterations × 12 clusters — reuse them).
  Bar to beat: a plain **Random Forest on DRFP reached >90 % precision at confidence > 0.85 OOD**;
  **diversity, not size, drove performance** (their CRDS metric; Spearman 0.967 with ROC-AUC).
- **Our differentiator vs Neves:** they train *fully supervised with real HTE negatives*. We train
  **from literature (positive-only, biased) data via PU learning** — no proprietary HTE core needed.

**The pipeline:** literature positives (P) + full combinatorial substrate grid minus test
(unlabeled, U) → PU learning recovers reliable negatives / propensity → feasibility classifier →
evaluate on JnJ OOD → plate-design + hit-rate-uplift demo.

---

## 1. PU method direction — the methodological core (agents A & B)

**Key insight: our data is NOT SCAR.** PAYN assumes **SCAR** (labeling probability constant:
`P(labeled|positive)=c`). But literature positives are *reported* reactions, and reporting is driven
by yield, novelty, substrate/ligand popularity, and journal scope — all **feature-correlated**. So
our data is **SAR** (propensity `e(x)=P(reported|feasible,x)`) at best, **SNAR** at worst. This is
precisely the gap PAYN leaves open and our contribution exploits. (Jia, Nature 2019: reagent
popularity is statistically **uncorrelated with success**, p=0.26 — absence from literature is a
social artefact, not evidence of infeasibility.)

**Recommended benchmark arms (priority order):**
1. **SCAR-spy (PAYN) baseline** — reproduce exactly; our reference + test harness (CatBoost + Optuna,
   spy-based reliable negatives). See `pu-payn-*.md`.
2. **PULSNAR — subtype-aware PU** — cluster literature positives by chemistry family, estimate a
   per-subtype label frequency α, use cluster-specific reliable-negative thresholds. Cheapest, most
   chemically interpretable "beyond-SCAR" win; directly attacks PAYN's single global spy threshold.
3. **SAR-EM (propensity-weighted)** — model `e(x)` from *reporting*-predictive features
   `x_e = {data source, substrate literature-popularity, publication year, reagent commonness}`;
   jointly fit propensity + classifier. The hard, interesting modeling problem: find `x_e` that
   predicts *reporting* but not *feasibility*.
4. **PUe (NeurIPS 2023)** — modern normalized-IPW/propensity beyond-SCAR with a GBM/deep base; strong
   SAR learner without hand-designing `x_e`.

**Cross-check** all class-prior α estimates (PULSCAR upper bound, PULSNAR, SAR-EM) against each other
as a sanity anchor. Use PAYN's Tanimoto-separability diagnostic (SI Table S8) as a per-subtype
pre-flight check.

**Generative PU (diffusion / flow / GAN) = STRETCH GOAL, not week-1.** All generators need the
unknown prior or a false well-separation assumption, and a generated reaction *vector* need not decode
to a valid enumerable coupling ("invalid chemistry"). The **CoVPU / "Positive Distribution Pollution"**
paper (validated on tabular, 0.1 %-labeled, selection-biased data — our regime) even shows
GAN-generated data *hurts*. **Adopt the ideas, not the generators:** prior-free variational losses
(VPU/CoVPU), balanced-PU π-invariant classifier (from Joint-Diffusion-PU), and the unifying reframe:
*few + publication-biased + unknown-prior positives are one problem — a polluted estimate of the
feasible-reaction distribution.* If we ever touch generative negatives, do it **discriminatively**:
rank cells of the **enumerable** BH grid by low PU posterior to *label* reliable negatives — never
synthesize descriptors. See `genpu-*.md`.

---

## 2. BH feasibility SOTA & metrics to match/beat (agent C)

| Work | Model / data | Split | Metrics we should adopt |
|---|---|---|---|
| **Neves** (our target) | RF on DRFP; JnJ HTE, real negatives | DRFP K-means leave-cluster-out | ROC-AUC, Balanced Acc, AUPRC, **precision@high-confidence** |
| **Zhong** Nat Commun 2025 | Bayesian NN + NUTS; 11,669 amide couplings | **one-/both-substrate-unseen** | Acc, F1-macro, AUC, **calibration (ECE)**, wrong-prediction curve, **aleatoric uncertainty ≈ reproducibility** |
| **Nippa** Nat Commun 2025 | GNN outcome filter → 26k virtual library → 14 subnanomolar inhibitors | 0D/1DA/1DN/2D, ≥5 % positive | **precision-first** funnel evaluation |
| **Angello** Science 2022 | general-conditions objective `f(c)=mean yield across substrates` | — | rank the 4 general-condition sets by predicted mean feasibility |

**Adopt:** Neves' precision@confidence + their exact folds; Zhong's substrate-unseen splits +
calibration + the uncertainty≈reproducibility readout (a genuine differentiator).

**TACIT ⊕ PAYN recipe (our headline method):** TACIT gives a **structural "does this C–N site react
at all" gate** (bond-ranking + stop action; recovers implicit negatives from positive-only data —
flags 51 % of literature failures). PAYN gives the **condition-resolved "≥10 % under these
conditions" via spy PU**. Build the **reliable-negative set = union(TACIT implicit negatives, PAYN
spies)**, train PU, predict JnJ. Test whether the two negative sets are complementary (likely — TACIT
is condition-agnostic/structural, PAYN is condition-resolved). See `bh-tacit.md`, `bh-JnJ-testset-spec.md`.

---

## 3. Literature data sources & no-leakage protocol (agent D)

**Ranked positives, and integration order for the source benchmark** (report each increment's lift on
JnJ + each source's Jia-style amine/aryl Gini + PU-negative recall vs measured negatives):
1. **USPTO (Lowe)** ~1.8 M, open — baseline positives (BH via NameRxn 10.1.x).
2. **+ ORD** ~2.4 M, CC-BY-SA — *first subtract the Dreher/Ahneman/AZ HTE sets it re-hosts.*
3. **+ bundled non-JnJ HTE `label=1`** (Merck/Roche/HiTEA/Dreher/AZ, 11,857 rows) — **substrate-disjoint
   from JnJ by construction** (verified: 0 shared amines, ≤2 shared aryls), in-domain positives + real negatives.
4. **+ Cernak 50,688** (Regan & Cernak / Das…Cernak, JACS 2026, `10.1021/jacs.6c05959`) — only ~4
   electrophiles × 2–3 nucleophiles but 166 ligands; **~71 % measured zero-yield wells = gold
   negatives** to validate PU-recovered reliable negatives. **Down-weight as positives** (tiny
   substrate set biases the model); its value is negative-validation + condition priors.
5. **Pistachio** (13.3 M, licensed) only if permitted; **Reaxys/SciFinder** for *validation counts
   only* — ToS forbid ML training/redistribution.

**No-leakage protocol (enforce in code before any source is used):**
1. Canonicalize components (RDKit canonical SMILES; strip atom-maps/salts/isotopes; neutralize) → InChIKey per aryl & amine.
2. **Substrate-level exclusion:** drop any literature reaction whose aryl *or* amine InChIKey ∈ JnJ substrate set (JnJ is a **substrate**-OOD test — same substrate under new conditions still leaks).
3. **Reaction-level dedup:** DRFP hash / RInChI; remove exact duplicates vs JnJ and cross-source (ORD⊃Dreher; HiTEA appears twice).
4. **Near-duplicate audit:** DRFP + Tanimoto; report count of high-similarity literature/JnJ pairs as a leakage metric.
5. **Cluster audit:** reuse Neves' `iteration_N cluster` columns — no training reaction in a JnJ-test cluster.
6. **Temporal guard:** prefer pre-2024 literature so JnJ chemistry can't leak backward.

⚠️ **Watch:** Cernak's `3-bromopyridine` IS a JnJ aryl (its 8 pairs don't coincide with JnJ's 888, so
pair-leakage = 0, but track it for aryl-OOD). See `data-sources-catalog.md`.

---

## 4. Impact story & demo (agent E)

**Narrative:** *Exploratory HTE / D2B plates waste most wells on reactions that fail — and a failed
well yields no molecule to assay. We train on **literature data only** via PU learning to pre-score a
substrate matrix, then design a plate that fits ~2× more real, testable, diverse molecules into the
same footprint — proven on an unseen J&J HTE set with **zero new experiments**.*

**Killer facts:** Cernak's D2B campaign spent **4 conditions × 768 compounds = 3,072 wells** to make
768 molecules *because they couldn't predict the right condition* (the "4× well tax" a model recovers);
crude products at **<12 % conversion were invisible in the pooled assay** (low yield masquerades as
inactivity → false negatives) — empirically validating why the yield/feasibility signal is load-bearing
for D2B. Delta vs Nippa: same library-design funnel, but a **PU literature-only** feasibility filter,
no proprietary HTE core (the democratizing angle).

**The week-1 experiment — hit-rate uplift on JnJ (zero new experiments):**
1. Train PU feasibility (+ optional PAYN yield) on **literature C–N only** (leakage-audited); JnJ stays OOD.
2. Score every JnJ candidate `u(w) = feasibility × Pr[ŷ ≥ 10 %]`.
3. Compare **model plate** (top-B, and a diversity-penalized variant) vs **random plate** (1,000× bootstrap ±CI) vs **oracle**, at B ∈ {96, 384}.
4. Report **uplift** (rate_model / rate_random), **Δ extra hits/plate**, **enrichment factor**. Oracle
   ceiling ≈ 2.56× (precision@96 = 100 %); a competent literature-PU model at 65–80 % precision →
   **~1.7–2.0× uplift, ~25–40 extra usable molecules per 96-well plate**.
5. **Honesty caveat for judges:** JnJ is an *optimization* set (39 % base rate is already enriched vs a
   true cold-start plate) → our measured uplift is a conservative lower bound.

**Two money-shot visuals:** (a) random-vs-model plate colored by *true* JnJ outcome (≈39 % vs ≈75 %
green, side by side); (b) enrichment curve (hit-rate vs plate size: model vs random±CI vs oracle).
Consider an interactive Artifact (budget slider → live hit-rate). Load the `dataviz` skill first. See
`bio-impact-and-demo-plan.md`.

---

## 5. Concrete benchmark matrix (what to actually run)

Two crossed axes + a fixed evaluation:
- **Data-source axis:** {non-JnJ HTE} ⊂ {+USPTO-BH} ⊂ {+ORD-BH} ⊂ {+Cernak-negatives}.
- **PU-strategy axis:** {SCAR-spy (PAYN)} · {PULSNAR subtype} · {SAR-EM propensity} · {PUe} · [stretch: VPU].
- **Evaluation (fixed):** JnJ OOD via Neves DRFP-cluster folds → {ROC-AUC, AUPRC, Balanced-Acc,
  precision@conf, ECE} + the **hit-rate-uplift** demo metric.
- **Features:** recompute **DRFP** from SMILES (do NOT use the truncated `DRFP`/`QM` columns);
  optionally add RDKit/Auto-QChem descriptors and one-hot conditions.

---

## 6. Read-these-first (onboarding path)
1. `bh-JnJ-testset-spec.md` — the test set + OOD protocol (start here).
2. `pu-sar-beyond-scar-bekker-2019.md` + `pu-payn-boser-2026.md` — why SCAR fails us; the baseline.
3. `data-sources-catalog.md` — where positives come from + leakage protocol.
4. `bh-neves-ood.md` + `bh-tacit.md` — the target method & the feasibility gate.
5. `bio-impact-and-demo-plan.md` — how we demo impact.
6. `genpu-positive-distribution-pollution-liang2023.md` — the one generative-PU idea worth knowing.

---

## 7. Paywall pull-list (🔒 = Florian to fetch via Münster) — highest value first
- 🔒 Rinehart & Doyle, substrate-adaptive Pd C–N conditions (origin of the 4 P2Et general sets), *Science* 2023, 381, 965 — `10.1126/science.adg2114`.
- 🔒 Cernak 50k C–N (paper for the local data), *JACS* 2026, 148(25) — `10.1021/jacs.6c05959` (open preprint: ResearchSquare `rs-7603680`).
- 🔒 King-Smith et al., HTE "reactome" (source of HiTEA), *Nat. Chem.* 2024, 16, 633 — `10.1038/s41557-024-01470-8`.
- 🔒 Ha et al., Merck–MIT Pd C–N models (source "Merck 2025"), *JACS* 2025, 147, 19602.
- 🔒 Elkan & Noto 2008 (SCAR foundation) — `10.1145/1401890.1401920`; Bekker & Davis PU survey, *Mach. Learn.* 2020 — `10.1007/s10994-020-05877-5`; SAR — `10.1007/978-3-030-46147-8_5`.
- 🔒 Gao & Coley 2025, publication bias vs reactivity — `10.1021/jacs.4c15660`.
- 🔒 Strieth-Kalthoff/Glorius, failed experiments, *Angew.* 2022 — `10.1002/anie.202204647`; Maloney/Coley negative data, *Org. Lett.* 2023 — `10.1021/acs.orglett.3c00792`.
- 🔓 Open & useful now: Verifying-SCAR test `arXiv:2404.00145`; PUe OpenReview `6vtZIoxZoJ`; solid-state synthesizability-from-literature PU `10.1039/D5DD00065C`; DRFP `10.1039/D1DD00006C`; substrate-reactivity-via-PU `10.1021/acsomega.5c05523`.
