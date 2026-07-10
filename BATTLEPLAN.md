# BATTLEPLAN.md — where we are, and the four directions we now attack

> **Read this first in a fresh session.** It is a self-contained brief: (1) what we found so far and
> how, (2) *why* those findings push us to four new work-streams, and (3) exactly what each
> work-stream is. Pointers to deeper docs are inline (read them only when you dig into that stream).
> Author: Jan's session (Claude), 2026-07-09. Companion to `CLAUDE.md` (charter) and
> `LAB_JOURNAL.md` (append-only log — add a dated entry there when you start any stream below).

---

## 0. The one-paragraph problem

Predict **Buchwald–Hartwig (BH) C–N coupling feasibility** (yield ≥ 10 %) on the held-out **JnJ HTE
2024** plate, training on **literature data only** via **Positive-Unlabeled (PU) learning**
(positives P = literature reactions that worked; unlabeled U = the candidate substrate/condition
space). Use case = pre-score a large substrate matrix so we design higher-hit-rate plates. Full
charter, data map, and collaboration rules: **`CLAUDE.md`**.

---

## 1. What we did, and what we found (the honest state)

Two parallel code tracks in this shared repo:
- **`src/bhpu/`** (Florian) — the main pipeline: C–N mining, domain adaptation, features, transfer
  harness, plate demo. Results in **`RESULTS.md`**.
- **`src/pubench/`** (Jan, this branch) — a config-driven PU **benchmark**: 6 methods × 3
  featurizers, mypy-strict, 24 tests. What's in it: **`docs/pubench/PACKAGE.md`**. Every run +
  number: **`docs/pubench/EXPERIMENTS.md`**. Rundown: **`docs/pubench/README.md`**.

**Data assets already built** (leakage-audited clean — see `docs/pubench/EXPERIMENTS.md` §Leakage
and `docs/decisions/004-cn-coupling-corpus.md`):
- `final_datasets/positives_ord_bh.parquet` — 211 open ORD Pd-BH positives (publishable headline).
- `final_datasets/private/positives_merged_bh.parquet` — **64,061** Pd-BH positives (ORD+Pistachio;
  Pistachio is **licensed → gitignored, never publish**).
- `final_datasets/unlabeled_grid.parquet` — 52,736 unlabeled JnJ-space rows (U).
- `final_datasets/test.parquet` — JnJ (val 2,624 / test 1,920, **substrate-disjoint**).

### Headline numbers (JnJ TEST ROC-AUC, model selected on val, test touched once)
| approach | TEST ROC-AUC | source |
|---|---|---|
| naive transfer (historic) | ~0.52 | early baselines |
| naive supervised (CatBoost, U-as-neg, subs-only ECFP) | 0.594 | pubench E2 |
| **regularised HistGB P-vs-U, subs-only ECFP** | **0.678–0.711** | pubench E5 |
| **IW-PU-bagging (discriminator weights), subs-only ECFP, lit+ORD** | **0.720** | bhpu, `RESULTS.md` |
| plate-design demo | **2.32× hit-rate uplift** (100 % vs 43 % random) | bhpu, `RESULTS.md` |

### The four findings that define the battleplan
1. **The domain gap is the stated ceiling.** A P-vs-U discriminator scores **AUC = 1.000** — literature
   substrates and the JnJ matrix look non-overlapping in ECFP space. Diagnosis:
   `docs/decisions/003-crossdomain-transfer-diagnosis.md`,
   `docs/literature/notes/theory-crossdomain-pu.md`.
2. **The gain so far is transductive, not inductive.** Remove TEST substrates from U → TEST collapses
   to **0.40** (below random). So today's model is an honest **plate pre-scorer over a *known*
   substrate library** (our actual goal), **not** a novel-substrate predictor. (`RESULTS.md` §Robustness.)
3. **Substrate-only wins *because* conditions-as-fingerprints crash.** subs-only ~0.59–0.72 vs
   subs+cond 0.26–0.41. Conditions drive **43 %** of JnJ label variance and give **+0.16 AUC
   in-domain**, but as **fingerprints** they act as a literature-vs-JnJ domain tag (lit and JnJ share
   only **8/81** catalysts). `docs/decisions/005-condition-encoding.md`.
4. **Physics descriptors, so far, are a robust negative — but only tested substrate-only.** xTB and
   site-specific morfeus/g-xTB (correct atoms, conformer ensembles) did **not** beat ECFP and did
   **not** fix the inductive gap (`RESULTS.md` §WS4/WS5). Caveat flagged in
   `docs/decisions/006-pd-bh-conditions.md`: this was measured **without conditions**, so it is not
   settled.

### Two facts about the *current code* that reframe everything (verified 2026-07-09)
- **No dimensionality reduction is applied to the classifier features.** HistGB/CatBoost train on raw
  **4,096-dim** (subs-only) or **10,243-dim** (subs+cond) **sparse binary** ECFP. (`src/pubench/features/ecfp.py`.)
- **The IW-PU that "failed" never got its fix.** `src/pubench/models/iw_pu.py` *has* an optional
  `project_dim` (random-project P/U before the density ratio), but it was set **only in the cancelled
  E3 config**. Every reported IW-PU number ran on raw high-dim, where the RuLSIF density ratio
  collapsed to uniform → IW silently degenerated to nnPU (0.389). **IW-PU-with-DR has never run.**
- **Propensity is a stub.** `needs_propensity` is a flag in `src/pubench/models/base.py` plumbed into
  provenance, but **no propensity / SAR-EM model exists.** The `w=d/(1−d)` in `src/bhpu/adapt/` is
  covariate-shift weighting, *not* a reportability propensity `e(x)`.

---

## 2. Why we move on (the argument in four lines)

- **Substrate-only is chemically uninteresting** — in BH, conditions decide whether a reaction works
  (43 % of the variance). A feasibility model blind to catalyst/base/solvent is answering the wrong
  question. → **Stream A.**
- **"AUC = 1.0 ⇒ disjoint supports" is partly a dimensionality artifact.** In 4,096 sparse dims almost
  any two finite molecule sets are linearly separable, and **importance weighting provably breaks
  under disjoint support** (weights explode/degenerate — which is exactly why IW collapsed). The
  overlap has never been measured in a *reduced* space, and IW has never run *with* reduction. →
  **Stream B.**
- **Conditions must enter as a shared physical space, not fingerprints**, so a bulky electron-rich
  literature ligand and a different-but-similar JnJ ligand map *close* instead of becoming a domain
  tag. Condition descriptors are **cheap** (a few dozen unique ligands/bases/solvents, not 64k
  molecules). → **Stream C.**
- **The whole task is reporting bias**, i.e. textbook **beyond-SCAR / SAR propensity**: a reaction is
  "positive" iff someone *reported* it working. Non-chemical "reportability" features (how often a
  substrate is reported; its price/availability) are near-ideal instruments to separate *"unreported
  because it fails"* from *"unreported because nobody tried"* — the single hardest inference in the
  project, and completely unexplored. → **Stream D.**

Streams attack **different failure modes** and are largely independent — run them in parallel.

---

## 3. The four work-streams

> Convention for all streams: keep the **frozen protocol** (`src/bhpu/eval/protocol.py`,
> substrate-disjoint val/test, select on val, **touch test once**); log leakage checks; record every
> run in `docs/pubench/EXPERIMENTS.md` (metric of record = **TEST ROC-AUC**) and a `LAB_JOURNAL.md`
> entry. Baseline to beat: **0.711** (subs-only regularised HistGB) / **0.720** (bhpu IW-PU).

> **UPDATE 2026-07-09 (E6 — featurization built + run; see EXPERIMENTS.md E6):**
> - **Stream B — mostly refuted.** Condensation does **not** shrink the P-vs-U gap (discriminator AUC
>   still 1.0 at 64-d) and **IW-PU did not wake up** (< HistGB on every featurizer). BUT the by-product
>   is a win: **correlation-condensation (`corr08`, 982-d, drop rare bits + |r|>0.8) = TEST 0.723 ≥ raw
>   0.711 at ~4× lower dim**, while dense **SVD is worse (0.672)**. → `reduced_ecfp method=correlation`
>   is the new cheap substrate default; SVD deprioritised.
> - **Streams A & C — refuted in the current PU framing.** Conditions as **physical descriptors** crash
>   transfer exactly like fingerprints (cond_only 0.32; corr08+cond 0.34). Cause is *distributional*
>   (P vs U/test occupy different condition regions → condition axis = domain tag in a P-vs-U contrast),
>   not representational. The descriptor build is done and correct (236 ligands, full test support), but
>   conditions can't help a raw P-vs-U model. **Redirect:** use conditions only via a **two-part model**
>   (substrate-PU ⊕ in-domain condition model, WS6 idea) or by **balancing P/U condition distributions**
>   → this merges into **Stream D**.
> - **Net:** substrate side is near ceiling (~0.72); the remaining upside is **Stream D (propensity /
>   distribution reweighting)** and the two-part conditions model. Do those next.

> **UPDATE 2026-07-09 (later) — Stream D done, NEGATIVE (see LAB_JOURNAL + ADR-007):**
> - Conditions retired via **ADR-007**: JnJ conditions are OOD by design (54.6 % of test bases <1 % in
>   literature); scope is substrate-level feasibility under general conditions. Streams A/C closed.
> - **Stream D (propensity) is a negative.** Built report-frequency (ORDerly, 530k keys) and ZINC
>   in-stock availability (9.04M keys) as non-chemical reportability signals. As features: redundant.
>   As SAR propensity weights: **hurt** (0.72→0.70). Root cause: propensity/SAR assumes P and U are the
>   same population; ours (literature vs JnJ) are not — the correction reweights by a partial domain tag
>   with weak feasibility signal. Same literature↔JnJ gap as everywhere else.
> - **Conclusion: ~0.72 substrate-only (corr08) is the honest ceiling for pure literature→JnJ transfer.**
>   Every lever (features/conditions/propensity) fails for the same, now-quantified root cause. Only
>   **few-shot in-domain** data breaks it (ADR-007). Next: packaging/plate-demo/write-up, or few-shot.

### Stream A — Use conditions (substrate **+** conditions), done right
**Goal:** move from substrate-only to substrate+condition modelling so the model can express the 43 %
of feasibility that conditions control — *without* re-triggering the domain-tag crash.

**Why the naive version fails:** condition **fingerprints** separate literature from JnJ (8/81 shared
catalysts) → the model learns provenance, not chemistry (`docs/decisions/005-condition-encoding.md`).

**Plan:**
- Encode conditions in a **shared, transfer-robust representation** (this dovetails with Stream C's
  condition descriptors — A defines the *modelling*, C provides the *features*).
- Ensure **U spans the JnJ condition space** (already partly done in `bhpu/scripts/17`) *and* that P
  is re-expressible in the same condition-descriptor axes — the missing half last time.
- Compare `substrate ⊕ condition` vs substrate-only under the frozen protocol; report transductive
  **and** inductive.
- Re-open the deferred question in `docs/decisions/006`: physics substrate descriptors may only become
  useful **once conditions are present** (a bulky ligand matters more for a hindered substrate).

**Success:** subs+cond > subs-only on TEST (beats 0.711) *without* the discriminator AUC snapping to
1.0 on the condition axis.

### Stream B — Condensed / dimensionality-reduced ECFP, then re-run PU (does IW wake up?)
**Goal:** test whether the "no overlap / IW is useless" story is real or an artifact of high-dim
sparse fingerprints.

**Plan:**
- Reduce the ECFP before modelling: **TruncatedSVD** (correct for sparse binary; *not* dense PCA),
  and/or **count/hashed folded** fingerprints to a modest width (e.g. 128–512). Compare to raw 4,096.
- **Re-measure the P-vs-U discriminator AUC in the reduced space.** If it drops below 1.0, the
  "disjoint support" ceiling is softer than claimed and importance weighting becomes well-posed.
- **Run the IW-PU that was cancelled**: set `project_dim` in `src/pubench/models/iw_pu.py`, or feed it
  the SVD-reduced features, so the RuLSIF density ratio is informative instead of degenerate.
- Keep the two IW flavours distinct: **discriminator weights `w=d/(1−d)`** (bhpu — robust in high dim,
  ESS ~6.6k, already at 0.72) vs **RuLSIF density ratio** (pubench — collapsed in high dim). Reduction
  is expected to help the second far more than the first.

**Success:** on reduced features, IW-PU ≥ its own nnPU baseline (i.e. weights stop being uniform) and
ideally ≥ 0.711; discriminator AUC < 1.0 documents that overlap exists once dimensionality is sane.

### Stream C — Better features: cheap condition descriptors (± physics substrate descriptors)
**Goal:** replace condition *fingerprints* with a **shared physical descriptor space**, and decide
whether to also compute physics descriptors for substrates.

**Plan (cheap first, this is the highest leverage-per-effort move):**
- **Condition descriptors — cheap and immediate.** Only a few dozen unique ligands/bases/solvents
  exist across the corpus (unlike 64k substrates), so no GPU needed:
  - **Ligand:** %V_bur, cone angle, electronic params — pull from the **Kraken** phosphine descriptor
    set where available; self-compute (morfeus/RDKit) for the rest. Join on the canonical ligand key
    from `src/bhpu/chem/conditions.py` (already collapses precatalysts → free-ligand identity).
  - **Base:** pKa / pKaH of the conjugate acid; **solvent:** a handful of continuous parameters
    (polarity, etc.). Small lookup tables keyed on the canonicaliser's base/solvent keys.
- **Physics substrate descriptors — the expensive option, gated.** xTB/morfeus for substrates was a
  **negative substrate-only** (finding 4), and Florian's g-xTB/morfeus cover 100 % of test but only
  ~1 % of the merged-P (Pistachio) vocabulary → a **GPU job** (Florian's `gpuma`) to compute them for
  the P vocab. **Only invest here if** Stream A shows substrate descriptors turn useful *once
  conditions are present*, or Stream B shows the overlap gap survives in a reduced space. Otherwise
  prefer **condensed substrate ECFP ⊕ condition descriptors** (cheap combo) as the default feature.

**Success:** condition-descriptor features let subs+cond beat subs-only in **transfer** (Stream A's
target), and the discriminator AUC on the condition axis is < 1.0 (descriptors bridge the 8/81
catalyst gap that fingerprints could not).

### Stream D — Propensity estimation from **non-chemical** features (reporting frequency, price)
**Goal:** model the **reportability propensity** `e(x) = P(labeled | feasible, x)` explicitly — the
beyond-SCAR / SAR framing — using features that predict *whether a reaction gets reported*, not
whether it works. Theory: `docs/literature/notes/pu-sar-beyond-scar-bekker-2019.md`,
`docs/literature/notes/pu-sar-em-lidta-2018.md`; contrast with SCAR/PAYN in
`docs/literature/notes/payn-baseline.md`.

**Why it's the right frame:** a reaction is positive-labeled iff *published as working*. Absence in
literature conflates *"fails"* with *"nobody tried"*. Propensity features disentangle them:
- cheap/common/heavily-reported substrate that is **absent** → high propensity → its absence is
  **informative** → a reliable negative;
- expensive/rare/obscure substrate that is absent → low propensity → absence says nothing → must
  **not** be treated as negative.
This gives **principled reliable negatives** (what spy/PAYN approximate heuristically), and — bonus —
reporting frequency is a **shift-robust feature axis** that is *not* a pure domain tag (a common
building block is common in both lit and JnJ), unlike condition FPs.

**Plan:**
- **Reporting frequency — free, do this first.** We already mined ORD+Pistachio; count how often each
  substrate appears as a *building block across all reaction types* (a `groupby(substrate).count()`),
  which is closer to "popularity/availability" than "BH-feasibility". Add as a propensity feature and
  as a model feature.
- **Price / commercial availability — second pass, the cleaner instrument.** Join substrates to a
  catalog (Sigma-Aldrich / eMolecules / ZINC purchasability). Availability is set by market factors,
  so it's **less confounded with the label** than raw report count.
- **Model:** fit `e(x)` from these non-chemical features, then either (a) SAR-EM style
  reweighting/relabelling of U, or (b) propensity-corrected PU risk. Compare against the current
  IW-PU (0.72) and regularised-HistGB (0.711) baselines.

**Honest caveat to respect:** report *count* is partly confounded (things get reported because they
work — violates the SAR separability assumption). Prefer **price/availability** as the clean
propensity signal, and if using report-count, use the cross-reaction-type building-block count, not
the BH-specific one.

**Success:** a propensity-weighted arm beats 0.72 in transfer, and/or yields visibly better reliable
negatives (cleaner U relabelling) than spy/PAYN — and, ideally, improves the **inductive** number
(finding 2), which nothing has moved yet.

---

## 4. Priority (cheap → expensive; each hits a different failure mode)
1. **D (reporting frequency)** — free from the corpus we already have; fastest to a real number.
2. **C (condition descriptors)** — cheap (dozens of ligands/bases/solvents), unblocks **A**.
3. **B (reduced ECFP + re-run IW)** — cheap; tests whether the "domain-gap ceiling" is partly an
   artifact and whether IW was ever given a fair chance.
4. **A (substrate+conditions modelling)** — uses C's features; the chemically right target.
5. **C-physics (xTB/morfeus for all substrates, GPU)** — **deferred/gated** on A or B showing it's
   worth the compute (per the "no wasted compute on a negative" rule).

## 5. Deeper reading map (open only when you enter that stream)
- **What's in the pubench code:** `docs/pubench/PACKAGE.md`
- **Every benchmark run + number:** `docs/pubench/EXPERIMENTS.md`
- **Deferred methods & why:** `docs/pubench/FRONTIER_METHODS.md`
- **Main-pipeline transfer results + plate demo:** `RESULTS.md`
- **Domain-shortcut theory:** `docs/literature/notes/theory-crossdomain-pu.md`,
  `docs/decisions/003-crossdomain-transfer-diagnosis.md`
- **Condition-encoding crash + fix rationale:** `docs/decisions/005-condition-encoding.md`,
  `006-pd-bh-conditions.md`
- **SAR / propensity theory (Stream D):** `docs/literature/notes/pu-sar-beyond-scar-bekker-2019.md`,
  `pu-sar-em-lidta-2018.md`
- **Corpus provenance + leakage:** `docs/decisions/004-cn-coupling-corpus.md`,
  `final_datasets/assembly_report.json`
- **Charter / data map / commands / SLURM:** `CLAUDE.md`, `docs/slurm.md`
