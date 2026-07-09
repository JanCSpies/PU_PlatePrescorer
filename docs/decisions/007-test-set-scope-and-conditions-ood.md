# 007 — Test-set scope: substrate-level feasibility; JnJ conditions are OOD from literature

**Status:** decided 2026-07-09 (Jan's session, claude). Decision: **scope the deliverable to
substrate-level feasibility under general HTE conditions; retire cross-domain condition modelling
(fingerprints *and* descriptors) for literature→JnJ; keep JnJ as the test set.** Refines
[005](005-condition-encoding.md) / 006 (the "descriptors will bridge conditions" expectation does not
hold for the PU-transfer setting).

## Context — what we are actually building

Literature-only, no-wet-lab feasibility triage: train a PU model on **literature positives P** (ORD /
Pistachio) + a **self-assembled unlabeled pool U** to pre-score a set of BH C–N reactions a chemist is
about to run, so they can **skip the ones unlikely to work** and save resources. The only wet-lab
needed is a **one-time HTE test set** (with positives *and* negatives) covering the space of interest.
We use **JnJ HTE 2024 (Neves)** because it is (about the only) large public BH HTE with explicit
negatives over a broad substrate space. This decision records what JnJ does and does not validate.

## What we found (E6 + diagnostics, 2026-07-09)

**JnJ's reaction conditions are largely out-of-distribution from the literature corpus**, especially
the base:

| base | literature (P) | JnJ test |
|---|---|---|
| P2-Et phosphazene | ~0 % | **33.6 %** |
| phosphate | 2.2 % | 26.7 % |
| carbonate | 49.7 % | 18.8 % |
| HMDS | ~0 % | 9.5 % |
| ArO⁻ / TMSO⁻ | ~0 % | 8.2 % |

- **54.6 %** of JnJ test rows use a base that is **< 1 %** frequency in the literature; the *median*
  literature-frequency of the base a test row uses is **0.0 %**.
- A P-vs-U discriminator on the **condition descriptors alone** scores **AUC 0.998** — conditions are
  a near-perfect domain tag (even though ligand *identities* overlap 80–96 %; it is the base/solvent
  axes and the joint distribution that differ).
- In E6, encoding conditions **crashes transfer** whether as fingerprints *or* physical descriptors:
  `cond_only` ≈ 0.32 (below random), and adding conditions to the best substrate representation
  **destroys** it (corr08 0.723 → corr08+cond 0.340). See `docs/pubench/EXPERIMENTS.md` (E6).

**Why this happens (chemistry, not a bug):** JnJ is a **general-conditions HTE screen** — it uses a
few *robust, substrate-tolerant* condition sets (P2-Et phosphazene, HMDS, silanolate) chosen so one
plate works across many substrates. Literature reports *bespoke, cost-optimised* conditions per
reaction (cheap carbonate). Neves *designed* JnJ to be OOD ("Robust **OOD** prediction of BH
reactions"). So literature simply never explores JnJ's condition regime — a **data-coverage gap** no
algorithm or representation can close.

**Why the condition crash is *below* random (0.32), while the equally-separable substrate axis
*helps* (0.72):** it is about *alignment*, not separability. "Looks like a literature substrate" weakly
*correlates* with JnJ feasibility (shared substructures); "looks like a literature (patent) condition"
*anti*-correlates with JnJ feasibility, because JnJ's working conditions are the general ones that are
rarest in patents.

## Decision & reasoning

1. **Scope the model to substrate-level feasibility under general conditions.** The dominant driver of
   "will aryl X + amine Y couple *at all*" is substrate compatibility, largely independent of which
   *good* base/ligand is chosen. JnJ's label ("works under robust general conditions") is a sound proxy
   for "coupleable under *some* reasonable conditions" — exactly the triage signal the use case needs.
   The substrate-only number (**~0.72 ROC-AUC / 2.3× plate uplift**) is the honest headline.
2. **Retire cross-domain condition modelling for literature→JnJ** — fingerprints (005) and descriptors
   (006) both. The obstacle is *distributional coverage*, not representation, so a better encoding
   cannot fix it. The descriptor featurizer (`condition_descriptors`, 236 ligands cached, full test
   support) is correct and kept — it could serve a *future in-domain / few-shot* condition model, just
   not this transfer.
3. **Keep JnJ as the test set.** It is the best available large HTE with explicit negatives over a
   broad, pharma-relevant substrate space; its substrates are representative of the target chemistry.
   The condition regime being non-classic is a **scope caveat, stated explicitly**, not a reason to
   discard it (no better public pos+neg HTE exists).
4. **Deployment is transductive — and that is legitimate here.** In the use case the chemist has a
   *defined series of reactions to score*, so U is assembled to *be that candidate series*; the scored
   substrates are always in U. The transductive ~0.72 is therefore the correct operating number (the
   inductive-collapse caveat in `RESULTS.md` does not bite this use case).

## Consequences

- The substrate-only 0.72 is **not** secretly using conditions (they are OOD and the model ignores
  them) → a more credible number, honestly scoped.
- **To model condition effects would require different data**: a *few-shot in-domain* pilot (run a
  small plate in the target conditions, learn the condition effect, scale) or a general-conditions HTE
  with negatives. Neither blocks the core literature-only deliverable.
- The remaining literature-only lever is on the **positives / substrate** side: **Stream D
  (propensity)** — de-biasing *which literature substrates count as positives* via non-chemical
  reportability signals (report frequency, price). See `BATTLEPLAN.md`.

See `docs/pubench/EXPERIMENTS.md` (E6, E-DIAG-2), `LAB_JOURNAL.md` (2026-07-09), `RESULTS.md` §WS6,
and ADRs [005](005-condition-encoding.md) / 006.

## Update (E7, 2026-07-09) — carbonate salvage tests the hypothesis directly; decision holds, sharpened

Florian(human) asked to *test* the OOD-conditions diagnosis rather than only assert it: restrict JnJ to
its **literature-common-condition subset** (carbonate base, `BVKZGUZCCUSVTD`; 18.8 % of test, 49.7 % of
P) and re-run — with both the **normal P** and a **carbonate-only P**. Built a config-driven base filter
(`BaseKeyFilter`) and ran three regimes (E7c full / E7a test-carb / E7b all-carb). Carbonate test n=360;
95 % CI ≈ ±0.06. Result (TEST ROC-AUC, `sup_histgb_l2`):

| featurizer | full test | carb test (full P/U) | carb test (carb P/U) |
|---|---|---|---|
| corr08 (subs) | 0.723 | 0.698 | 0.695 |
| corr08+cond | **0.340** | **0.570** | **0.558** |
| cond_only | 0.321 | 0.497 | 0.506 |

**What it changes:** the earlier claim that conditions score *below random* (0.32–0.34) was a
**distributional artifact** and is removed in-distribution — restricting the *test* to carbonate lifts
`corr08+cond` to 0.57 and `cond_only` to ~random. JnJ's working conditions (P2-Et/HMDS) are the ones
literature never uses, so on the full test "literature-like condition" *anti*-correlates with
feasibility; on carbonate that anti-correlation is gone.

**What it confirms (decision unchanged):** conditions still **do not help** even fully in-distribution
(`corr08+cond` 0.558 < `corr08` 0.695; `cond_only` ≈ 0.51), and matching the *training* condition regime
(E7a→E7b) changes nothing. The obstacle is therefore **not only** distributional coverage but that a
**positive-only P carries no transferable condition→feasibility signal** — P has only *working*
condition/substrate combos, so a P-vs-U contrast has no failed-condition examples to learn from. Point 1
of the decision (scope to substrate-level feasibility) and the "conditions need different data" caveat
**stand and are strengthened**: condition modelling requires **in-domain negatives** (few-shot, or an HTE
with negatives *in the target conditions*), not merely a balanced condition distribution. The
substrate-only number is regime-robust (~0.70 on carbonate ≈ 0.72 on full). See `EXPERIMENTS.md` (E7).
