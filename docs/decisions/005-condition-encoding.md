# 005 — Encoding reaction conditions: fingerprints fail, descriptors needed

**Status:** investigated 2026-07-08 (Florian's session, claude). Decision: **encode conditions via
descriptors, not fingerprints; do not use structural condition FPs for literature→JnJ transfer.**

## Context

The winning feasibility model encoded **only the substrates** (aryl ⊕ amine ECFP), ignoring the
catalyst/ligand/base/solvent. Florian pointed out that conditions determine whether a coupling works
— the reason HTE is run. Verified against the data:

- JnJ is a genuine condition screen: **1,137 condition sets** (32 catalysts × 8 bases × 6 solvents).
- **43% of the JnJ label variance is condition-driven** (within-pair); **35% of pairs flip** outcome
  across conditions.

So the reported substrate-only 0.72 is structurally blind to ~43% of the signal.

## What we tried

1. **Structural condition fingerprints** (`features/condition_fp.py`): ECFP of the catalyst/base/
   solvent SMILES (JnJ catalyst SMILES are full Pd precatalyst complexes → metal + ligand captured).
2. **Condition-spanning U** (`scripts/17`): rebuilt the unlabeled grid as unmeasured pairs ×
   conditions sampled from the JnJ condition distribution (not the 4 fixed general conditions), so U
   lives where the test lives in condition space too.
3. **In-domain diagnostic** (`scripts/18`): supervised JnJ VAL→TEST (substrate-disjoint, conditions
   shared), substrate vs substrate+condition.

## Result

| setting | substrate | substrate + condition-FP |
|---|---|---|
| **in-domain** (JnJ val→test, conditions *shared*) | 0.609 | **0.769 (+0.16)** |
| **transfer** (lit→JnJ, conditions *disjoint*) | 0.69 | **0.26 (crash)** |

## Decision & reasoning

- **Conditions carry large, learnable signal** (+0.16 AUC in-domain). They must be encoded — Florian
  is right, and this is the highest-value remaining direction.
- **Structural condition fingerprints must NOT be used for the literature→JnJ transfer.** Literature
  and JnJ share only **8 of 81 catalysts**, so condition FPs place the positives (literature) and the
  target (JnJ) in disjoint regions → the P-vs-U contrast becomes "is this a literature condition?"
  and dominates the substrate signal (domain-shortcut AUC 1.0; TEST → 0.26). This is the same
  mechanism that made reagent-inclusive **DRFP** score below random. Rebuilding U over JnJ conditions
  was necessary but not sufficient — the positives still don't use JnJ's conditions.
- **Next lever = condition DESCRIPTORS** (a shared *physical* space): ligand %V_bur / cone angle /
  electronics, base pKa, solvent polarity/coordinating parameters. These align functionally-similar
  but structurally-different conditions across literature and JnJ, which fingerprints cannot. A
  combined model — literature-PU for substrate feasibility ⊕ condition descriptors for condition
  effects — is the natural synthesis (substrate signal 0.72 and condition signal +0.16 are
  complementary).

See `LAB_JOURNAL.md` (2026-07-08) and `RESULTS.md` §WS6.
