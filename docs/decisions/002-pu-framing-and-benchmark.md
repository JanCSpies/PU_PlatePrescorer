# 002 — Project framing: PU learning setup, splits, and benchmark design

**Date:** 2026-07-07 · **Status:** accepted (revisit as experiments land) · **Author:** Florian's session (claude)

## Context
After the literature deep dive (`docs/literature/SYNTHESIS.md`) and dataset inspection, we fix the
learning problem, the data roles, and the benchmark axes so both sessions build against the same spec.

## Decision

**Task.** Binary **feasibility** of BH C–N couplings, label = **`Yield ≥ 10 %`** (verified).

**Data roles.**
- **Test (held-out, OOD):** `JNJ HTE 2024` (Neves). Never used for training or tuning. Evaluate with
  Neves' **DRFP + K-means leave-cluster-out** folds (reuse the file's `iteration_*` columns) and, as a
  second view, Zhong-style **one-/both-substrate-unseen** splits.
- **Positives (P):** reactions with `labels == 1` mined from literature sources, integrated
  incrementally for the source-benchmark (non-JnJ HTE ⊂ +USPTO ⊂ +ORD ⊂ +Cernak-negatives).
- **Unlabeled (U):** the **full combinatorial substrate grid** (aryl × amine under the 4 general
  conditions) **minus** the JnJ test reactions, after leakage removal.

**Method.** PU learning beyond SCAR. Benchmark arms (priority): SCAR-spy (PAYN baseline) → PULSNAR
subtype-aware → SAR-EM propensity → PUe. Reliable-negative set = **union(TACIT implicit negatives,
PAYN spies)**. Generative PU (diffusion/flow/GAN) is a **stretch goal**; if attempted, use it
discriminatively to *label* reliable negatives on the enumerable grid, never to synthesise descriptors.

**Features.** Recompute **DRFP** from SMILES (the raw `DRFP`/`QM` columns are truncated and unusable);
optionally add RDKit/Auto-QChem descriptors + one-hot conditions.

**Benchmark axes.** data-source × PU-strategy, evaluated on JnJ OOD with
{ROC-AUC, AUPRC, Balanced-Acc, precision@confidence, ECE} + the **hit-rate-uplift** demo metric.

**No-leakage protocol (mandatory before any source becomes positives).** Canonicalise (RDKit) →
InChIKey → substrate-level exclusion vs JnJ (it is a *substrate*-OOD test) → reaction-hash dedup →
near-duplicate + Neves-cluster audit → temporal guard (prefer pre-2024). See SYNTHESIS §3.

## Rationale
Literature positives are SAR/SNAR (reporting is feature-correlated), not SCAR — the exact gap beyond
PAYN. JnJ is substrate-OOD with real negatives, so it is an honest external test of a literature-only
PU model. The hit-rate-uplift metric turns the classifier into a demoable drug-discovery impact story
with zero new experiments.

## Consequences / revisit-if
- If SAR-EM's propensity features `x_e` cannot be separated from reactivity signal, fall back to
  PULSNAR subtype α-estimation.
- If leakage audit finds substantial literature/JnJ overlap for a source, that source is demoted to
  validation-only.
- The precise "general conditions" subset for U-space generation is finalised in code against
  `docs/literature/notes/bh-JnJ-testset-spec.md`.
