# 001 — Drop the Ahneman (`Dreher Doyle 2018`) source

**Date:** 2026-07-07 · **Status:** accepted · **Author:** Florian's session (claude)

## Context
The raw curated dataset contains a source labelled `Dreher Doyle 2018` (4,312 rows). We needed to
decide whether to keep it as literature training data.

## Findings
Structural fingerprint of the source: **15 aryl halides × 1 fixed amine × 4 ligands × 3 bases ×
23 isoxazole additives**, and it is the **only** source with `Additive_present == 1` (4,132 rows;
plus 180 no-additive controls = 15×4×3). This is unmistakably **Ahneman, Estrada, Lin, Dreher,
Doyle, *Science* 2018** (DOI: 10.1126/science.aar5169) — the canonical BH C–N *additive* screen,
often cited as the "Ahneman/Doyle dataset".

The scientific variable in that dataset is the **isoxazole additive**. But the curated file has
**no additive-identity column** — only engineered flags (`Additive_present`,
`Additive_aromatic_rings`, `Additive_heteroatom_count`, …). Two rows that differ *only* in which of
23 additives was used therefore have **identical** `Aryl/Amine/Catalyst/Base/Solvent/Product SMILES`
and identical recomputable reaction featurisation, but different `Yield`/`labels`. The additive
information is irretrievable from this file.

## Decision
**Drop all `Source == "Dreher Doyle 2018"` rows from training/unlabeled data.** They are written to
`datasets/processed/dropped_ahneman_dreher_doyle_2018.*` for transparency only.

## Rationale
- Feeding irreducibly ambiguous rows (same features, contradictory labels) into a feasibility model
  injects pure label noise concentrated on a single narrow substrate region.
- The splitter asserts Ahneman is the *only* additive-bearing source, so dropping it removes no
  additive signal from any other source.
- Consistent with the user's explicit instruction.

## Consequences / revisit-if
- If a version of the Ahneman data **with additive SMILES** is obtained, it could be re-added as a
  distinct, additive-aware experiment (a separate feasibility regime), not merged blindly.
- We lose a well-known BH benchmark for *additive* effects — out of scope for substrate-feasibility.
