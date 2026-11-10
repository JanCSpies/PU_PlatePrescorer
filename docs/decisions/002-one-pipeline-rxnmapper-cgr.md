# 002 — One pipeline for every source: RXNMapper mapping, CGR check, roles from the mapping

- **Status:** Accepted, 2026-10-09
- **Supersedes:** 001 in part (how roles, agents and solvents are obtained; mapping) · **Superseded by:** —

## Context

The repository will take many sources (ORDerly, ORD, USPTO 76-16 and 01-16, Pistachio, CHORISO, CJHIF)
and several reaction classes (amide coupling first, Buchwald–Hartwig next). HTE sets (Chemlex, HiTEA)
are benchmarks only. Sources differ in shape: role columns (ORDerly, CHORISO), one reaction SMILES
(USPTO, Pistachio, CJHIF), or one unsorted list of every molecule (ORD). ADR 001 made rxnutils the
standardisation layer, kept each source's own atom mapping, and took agents and solvents from the
source columns. That only works for sources that are mapped and have role columns.

Measured on 2026-10-09 (scratch run, LAB_JOURNAL):

- **RXNMapper** (rxnmapper 0.4.3, called through rxnutils) on 500 random Chemlex reactions: 0 failures,
  500/500 acid and amine identified correctly, 498/500 with the carboxyl C–N bond exactly right. The
  2 errors are acids with a second, amide C=O. ~20 reactions/s on one CPU (~60/s for smaller ORDerly
  reactions).
- On 2,000 ORDerly reactions, RXNMapper and ORDerly's source mapping pick the same reactants in 91.8 %
  (amide products: 93.5 %). Disagreements go both ways: RXNMapper adds DCM or Pd(PPh3)4 as a reactant;
  the source keeps TFA, HCl, DMSO or DMF as reactants (as ADR 001 already noted).
- **CGR dynamic bonds** (rxnutils `cgr_dynamic_bonds` = bonds formed + broken, CGRtools): all 498
  correct Chemlex mappings have 2, both wrong ones 4. Of 137 acid + amine → amide reactions in the
  ORDerly sample, 116 have 2; the others are multi-step reactions (coupling + ester hydrolysis or ketal
  removal), imide formation, data errors, non-couplings (Boc protection) and 2 RXNMapper errors.
- Literature: RXNMapper is the most accurate open mapper (Lin et al., *Mol. Inform.* 2022, 41, 2100138:
  83.7 % on the Golden set; consensus mapping adds 0.3 % at 30× the time). 28 % of its errors keep the
  same chemical distance; for two-substrate couplings these are wrong-atom choices among equivalent
  groups, which the single-reactive-group filter removes. Roles = molecules that contribute atoms to
  the product (Schneider), and standardisation is redone after mapping because RXNMapper reorders and
  re-writes molecules (Gimadiev et al., *Mol. Inform.* 2021, 40, 2100119).

## Decision

1. **One pipeline; only the adapter depends on the source.** Every source goes through the same stages:
   ingest → reaction SMILES → mapping → rxnutils clean → classify → standardise → positives /
   benchmark labels. Benchmarks (Chemlex, HiTEA) run through the same stages.
2. **RXNMapper maps every reaction.** The source's own mapping is kept only as `source_rxn` for checks.
   The mapper runs in a separate environment (Python 3.12, torch, transformers, `setuptools<81`)
   through rxnutils' mapping script, called directly so its confidence and batching are kept. Mappings
   go into one cache for all sources, keyed by a hash of the canonical unmapped reaction SMILES
   (licensed sources: a separate cache under `private/`).
3. **Roles, agents and solvents come from the mapping.** After rxnutils `reactants2reagents`, every
   molecule that contributes no atom to the product is a reagent; the condition key is built from these
   reagents. This replaces ADR 001's "agents and solvents stay as the source gives them". Source role
   columns stay in the ingest table for comparison only.
4. **Reaction classes are configuration.** One YAML per class: two substrate roles (SMARTS with the
   reacting atom), the bond formed between them, `allowed_dynamic_bonds` (amide: `[2]`), the
   single-reactive-group SMARTS and condition families. Adding a class adds a file, not code.
5. **The CGR check filters positives only.** `classify` records the number of dynamic bonds for every
   reaction; the positives stage keeps only allowed values. Benchmark reactions are never removed for
   mapping quality: their roles and outcomes are given, so the count is recorded as a check of the
   mapper only.

## Consequences

- One code path for P, U and TEST, as the charter requires; a new source needs only an adapter.
- RXNMapper needs a GPU for the full corpora (~20 M reactions); full runs happen on the cluster in
  shards, sample runs locally on CPU. Tests use a fake mapper and a small fixture mapped once for real.
- Some valid reactions are lost when RXNMapper maps them wrongly (2 of 137 in the ORDerly sample). I
  accept this: clean positives matter more than more positives. The positives report shows which
  substrates the filter removes, so a systematic loss (e.g. acids that already carry an amide) is seen.
- The CGR count used here (formed + broken bonds) is not the "chemical distance" of Chen et al.
  (*WIREs Comput. Mol. Sci.* 2013, 3, 560), which also counts bond-order and charge changes. For
  couplings this is an advantage (an acid and its carboxylate both give 2); classes where bond orders
  change will need the order changes too.
- rxnutils stays the reaction-level layer (ADR 001 point unchanged); our own code fills only the gaps:
  canonical reaction SMILES and cache, class engine, molecule standardisation, positives, labels.

## Alternatives considered

- **Keep each source's mapping and role columns** (ADR 001): not possible for unmapped sources (Chemlex,
  HiTEA, CJHIF), and the source mapping has its own errors.
- **LocalMapper** (slightly more accurate on the Golden set, SynRXN 2026): needs DGL, which has no
  wheels for current Python versions, and rxnutils has no wrapper for it. Revisit if RXNMapper fails on
  a class.
- **Consensus of several mappers**: +0.3 % accuracy for 30× the time (Lin 2022).
