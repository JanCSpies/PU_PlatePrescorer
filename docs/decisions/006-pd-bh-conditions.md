# 006 — Pd-catalysed BH only, and clean+unified conditions in mining

**Status:** decided 2026-07-08 (Florian + Jan). Guides the condition-aware phase. Companion to
`005-condition-encoding.md` and `bhpu.chem.conditions`.

## Decisions

1. **Mine Pd-catalysed Buchwald–Hartwig only.** When mining ORD/Pistachio, **require a Pd species**
   and reject Cu-catalysed (Ullmann/Chan–Lam), Ni, and metal-free SNAr couplings. Jan flagged that
   many product-confirmed aryl C–N couplings in the corpora are Cu-SNAr / nucleophilic aromatic
   substitution, **not** BH — mixing them pollutes condition learning. (The earlier corpus was *not*
   Pd-filtered; it was acceptable for substrate-only signal but must be Pd-clean for conditions.)

2. **Mine substrates AND conditions, then clean + unify them — essential for learning.** Every mined
   reaction must carry a **unified** condition (`bhpu.chem.conditions.condition_from_reagents`):
   metal(s) / ligand / base / solvent, where a precatalyst (RuPhos-Pd-G4) and metal+free-ligand
   (Pd + RuPhos) collapse to the **same** representation (validated). Without this, the same
   catalytic system reads as different features across sources and is unlearnable.

3. **The unlabelled U must span conditions.** The JnJ-derived grid U is no longer substrate-only ×
   4 general conditions; it is unmeasured pairs × a spread of real condition sets (bigger, but that
   is the point — more to learn). See `scripts/17` `_build_condition_grid` (to be generalised).

4. **Re-test the descriptor verdict *with* conditions.** The WS5 site-descriptor negative was
   measured **substrate-only**. Substrate descriptors (morfeus/g-xTB) may hold information only
   usable once conditions are present — so re-evaluate `substrate-descriptors ⊕ conditions` before
   treating descriptors as settled-useless. (RESULTS §WS5 carries this caveat.)

## Next-step plan (the "kick off")

1. **Miner extension** (`bhpu.data.mining`): `detect(..., require_pd=True)` + attach the unified
   `CanonicalCondition`; emit **reaction-level** `(aryl, amine, ligand, base, solvent, yield?)`
   records (dedup by pair+condition), keeping the pair-level path for the substrate-only corpus.
   Provenance: Pd-filter counts, unique ligands, JnJ-ligand overlap.
2. **Re-mine** ORD (open) + Pistachio (licensed → `datasets/processed/private/`) with `require_pd`
   and conditions (SLURM; `--nice=1000000`).
3. **Condition-aware transfer** (`scripts/17` generalised): P = mined Pd-BH reactions with unified
   conditions; U = JnJ grid × condition spread; features = substrate ECFP ⊕ unified condition
   (ligand/base/solvent ECFP + metal). Now P and JnJ condition spaces **overlap** (ORD covers 47% of
   JnJ ligands; 73% of ORD ligand-bearing couplings use a JnJ ligand), so conditions should help
   (recall +0.16 in-domain) instead of acting as a domain tag (WS6 crash).
4. **Descriptor re-test**: add substrate morfeus/g-xTB descriptors to the condition-aware model.
