# Catalog of literature reaction-data sources for mining C–N (Buchwald–Hartwig) positives

Scope: candidate sources to supply **literature POSITIVES** (and, where available, measured negatives) for PU learning of BH C–N coupling feasibility. Test set = **JnJ HTE 2024** (Neves et al.). Everything else in the combinatorial substrate space is **UNLABELED**. Sizes/coverage below are from the sources' own reports and from direct inspection of the local files.

## What we already have locally (inspected 2026-07-07)
- `datasets/raw/Neves/BH_HTE_Curated_Dataset_v20260219.csv` — **27,497 rows, 44 cols**, the Neves *curated* BH dataset. Columns include `Source, Aryl SMILES, Amine SMILES, Catalyst Name/SMILES, Base, Solvent, Product SMILES, DRFP, DRFP_subst_prod, QM, Yield, labels (0/1), iteration_0..9 cluster/train_test`. Per-source breakdown (rows; label=1 positives / label=0 negatives):
  - **JNJ HTE 2024** 11,328 (4,432 / 6,896) ← **TEST SET**, ~39 % feasible; 110 aryl × 68 amines; 32 Pd pre-catalysts; bases Cs2CO3/NaOtBu/LiHMDS/P2Et/NaBHT/BTMG/K3PO4/NaOTMS; solvents DMSO/dioxane/DMA/t-AmOH/2-MeTHF/MeCN.
  - **Dreher Doyle 2018** 4,312 (3,069 / 1,243) — 15 aryl × 1 amine (Ahneman isoxazole-additive set).
  - **Merck 2025** 4,204 (1,796 / 2,408) — 357 aryl × 348 amines (broad substrate scope).
  - **Roche 2023** 3,359 (1,477 / 1,882) — 37 aryl × 32 amines.
  - **HiTEA 2024** 2,632 (505 / 2,127) — 24 aryl × 21 amines (= King-Smith Nat. Chem. "reactome").
  - **Dreher 2014** 768 (54 / 714); **AZ ELN 2023** 750 (591 / 159); **Dreher 2016** 144 (41 / 103).
- `datasets/raw/Cernak/50K_dataset_v2.csv` — **50,688 rows** (Regan & Cernak). Condition screen: ~**4 aryl halides** (3-Br/3-I-pyridine, 4-I/4-Br-benzotrifluoride) × **2–3 amines** (morpholine, 4-phenylpiperidine) × **166 ligands** × Pd/Cu/Ni metal sources × ~18 bases × 5 solvents. Yields present; **~71 % of scored wells are 0 % (failed)** → dense measured-negative source. Plus `morpholine_yield_v1.csv`, `piperidine_yield_v1.csv`.

**Critical, already-measured leakage fact:** the non-JnJ curated sources share **≈0 substrates** with the JnJ test set (Merck 0/357 aryl & 0/348 amine; AZ 0/296 & 0/229; HiTEA 0/24 & 0/21; Roche 0/37 & 0/32; only Dreher/Doyle shares 1–2 aryl halides, 0 amines). Neves *built* the curated set to be substrate-disjoint (unique-structure-per-entity + DRFP/K-means OOD split). So our added literature sources must be held to the **same substrate-disjoint standard**.

## Comparison table

| Source | Approx size | C–N / BH coverage | Access & license | Leakage risk vs JnJ | Recommended use |
|---|---|---|---|---|---|
| **USPTO (Lowe)** | ~1.8 M reactions (US grants+apps 1976–2016); ~480 k–853 k after cleaning | Large; BH/C–N isolable via NameRxn class 10.1.x or RXNMapper + aryl-C–N SMARTS. Text-mined, conditions noisy/incomplete | **Public / open** (US patents; figshare 10.6084/m9.figshare.5104873; RDF/SMILES). Free, redistributable | **Low–Med** — patent substrates ≠ JnJ HTE scope, but must hash-audit; some HTE papers also patented | **Primary positives baseline** — largest free, diverse; positives only |
| **ORD (Open Reaction Database)** | **~2.4 M reactions** | Substructure SMARTS query for C–N; contains Ahneman BH (3,955) **and a ~74 k C–N HTE subset** (aryl halide × additive × Pd × base) | **Open**: data **CC-BY-SA 4.0**, schema/code Apache-2.0; GitHub `open-reaction-database`, structured protobuf | **Med** — *contains the very HTE sets already in our curated file* (Dreher/Ahneman/AZ). Must exclude those to avoid double-count/contamination | **Positives + structured conditions**; de-dup against curated file first |
| **Pistachio (NextMove)** | **~13.3 M** (2025 update) patent reactions | Pre-classified NameRxn C–N/BH classes; cleaner atom-mapping than raw USPTO | **Commercial** (Docker license); J&J likely has in-house; Münster/academia often not | **Med** — patent-derived like USPTO | **Optional, if licensed** — richer/cleaner superset of USPTO; positives |
| **Reaxys** (Münster) | tens of millions of reactions | Excellent C–N/BH filtering by reaction class; curated conditions/yields | **Licensed (Elsevier)**; **ToS forbid bulk ML training / redistribution of extracts** | Cannot externally audit or publish extracts | **Validation counts only**, *not* publishable training data 🔒 |
| **SciFinder / CAS** (Münster) | very large | C–N reaction search | **Licensed (CAS)**; same ML/redistribution restrictions | as Reaxys | **Manual lookups only** 🔒 |
| **Cernak 50,688 (Regan & Cernak)** | **50,688 rows**; ~4 aryl × 2–3 amine × 166 ligand × Pd/Cu/Ni | Deep *condition* screen for a few substrates; **~29 % feasible / ~71 % measured 0 %** | Local copy present; JACS 2026 (10.1021/jacs.6c05959) 🔒 + open preprint (ResearchSquare rs-7603680 / ChemRxiv) | **Low on substrates** (only ~4 aryl/2–3 amine — audit those exact SMILES vs JnJ) | **(1) Positives** for its combos; **(2) GOLD measured negatives** to validate PU reliable-negatives; **(3) condition prior** |
| **Bundled non-JnJ HTE** (Merck 2025, Roche 2023, HiTEA 2024, Dreher '18/'14/'16, AZ 2023) | **16,169 rows** (27,497 − JnJ 11,328); labeled 0/1 | In-domain BH, broad substrates; **has explicit negatives** | Already in repo; mixed open + industrial licenses | **Low** — substrate-disjoint from JnJ by construction | **In-domain positives (label=1)** + **PN negatives / PU validation** |

## How to tag / extract C–N (BH) reactions per source
- **NameRxn (NextMove) reaction classes:** BH aminations sit under superclass **10 "N-substitution / N–C bond formation"**, e.g. `10.1.1` (chloro-N-arylation), `10.1.2` (bromo-), Buchwald–Hartwig specific subclasses — the cleanest programmatic filter for USPTO/Pistachio/ORD.
- **Fallback SMARTS/atom-map filter:** reaction where a new bond forms between an aromatic carbon bearing a leaving group (Cl/Br/I/OTf) and an amine N, catalyzed by Pd/Cu/Ni; verify with RXNMapper atom-mapping.
- **HTE sources** come pre-structured (aryl/amine/catalyst/base/solvent columns) — no mining needed, only canonicalization + label thresholding (>1 % yield = feasible, per Toniato).

## Recommended integration order for the source benchmark (positives)
1. **USPTO C–N positives** — free, largest, most diverse → baseline PU model.
2. **+ ORD C–N positives** (minus the HTE sets already in curated) — adds structured conditions.
3. **+ bundled non-JnJ HTE label=1** (Merck/Roche/HiTEA/Dreher/AZ) — in-domain BH positives, substrate-disjoint from JnJ.
4. **+ Cernak positives** — dense condition coverage for a few substrates.
5. **Pistachio** (if licensed) as a USPTO upgrade; **Reaxys/SciFinder** for validation counts only, never in the publishable training set.

Report each increment's effect on JnJ feasibility prediction → this *is* the source benchmark. Also report each source's **bias profile** (Jia-style amine/aryl Gini + Pareto curve) and its **measured-negative recall** (do PU-inferred negatives match the source's real 0 % wells).

## No-leakage protocol (applies to every source before it enters training)
1. **Canonicalize** each component: RDKit `MolToSmiles(canonical=True)`, strip atom maps/salts/isotopes, neutralize; compute **InChIKey** per aryl and per amine.
2. **Reaction hash:** DRFP (reuse the curated file's `DRFP` column recipe) and/or reaction-InChI (RInChI); or sorted tuple of (aryl_InChIKey, amine_InChIKey [+ catalyst/base/solvent for condition-level]).
3. **Substrate-level exclusion:** drop any literature reaction whose aryl **or** amine InChIKey ∈ JnJ substrate set (JnJ is an OOD *substrate* test — same substrate under different conditions still leaks).
4. **Reaction-level exclusion:** drop exact hash duplicates vs JnJ and cross-source duplicates (ORD ⊃ Dreher/Ahneman, HiTEA appears twice, etc.).
5. **Near-duplicate audit:** DRFP + Tanimoto; flag/report literature reactions with high similarity to any JnJ test reaction as a leakage metric.
6. **Cluster audit:** reuse Neves' DRFP + K-means clusters (the `iteration_N cluster` columns) — ensure no training reaction sits in a cluster assigned to the JnJ test fold.
7. **Temporal guard (Segler/Sheridan):** prefer sources dated before the JnJ 2024 campaign so JnJ chemistry cannot leak backward into patents.
