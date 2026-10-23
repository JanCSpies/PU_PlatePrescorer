# Rebuild plan — one pipeline for every source and reaction class

Status: draft · 2026-10-01 · restructured 2026-10-09 (ADR 002)

## Goal

Turn any literature reaction file into positives `P`: pairs of substrates per reaction class × condition
family, every molecule in one consistent form, deduplicated across sources, with provenance. HTE sets
(Chemlex, HiTEA) go through the same stages and become benchmark labels; they are never training data.
The model side (unlabeled space `U`, features, models, plate design) starts from these tables.

## Principles

- **Only the adapter depends on the source.** Everything after ingest is the same code for every source
  (ADR 002). A new source = one adapter; a new reaction class = one YAML file.
- **Every class is a two-substrate coupling** (an HTE plate is a matrix A × B): role A + role B →
  product, with one bond formed between the two reacting atoms.
- **Stages talk through tables** (CLAUDE.md §5). Each stage reads one parquet table and writes
  `X.parquet`, `X.rejected.parquet` (with a reason per row) and `X.report.json` (rows in/out, rejects
  by reason, git commit, sha256 of each config, input size and time, rxnutils/RDKit/rxnmapper versions).
- **rxnutils is the reaction layer** (ADR 001), **RXNMapper maps everything** (ADR 002); our code only
  fills the gaps.
- **Licensed and internal data** (Pistachio, the group inventory, and everything derived from
  them; `licence` `licensed` or `internal`, ADR 004) are written only under `private/`;
  `io.write_table` refuses anything else.
- **Benchmarks are never filtered** for mapping quality; their mapping checks are recorded only.

## Stages

```
1 ingest       adapter (source-specific) → 1_records          record_id = "source:source_id"
2 rsmi         canonical unmapped "all molecules>>products" + hash → 2_rsmi
─ map          global: hashes missing from the cache → shards → RXNMapper (GPU) → cache
3 mapped       join with the cache; failure = confidence 0 or no atom map → 3_mapped
4 clean        rxnutils pipeline (configs/clean.yml) → 4_clean
5 classify     class YAMLs: formed bond → two substrates → roles; records dynamic bonds → 5_classified
6 standardise  molecules, keys, condition key from the reagent side → 6_standardised
7 outputs      positives (filters, families, dedup across sources, leakage counts)
               benchmark labels (per-benchmark rule)
```

| stage | code | config | output (under `derived/<source>/`) |
|---|---|---|---|
| 1 ingest | `ingest/<source>.py`, `scripts/ingest.py` | — | `1_records` |
| 2 rsmi | `reactions/smiles.py`, `scripts/rsmi.py` | — | `2_rsmi` |
| map | `reactions/mapping.py`, `scripts/map_shard.py` | `paths/<machine>.yml` | `derived/cache/rxnmapper-<version>/part-*.parquet` |
| 3 mapped | `reactions/mapping.py`, `scripts/mapped.py` | — | `3_mapped` |
| 4 clean | rxnutils, `scripts/clean.py` | `clean.yml` | `4_clean` |
| 5 classify | `classify/engine.py`, `scripts/classify.py` | `classes/<class>.yml` | `5_classified` |
| 6 standardise | `standardise/molecule.py`, `scripts/standardise.py` | `standardise.yml` | `6_standardised` |
| 7 positives | `positives/build.py`, `scripts/positives.py` | `classes/<class>.yml` | `derived/positives/` (+ `private/`) |
| 7 benchmark | `benchmarks/labels.py`, `scripts/benchmark.py` | `benchmarks/<name>.yml` | `derived/private/benchmarks/<name>/` (ADR 009) |

`scripts/run_pipeline.py --source X --paths configs/paths/<machine>.yml` runs stages 1–6 for one
source; mapping misses are written as shards for the cluster. Paths come from arguments or
`configs/paths/{local,cluster}.yml` (data roots, output root, RXNMapper interpreter); never from code.

## Stage 1 — ingest

**`ReactionRecord`** (frozen dataclass, `src/platepu/ingest/record.py`):

| field | type | meaning |
|---|---|---|
| `source` | str | file name key, e.g. `"orderly"` |
| `source_id` | str | ID in that file (ORD id, patent:paragraph, row number) |
| `licence` | str | `"open"`, `"noncommercial"` (Chemlex, ADR 009), `"licensed"` (Pistachio) or `"internal"` (the inventory, ADR 004) |
| `reactants` | tuple[str, ...] | SMILES as given by the source |
| `agents` | tuple[str, ...] | catalysts and reagents as given (empty if the source pools them) |
| `solvents` | tuple[str, ...] | solvents as given (empty if the source does not separate them, e.g. USPTO) |
| `products` | tuple[str, ...] | SMILES as given |
| `source_rxn` | str \| None | the source's own (mapped) reaction string, kept for checks only (was `mapped_rxn`; ADR 002) |
| `temperature_c` | float \| None | |
| `reaction_time_h` | float \| None | reaction time in hours |
| `yield_pct` | float \| None | as given, may be above 100 |
| `yield_kind` | str \| None | what `yield_pct` measures: `"isolated"` (literature; assumed, patents sometimes report HPLC or crude yields), `"lcms_uv_ratio"` (HTE conversion from LC-MS UV peak areas), or `None` when there is no number. A label only: the two kinds are never put on one scale |
| `year` | int \| None | year of the record's date (for patents: grant year, not experiment year) |
| `source_document` | str \| None | the source's own grouping of records, kept to split by time or document: ORDerly's `extracted_from_file` (an ORD dataset file; one grant year each, a time block of the patent corpus, not a patent; counted 2026-10-20: 489 values, median 707 reactions). `None` for Chemlex. ADR 006, item 4 |

`record_id = "source:source_id"` is the key through every later table. `records_to_table` /
`table_from_records` convert records to parquet with an explicit schema (tuples → `list<string>`,
missing → null), tested by a round trip.

Rules: missing values are `None`, never `""`, `"empty"` or `"nan"`. The adapter does **no chemistry**:
it only reads and reshapes. A row it cannot read becomes a rejected row with a reason. Source role
columns (`agents`, `solvents`) are kept for comparison only: roles used downstream come from the mapping
(ADR 002). Every adapter has the signature `read_x(path) -> (records, rejected)`; the `ADAPTERS` dict
lives in `scripts/`, not in the package.

**ORDerly specifics** (decided 2026-10-01, from the 2,000-row sample):
- `source_id` = the ORDerly row `index` as a string (unique in the sample; check on the full file).
- `solvents` = `solvent_000/001`; `agents` = `agent_000`–`agent_002`, as given, even when wrong (Et₃N
  listed as solvent, work-up NaHCO₃/EtOAc listed as agent/solvent).
- Salts arrive split into ions, each ion listed once (Na₂CO₃ → `O=C([O-])[O-]` + one `[Na+]`;
  ammonium acetate → `[NH4+]` as reactant, acetate as agent). The full stoichiometry is still in
  `source_rxn` (`rxn_str`).
- `year` = `grant_date.year`. `date_of_experiment` is empty in all sample rows, and the month in
  `grant_date` is garbled (stored in the minutes field), so only the year is used.
- `yield_000` above 100 % (80 of 2,000 rows) is kept as given; a later stage filters it and reports
  the reason.
- `reaction_time_h` = `rxn_time`, as given. ORDerly uses conventions here too ("room temperature" →
  25 °C, "overnight" → 8 h) and misses some times stated in the procedure (row 167959: "for 3 hours",
  `rxn_time` empty). Atmosphere (N₂, Ar) is not captured.
- Verified by 5 hand-checked rows in `tests/data/orderly_5rows.parquet`.

**Chemlex specifics** (benchmark; Zhong et al., *Nat. Commun.* 2025, 16:4522): 11,669 measured
acid–amine couplings in DMF, failures included; read in place from
`Datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx` (one sheet).
- `load_chemlex(path)` reads the sheet (0-based row index = `source_id`); `records_from_table(table)`
  and `splits_from_table(table)` are pure, so tests run on a fixture with the original row index.
- `reactants=(Acid, Amine)`, `agents` = `Reagents` split on `"."`, `solvents=(Solvent,)`,
  `products=(Products,)`, `yield_pct` = `Conversion`, `yield_kind="lcms_uv_ratio"`; no source mapping,
  temperature, time or year. Both HATU drawings and PyBrOP's mis-charged `F[P+](F)(F)(F)(F)F` stay as
  written.
- Checked 2026-10-09: no missing value in any column except `Stratified_Split_Both_Unseen` (4,475),
  conversion always 0–100, no multi-fragment acid, amine or product, so no rejected rows: a missing
  value or a conversion outside 0–100 raises `ValueError`. The 116 exact duplicate rows are all 0 %
  conversion: replicates that both gave nothing.
- **Splits** live in a side table keyed by `source_id` (`random`, `one_unseen`, `both_unseen`; values
  `"train"`, `"test"` or `None`). The random split puts replicates of one reaction on both sides in 212
  of 505 replicate groups (221 HATU reactions); both stratified splits never do. Results are reported
  on the both-unseen split.
- Verified by 6 hand-written rows in Chemlex's layout, `tests/data/chemlex_like_6rows.parquet`
  (2026-10-20; they replace 6 rows cut from the real file, so no Chemlex data is in the repository).

Checked 2026-10-09: 625 k ORDerly records build and write in 9.8 s, so adapters read whole files
(Pistachio, 7.2 M, about 2 min).

## Stage 2 — canonical reaction SMILES

`reactions/smiles.py`: every molecule of the record (reactants, agents, solvents) goes left of `>>`,
products right; each molecule is parsed and written as RDKit canonical SMILES (no other change, so this
is not standardisation), each side sorted. `rsmi_hash` = sha1 of that string, the mapping cache key.
Unparsable molecule → rejected (`unparsable`). Output `record_id, rsmi, rsmi_hash`. To count first:
USPTO `|f:|` groups (strip and count) and unparsable rows per source.

## Mapping — global, cached

- `rxnutils/data/mapping.py` (rxnmapper 0.4.3) is called directly with the RXNMapper environment's
  Python (3.12, torch, transformers, `setuptools<81`, rxnutils installed without dependencies), so its
  `confidence`, `--batch` and `--mapper_batch_size` are kept.
- The cache is append-only files `derived/cache/rxnmapper-<version>/part-*.parquet` with
  `rsmi_hash, rsmi, mapped_rsmi, confidence`; licensed reactions go to
  `derived/private/cache/…`. Lookup: open cache, then private.
- `scripts/map_shard.py` maps one shard (one SLURM array task) and writes one new part file; no file is
  ever edited, so parallel tasks are safe.
- `to_map(stage2_tables, cache)` lists the reactions not yet in the cache, and
  `map_reactions(items, mapper)` maps them with the mapper as an argument: tests use a fake; a small fixture of
  open rows is mapped once for real and kept in `tests/data/`; one test marked `rxnmapper` runs only when
  `RXNMAPPER_PYTHON` is set.
- Failure = `confidence == 0` or no atom map in the output (the script then returns the input
  unchanged); counted per source. Token length (RXNMapper's limit is 512 tokens) is not recorded
  yet: 15 failures in 625,664 ORDerly reactions did not call for it.
- Verified on 500 Chemlex rows (2026-10-09): 0 failures, 500/500 roles, ~20 reactions/s on CPU.

## Stage 4 — clean (rxnutils)

`configs/clean.yml` (replaces `rxnutils_std.yml`): `remove_unchanged_products` → `reactants2reagents`
→ `remove_extra_atom_mapping` → `count_components` → `product_atommapping_stats` →
`cgr_dynamic_bonds`. Counted 2026-10-09: `remove_unsanitizable` changed nothing after stage 2, and
reaction keys come later from standardised molecules, so both are left out. Rejects: `no_product`,
`no_reactant`, `no_cgr`; one-reactant reactions go on (classify rejects them as `no_formed_bond`). Desalting and neutralising stay off (ADR 001). After `reactants2reagents`, every
molecule that gives no atom to the product is a reagent (Schneider; Gimadiev 2021): this is where roles
and the reagent side come from for every source.

`cgr_dynamic_bonds` counts bonds formed + broken (not bond-order or charge changes). Verified
2026-10-09: 498/500 Chemlex at 2, the 2 mis-mapped ones at 4. Also to count here (Gimadiev 2021):
reactions with radicals and with isotopes.

## Stage 5 — classify

One YAML per class in `configs/classes/` with:
- two roles, each a SMARTS with the reacting atom marked (charge-tolerant, because classify runs before
  standardisation: carboxylates, ammonium salts);
- `formed_bond: [role_a, role_b]`;
- `allowed_dynamic_bonds` (amide coupling: `[2]`);
- `positives.single_reactive_group`: whether stage 7 counts the role SMARTS on the standardised
  substrates and keeps only one group of its own role and none of the other (`classify/groups.py`,
  also used by the lab library);
- condition families, each "condition key contains reagent X" (e.g. `HATU_DIPEA_DMF`).

Engine: take the bond formed between atoms of two different reactant molecules in the mapped reaction;
each role SMARTS must match its molecule at that atom. Output one row per (reaction, class match):
`record_id, class, substrate_a, substrate_b, n_dynamic_bonds, n_products`. Rejects, first reason that
applies: `unparsable` (a molecule RDKit cannot read), `no_formed_bond` (no bond between atoms of two
reactant molecules: one-substrate and intramolecular reactions, e.g. lactams), `no_class` (a formed
bond no class fits). Each reaction is parsed once. A reaction matching two classes keeps both rows
(full ORDerly: 38,459 rows from 38,454 reactions); positives decide by config. Nothing is filtered by
dynamic bonds here.

Decided 2026-10-09 from the counts: the amine role is a true amine only (N-H not on C=O, C=S, C=N,
SO2, C#N, N or O). Not amide coupling: Chemlex's sulfonamides, amidines/guanidines, thioamides (1,697
rows) and hydrazines/hydroxylamines (589 rows; hydrazides, hydroxamates, Weinreb amides). Benchmarks do
not take class or roles from the mapping: their roles come from their own columns and the class is
checked with the role SMARTS on the given molecules (stage 7b); the mapping result is a check only.

Verified by cases (acid + amine, carboxylate, ammonium, sulfonamide, Weinreb, hydrazide, lactam) and on
samples: ORDerly 141 couplings (121 at 2 dynamic bonds); the old crude rule's 137 differ by 6
non-couplings it counted (reductive methylation, alkylation, Boc protection, oxadiazole, imide) and
NH4+ couplings it missed. Chemlex 9,370 (9,356 at 2; 6 mis-mapped ones are `no_class` here).

## Stage 6 — standardise

**Scope now: amide couplings.** Built only for what the data has (counted 2026-10-09):

| question | Chemlex | ORDerly | so |
|---|---|---|---|
| multi-fragment SMILES (`.`) | 0 | 0 in all 8 columns | no step needed for reactions (the inventory needs one: `std_version` 2, below) |
| unparsable | 0 substrates; 7 agents (PyBrOP `F[P+](F)(F)(F)(F)F`) | 0 of 379 agent/solvent strings | fail with a reason, counted |
| HATU drawing | O-form 7,242, N-oxide 1,212 | O-form only (3,560 rows) | one synonym, O-form → N-oxide |
| DIEA vs DIPEA | `CCN(C(C)C)C(C)C` | same SMILES | no synonym |
| carboxylate as the acid (amide product + amine) | 2 charged acids | 1,181 (+12 with ammonium) | neutralise substrates |
| ammonium as the amine | 3 charged amines | 196 | neutralise substrates |
| stereo in substrates | 101 acids, 53 amines | common | two keys |
| covalent alkali/Mg/Zn | 0 | ~24,000 entries overall, 2 in HATU rows | not now; when a class needs it |

`src/platepu/standardise/molecule.py` (tests in `tests/test_standardise.py`, written 2026-10-09):
- `standardize(smiles, role, config) -> StdResult`, `role` in `substrate`, `agent`, `solvent`. Parse,
  drop atom-map numbers; **substrates** are reduced to their largest organic fragment (`std_version` 2,
  2026-10-09: the inventory writes salts and `[H+]`; reactions never have multi-fragment SMILES, so
  the pipeline did not change) and neutralised with RDKit's `Uncharger`, called directly with the
  same steps as rxnutils' `neutralize_molecules` (2026-10-20: importing rxnutils.chem pulls in
  rdchiral, which switches RDKit's log off for the whole process) (carboxylate → acid,
  ammonium → amine, sulfonate/phenolate protonated; nitro, azide and quaternary N stay); **agents and
  solvents** get synonyms from `configs/standardise.yml` (HATU O-form → N-oxide, its real structure)
  and keep whole ions. `StdResult` carries the original SMILES, the canonical SMILES (or `None` with a
  reason: `unparsable`, `neutralise_failed`, `no_inchikey` for a dummy atom), `key_stereo` (full
  InChIKey), `key_flat` (first block, stereo-agnostic) and `std_version`. RDKit's log is blocked
  only while a data molecule is read (`BlockLogs`), never for the process.
- `required(result, record_id)`: a substrate or product of a classified reaction (stage 6) or of a
  benchmark (stage 7b) that fails standardisation stops the stage with the record id. Counted
  2026-10-20: none fails in full ORDerly (38,459 reactions) or Chemlex (23,338 substrates), so a
  failure is input not seen yet; when a new source brings real failures, count them and add a
  rejected table then.
- `in_family(condition, members)`: the one family rule for positives and benchmark splits (every
  member present, extras allowed); `condition_text(key)` writes a key as `"A + B"` or
  `"(unreadable)"`, `condition_from_text` reads it back (benchmark labels store the text).
- `condition_key(agents, solvents, config)`: the sorted set of the standardised reagents, counter-ions
  dropped (`Cl`, `[Cl-]`, PF6⁻, BF4⁻; config). In the pipeline it is built from the **reagent side of
  the mapped reaction** (ADR 002), so it is the same for every source. `None` when any part is
  unparsable.
- Not now: covalent metal salts, organometallics, tautomers (open), DIPEA ≈ Et3N.
  `[I-]` and `[Li+]` in ORDerly HATU rows (counter-ions of substrates, ~1 %) stay in the key for now.

Verified 2026-10-09: 14 hard-case tests pass; with the old crude amide rule on full ORDerly the new
`condition_key` gives HATU + DIPEA | DMF = 1,376 and the next bins 547, 387, 326, 301, exactly as in
LAB_JOURNAL 2026-10-02; the exploration scripts were then deleted (last at commit `7161120`). An empty
condition key means no reagent was reported (ORDerly sample: 62 of 141 couplings; 51 have no agents
or solvents at all, 11 only HCl); null means a reagent could not be read.

## Stage 7a — positives

`positives/build.py`, one function called twice: with open sources only (published) and with all
sources (written to `private/`). Per class and condition family:
1. keep reactions with `n_dynamic_bonds` in `allowed_dynamic_bonds`, exactly one product, and both
   substrates passing the single-reactive-group filter (on the standardised molecule);
2. yield cut-off: config `min_yield_pct: null | 10 | 20` (decided 2026-10-09: yield only stored so far;
   P sizes reported at each value before choosing);
3. deduplicate after standardisation: reaction level (class, family, substrate `key_stereo`s, product)
   and pair level (class, family, `key_A`, `key_B`) with provenance (record_ids, number of sources,
   first year, max yield) — this removes CJHIF's re-merged CHORISO/ORD/USPTO rows;
4. leakage counts against every benchmark (ADR 010): a substrate's `key_flat` in the benchmark, and the pair in
   the benchmark (Chemlex `both_unseen` is defined by substrates);
5. report which substrates the filters remove (e.g. share of acids with a second amide before and after),
   so a systematic loss is visible.

## Stage 7b — benchmark labels

`benchmarks/labels.py`, rule per benchmark in `configs/benchmarks/<name>.yml`. Benchmarks are never
filtered for mapping quality. **Chemlex** (measured 2026-10-09 on the 515 HATU replicate groups:
replicates that disagree are rarely near the cut-off; median gap 62 conversion points, in 70 of 101
conflicting pairs one replicate is exactly 0 %):
- replicates of one reaction are kept only when **all are 0 %** or **all are ≥ 10 %**; anything else is
  `inconsistent` and leaves the test set (HATU: 162 of 515 groups). Reactions measured once (`single`)
  are kept;
- a kept reaction gets the **mean conversion** and is **positive when the mean is ≥ 20 %** (the paper's
  cut-off; conversions only);
- `label_reactions(table, keys, rule)` returns one row per reaction with `n_measurements`,
  `conversion_mean`, `status` (`single`, `all_zero`, `all_above_min`, `inconsistent`) and `label`
  (missing for `inconsistent`); inconsistent reactions stay in the table and are counted.

Roles come from the benchmark's columns; the class is checked with `fits_roles` (role SMARTS on the
given molecules); molecules go through the same `standardize()`; a reaction is (substrate keys,
condition). The mapping is a check only (`n_mapped_in_class`). Splits listed in `reaction_splits` must
agree across a reaction's measurements; if they do not, the split is None and `split_conflict` is set.

Verified 2026-10-09 on the full Chemlex: the old HATU check (all N-nucleophiles) gives the same 8,454
rows and 28.3 % positive; reactions 7,898 instead of 7,923 because two amines written as two SMILES of
one molecule now merge (a sulfonate and its acid; acarbose). Class amide_coupling: 9,383 rows →
8,883 reactions → 8,735 labelled, 28.1 % positive; HATU + DIPEA + DMF 6,177, 30.9 %; both-unseen test
1,942, 39.2 %. Chemlex's own `one_unseen` split treats acarbose as two amines (one train, one test):
4 split conflicts. A noise ceiling as ROC-AUC is left for the baseline step.

## Work plan

Each step: count the cases in the data first, then tests, then code; verified on samples; LAB_JOURNAL
entry; commit only when Luka asks.

| # | step | done when |
|---|---|---|
| — | ORDerly adapter + `ReactionRecord`; rxnutils run (ADR 001); Chemlex adapter, splits, test labels | done 2026-10-01 … 10-09 |
| 0 | ADR 002; this document; CLAUDE.md §10 | done 2026-10-09 |
| 1 | `ReactionRecord` parquet round trip, `record_id`, `source_rxn`; `scripts/ingest.py` (ORDerly, Chemlex) | done 2026-10-09 (625,697 in 9.8 s) |
| 2 | `reactions/smiles.py`, `scripts/rsmi.py` | done 2026-10-09 (ORDerly 33, Chemlex 7 unparsable) |
| 3 | `reactions/mapping.py`, `scripts/map_missing.py`, `map_shard.py`, `mapped.py` | done 2026-10-09 (13,170 mapped, 0 failed; roles checked in step 5) |
| 4 | `reactions/clean.py`, `configs/clean.yml`, `scripts/clean.py` | done 2026-10-09 (Chemlex 11,642 / 11,662 at 2) |
| 5 | `classify/engine.py`, `classes/amide_coupling.yml`, `scripts/classify.py` | done 2026-10-09 (ORDerly sample 141, 121 at 2; Chemlex 9,370) |
| 6 | `standardise/molecule.py`, `standardise/table.py`, `scripts/standardise.py` | done 2026-10-09 (1,376 reproduced; exploration scripts deleted) |
| 7 | `benchmarks/labels.py`, `benchmarks/measurements.py`, `configs/benchmarks/chemlex.yml`, `scripts/benchmark.py` | done 2026-10-09 (8,735 / 28.1 %; old check 8,454 rows / 28.3 % reproduced; old script deleted) |
| 8 | `positives/`, `io.write_table`, leakage counts | done for open sources 2026-10-09 (full ORDerly: 28,158 pairs, 1,177 HATU_DIPEA_DMF); the private run waits for Pistachio |
| 9 | USPTO adapter, `run_pipeline.py`, `configs/paths/`, sharded mapping | ORDerly + USPTO samples deduplicated across sources |
| 10 | HiTEA adapter + `classes/buchwald_hartwig.yml` | end to end; allowed dynamic bonds set from HiTEA counts; licence of each HTE source in `docs/literature/notes/` (ADR 005; HiTEA rows were in the prototype's training corpus, the rebuild reads no prototype table) |

## Open questions

- Which salts and organometallics keep their counter-ion as substrates? (list for config)
- First plate class: amide/SNAr (open data suffices) or a cross-coupling (Pistachio needed early)?
- Yield cut-off for positives (none, 10 or 20 %): chosen in step 8 from P sizes.
- How exactly is Chemlex conversion computed (wavelength, denominator)? Check Zhong et al., Methods.
- Cluster: RXNMapper environment, GPU and SLURM setup for the mapping shards.
- Stage 4 runs ~150–280 reactions/s on one core (CGR via SDF): ~1–1.5 days for 20 M reactions, so it
  needs sharding on the cluster too.
- Allowed dynamic bonds for Buchwald–Hartwig: from HiTEA counts in step 10.
