# Rebuild plan — data side (ingest and standardise)

Status: draft for Luka and Jan · 2026-10-01

## Goal

Turn any literature reaction file into one common table of reactions, with every molecule in one
consistent form, so that later stages (annotate → catalogue → positives) never see
source-specific formats. End product of the data side: a positives table per reaction class ×
condition cluster, handed to the model side.

## Order of sources

1. **ORDerly** — roles already in columns, best conditions, open. Builds the method: record,
   adapter, tests, report.
2. **USPTO 1976–2016 / 2001–2016** — open stand-in for Pistachio: same `reactants>agents>products`
   format, atom-mapped, CXSMILES `|f:|` salt groups.
3. **Pistachio** — reuses the USPTO adapter; output only under `private/`; no Pistachio rows in tests.
4. Later, if needed: CHORISO (named conditions), CJHIF journal rows, ORD HTE sets.

Facts behind this order: `Reaction Corpus Atlas` (2,000-row samples, 2026-10-01).

## Stage 1 — ingest

**`ReactionRecord`** (frozen dataclass, `src/platepu/ingest/record.py`):

| field | type | meaning |
|---|---|---|
| `source` | str | file name key, e.g. `"orderly"` |
| `source_id` | str | ID in that file (ORD id, patent:paragraph, row number) |
| `licence` | str | `"open"` or `"licensed"` |
| `reactants` | tuple[str, ...] | SMILES as given by the source |
| `agents` | tuple[str, ...] | catalysts and reagents as given (empty if the source pools them) |
| `solvents` | tuple[str, ...] | solvents as given (empty if the source does not separate them, e.g. USPTO) |
| `products` | tuple[str, ...] | SMILES as given |
| `mapped_rxn` | str \| None | atom-mapped reaction string, if any |
| `temperature_c` | float \| None | |
| `reaction_time_h` | float \| None | reaction time in hours |
| `yield_pct` | float \| None | as given, may be above 100 |
| `year` | int \| None | year of the record's date (for patents: grant year, not experiment year) |

Rules: missing values are `None`, never `""`, `"empty"` or `"nan"`. The adapter does **no chemistry**:
it only reads and reshapes. A row it cannot read becomes a rejected row with a reason.

**ORDerly specifics** (decided 2026-10-01, from the 2,000-row sample):
- `source_id` = the ORDerly row `index` as a string (unique in the sample; check on the full file).
- `solvents` = `solvent_000/001`; `agents` = `agent_000`–`agent_002`. Roles are kept as given, even when
  wrong (Et₃N listed as solvent, work-up NaHCO₃/EtOAc listed as agent/solvent). Fixing roles is
  the job of annotate.
- Salts arrive split into ions, each ion listed once (Na₂CO₃ → `O=C([O-])[O-]` + one `[Na+]`;
  ammonium acetate → `[NH4+]` as reactant, acetate as agent). The adapter keeps the columns as they
  are; the full stoichiometry is still in `mapped_rxn` (`rxn_str`). Rebuilding salts is a
  standardise question (step 3).
- `year` = `grant_date.year`. `date_of_experiment` is empty in all sample rows, and the month in
  `grant_date` is garbled (stored in the minutes field), so only the year is used.
- `yield_000` above 100 % (80 of 2,000 rows) is kept as given; a later stage filters it and reports
  the reason.
- `reaction_time_h` = `rxn_time`, as given. ORDerly uses conventions here too ("room temperature" →
  25 °C, "overnight" → 8 h, the most common time in the sample) and misses some times stated in the
  procedure (row 167959: "for 3 hours", `rxn_time` empty). Atmosphere (N₂, Ar) is not captured.

**One adapter per source** (`src/platepu/ingest/orderly.py`, …): a function that takes a file path
and yields `ReactionRecord`s plus rejected rows.

**Verified by**
- 5 hand-checked ORDerly rows in `tests/data/` (an amide coupling, a cross-coupling, a row without
  temperature, a row without yield, a row with a salt) → expected records written by Luka.
- A report for the 2,000-row sample: rows in, records out, rejected by reason.

## Stage 2 — standardise

`standardize(smiles, role) -> str | None` plus a reason when it fails.

- Covalent alkali salts become ionic (`O[Na]` → `[Na+].[OH-]`).
- **Agents keep the whole salt** (Cs2CO3 ≠ K2CO3).
- **Substrates** get their parent form (largest organic fragment, neutral), except reactive salts
  and organometallics (Grignard, organozinc, BuLi, trifluoroborates; list in config).
- Tautomers unified for identity; the original SMILES is kept alongside.
- Output carries `std_version`. Same function for P, U and TEST.

**Verified by** hard-case tests (from `standardiser_bench.py`: NaOH covalent = ionic, K2CO3 ≠ Cs2CO3,
Pd(OAc)2, pyridone = hydroxypyridine, C–metal bonds kept) and a grid image of 20 changed molecules.

## Work plan

| # | Commit topic (on `luka/rebuild-clean-slate`) | Work | Done when |
|---|---|---|---|
| 1 | `ingest-orderly` | `ReactionRecord` + ORDerly adapter + sample report; tests first | 5 hand-checked rows pass; report reviewed with Jan |
| 2 | — (`scratch/`, not committed) | Try rxnutils and chython on adapter output from ORDerly **and** the Pistachio sample | numbers for what each changed or dropped; 20 drawn examples per tool |
| 3 | `decision-standardise` | Decision record `docs/decisions/001-standardisation.md`, after Matt's and Almaz's replies | Jan and Luka agree |
| 4 | `standardise` | `standardize()` per the decision; hard-case tests first | all hard cases pass; RDKit round trip on all sample rows |
| 5 | `ingest-uspto` | USPTO adapter (`a>b>c`, `\|f:\|` groups) | hand-checked rows pass; report reviewed |
| 6 | `ingest-pistachio` | Pistachio adapter, output under `private/` | sample report; no Pistachio data committed |

After step 6: annotate (roles from mapping, reaction class, conditions as substance + family).
That stage gets its own section here before it is built.

## Open questions

- Which salts and organometallics keep their counter-ion as substrates? (list for config)
- rxnutils pipeline on top, or only individual functions, or chython alone? (step 3)
- First plate class: amide/SNAr (open data suffices) or a cross-coupling (Pistachio needed early)?
  — Jan
- Definition of a positive: reported at all, or yield above a cut-off? — later, before positives.
