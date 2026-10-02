# 001 — rxnutils is the reaction-standardisation layer

- **Status:** Accepted by Luka, 2026-10-02 · pending Jan's review (PR #1; merged there from PR #2)
- **Supersedes:** — · **Superseded by:** —

## Context

Every literature source (ORDerly, USPTO, Pistachio, …) has to be cleaned the same way before reactions
are classified and counted: molecules RDKit cannot read, products that also appear as reactants,
reagents written on the reactant side, and a stable identity for each reaction so duplicates can be
found. The rebuild plan (`docs/design/rebuild.md`) planned to compare rxnutils with chython first.

The ORDerly amide experiment (LAB_JOURNAL, 2026-10-02) ran the rxnutils pipeline on all 625,697 ORDerly
rows and compared the amide condition-bin count before and after:

- the counts per condition did not change (top 20 bins identical; HATU + DIPEA | DMF = 1,375 unique
  reactions either way);
- `reactants2reagents` moved non-reacting molecules to the reagent side in 65 % of rows;
- `core_key` / `full_key` gave usable duplicate detection (40 duplicates inside condition bins);
- only 206 rows had a molecule RDKit could not sanitise.

## Decision

rxnutils (`reaction-utils`) is the standardisation layer for every source. Each source adapter only
reads and reshapes; then every reaction goes through the same pipeline, `configs/rxnutils_std.yml`
(run with `scripts/run_rxnutils.py`):

trim → remove_unsanitizable → remove_unchanged_products → reactants2reagents →
remove_extra_atom_mapping → counts and mapping stats → remove_atom_mapping → core_key / full_key.

Desalting and neutralising stay **off**: RDKit's SaltRemover would delete all-salt reagents (NaOH).
Agents and solvents are never rebuilt from the reaction SMILES; they stay as the source gives them and
are joined back by row ID.

## Consequences

- One pipeline for P, U and TEST data, as the charter requires.
- `reaction-utils` needs pandas < 3 and Python < 3.14: the project is on pandas 2.3 and
  `requires-python = ">=3.13,<3.14"`. It also pulls in ~55 packages (metaflow, boto3, dask, …).
- rxnutils needs atom-mapped input; unmapped sources must be mapped first (rxnutils `rxnmapper` action).
- Gaps that our own steps must fill (stage 2 in `rebuild.md`): covalent metal notation (`O[Na]`,
  `[Na]OCC`) is not normalised; salt rules for substrates vs agents; tautomers.
- Wrong atom mapping in the source becomes a wrong role (seen: DMF kept as reactant, TFA/fumarate
  counter-ions kept as reactants). This must be checked per source, not trusted.
- Full ORDerly takes ~26 min on one core; `--max-workers` has no effect for these actions.

## Alternatives considered

- **chython**: not tested. Dropped because rxnutils already does what we need on real data and keeps
  us in RDKit, which the rest of the code uses.
- **Own RDKit-only code**: rejected; the charter prefers established libraries over custom code.
