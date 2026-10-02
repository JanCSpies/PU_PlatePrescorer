# Lab journal

Append-only. Newest entry on top; never rewrite an old entry. Each entry: context → done → result →
interpretation → next. Every number can be regenerated with the commands in its entry.

---

## 2026-10-02 — Amide-coupling condition bins in ORDerly, raw and after rxnutils — Luka

**Context.** Before building reagent-role assignment we need one number: how many amide couplings
ORDerly has under each exact condition combination (agents + solvents). Decision rule set in advance:
main bins with several hundred reactions → one bin can be one positive set, no roles needed; only tens
→ bins must be merged, and merging needs roles. Second question: does putting rxnutils (reaction
standardisation) in front change the count, and how many duplicates are there?

**Data.** ORDerly condition, train split: `~/University/Phd/Dataset/Datasets/orderly_condition_train.parquet`,
625,697 rows (test split not on disk). Salts are always split into separate agents (0 of 704,249 agent
values contain a dot); empty values are NaN; every molecule always gets the same role (0 molecules in
both agent and solvent columns); liquid bases (DIPEA, Et3N) are always filed as solvents.

**Done.**
- `scripts/explore_condition_bins.py`: crude amide rule (RDKit SMARTS, no atom mapping: carboxylic acid
  `[CX3](=O)[OX2H1]` + amine `[NX3;H2,H1;!$(NC=O);!$(NS=O)]` → product with `C(=O)[NX3]`; acid chlorides
  excluded); key = (sorted agents, sorted solvents) as ORDerly assigns them, SMILES canonicalised,
  atom maps stripped; counter-ions dropped from keys (Cl, [Cl-], PF6-, BF4-; so HBTU and TBTU share a key);
  reactions without agent or solvent excluded.
- rxnutils (reaction-utils 1.9.4) pipeline `configs/rxnutils_std.yml` run on all rows via
  `scripts/run_rxnutils.py` (reads ORDerly with `platepu.ingest.orderly.read_orderly`; rxnutils gets
  `row_id` + mapped reaction only; agents/solvents are joined back by `row_id`).
- `scripts/recount_condition_bins.py`: same rule and key on the raw ORDerly reactants and on the
  rxnutils reactants (after `reactants2reagents`); duplicates inside a bin removed by rxnutils `core_key`.

**Result.**

| step | raw | rxnutils |
|---|---|---|
| all rows | 625,697 | 625,697 |
| two reactants + product | 471,478 | 462,923 |
| amide coupling | 37,735 | 37,757 |
| binned | 15,985 | 15,990 |
| binned, duplicates removed (core_key) | 15,945 | 15,950 |

- Largest bin **HATU + DIPEA | DMF: 1,376 reactions, 1,375 after dedup.** Next: HBTU/TBTU + DIPEA | DMF
  547; HATU + DIPEA | DCM 387; EDC + HOBt | DMF 326; EDC | DCM 301. In the first run (before the
  counter-ion fix) there were 2,184 bins, 13 of them with ≥ 200 reactions, and the top 20 held 35 % of
  binned reactions.
- Duplicates: 40 inside bins (0.25 %). Whole file: 603,113 unique core_keys in 625,697 rows (3.6 % repeat
  a reaction, mostly under other conditions); 204 amide reactions appear under more than one key.
- rxnutils: `reactants2reagents` changed 65.3 % of rows; 206 rows had an unsanitizable molecule; 23 rows
  were left without a reactant (rejected with that reason). Top-20 bin sizes are identical raw vs rxnutils.
- 3,524 amide couplings have HCl as their only agent ("prepared analogously to Example X": the coupling
  reagent is in a referenced example ORDerly never read); 471 have only a base as "solvent".
- Runtime: the 15 s full count has no prefilter and no multiprocessing; RDKit parses only 370,996
  molecules for 471,478 reactions thanks to caching. rxnutils: 26 min on one core (`--max-workers` has
  no effect), 4.1 GB peak.

**Interpretation.**
- For HATU / DIPEA / DMF one bin is a positive set of ~1,375 literature reactions without any role
  assignment. Any other fixed condition gives ≤ ~550; using more of the literature needs merging.
- rxnutils does not change the exact-condition count. Its role changes come from atom mapping, and wrong
  mapping becomes a wrong role: 27 extra "amides" are acid counter-ions of salts (fumaric acid, TFA) kept
  as reactants; DMF stays a reactant when the mapper gives it the N of a record that lacks NH3; H2 in
  hydrogenations moves to reagents (hydrogens are never mapped).
- The amide rule misses acids written as carboxylate salts (1,159) and ammonium-written amines (92).

**Dependency change.** reaction-utils requires pandas < 3 and Python < 3.14, so the project moved to
pandas 2.3 (pandas-stubs 2.3) and `requires-python = ">=3.13,<3.14"`; all tests and mypy pass on 2.3.

**Reproduce.**
```
uv run python scripts/explore_condition_bins.py ~/University/Phd/Dataset/Datasets/orderly_condition_train.parquet
uv run python scripts/run_rxnutils.py ~/University/Phd/Dataset/Datasets/orderly_condition_train.parquet scratch/rxnutils/full
uv run python scripts/recount_condition_bins.py ~/University/Phd/Dataset/Datasets/orderly_condition_train.parquet scratch/rxnutils/full/rxnutils_out.tsv
```

**Next.** Decide the first plate condition (HATU / DIPEA / DMF is the only bin that clears the bar
without merging). Open: counter-ion acids as false amides; carboxylate/ammonium blind spots; whether the
~4,000 couplings with unrecorded conditions can be positives; final standardisation choice (rxnutils vs
chython) is still ADR 001 in `docs/design/rebuild.md`.
