# Modeling plan — amide coupling, HATU / DIPEA / DMF

Status: accepted 2026-10-09 (ADR 003, `docs/decisions/003-pu-model-amide-hatu.md`); remaining open
items are marked **[open]**.

## Question and setting

Which acid × amine pairs from our lab inventory work under HATU + DIPEA + DMF? I want a ranking of
pairs good enough to pick a plate: the model-picked wells should have a higher hit rate than random
wells.

- **Positives P:** amide couplings (acid + true amine) reported in ORDerly under a condition containing
  HATU, DIPEA and DMF, after the class filters (`docs/design/rebuild.md`, stage 7a). Size: from the full
  ORDerly run (expected ~1,000–1,500 pairs).
- **Unlabeled U:** acid × amine pairs of the lab inventory, 529 × 538 ≈ 285 k, minus pairs in P.
- **Validation (redefined 2026-10-22):** every labelled Chemlex row in the HATU family whose acid and
  whose amine appear nowhere in the test side, excluded on the flat (stereo-agnostic) InChIKey:
  2,615 reactions, 29.14 % positive. It used to be Chemlex's own `both_unseen == "train"` side alone
  (1,725 reactions, 32.8 % positive, 53 amines); that side shared 5 acids and 1 amine with the test
  side as stereoisomers. **Test:** both-unseen *test* side, 1,313 reactions (unchanged), read once at
  the end. Details under "Validation set" below.

## Model inputs (built 2026-10-09; written down 2026-10-20)

These three steps were built before this section existed (CLAUDE.md §2); it describes them as they
are. Each writes a table, a report and, where rows can be dropped, a rejected table through `platepu.io`.

| step | code | reads | writes | verified by |
|---|---|---|---|---|
| lab library | `library/inventory.py`, `scripts/library.py` | the inventory export (xlsx, header on row 3; internal) and the class YAML | `private/library/<class>.parquet`: one row per compound (by `key_stereo`) with its role, containers, name, CAS; `.rejected.parquet` per container: `no_smiles`, `unparsable`, `neutralise_failed`, `no_inchikey`, `no_role`, `both_roles`, `multiple_groups` | `tests/test_library.py`; full export 10,212 containers → 529 acids, 538 amines |
| unlabeled U | `library/unlabeled.py`, `scripts/unlabeled.py` | the library, the positives, the benchmark validation set | `private/unlabeled/<class>/<family>.parquet`: every role-A × role-B library pair not in P of the family (by stereo keys); report: overlap of lab and validation substrates with P (flat keys) | `tests/test_unlabeled.py`; 284,595 pairs, 7 lab pairs in P |
| validation / test | `benchmarks/split.py`, `scripts/benchmark_split.py` | `private/benchmarks/<name>/labels.parquet`, the family from the class YAML | `private/benchmarks/<name>/<family>/validation.parquet` and `test.parquet` (split `test`); labelled reactions in the family only (`in_family`); the report records `shared_acids` and `shared_amines` | `tests/test_benchmark_split.py`; Chemlex HATU_DIPEA_DMF 2,615 / 1,313, 0 shared acids, 0 shared amines (flat keys) |

### Validation set (redefined 2026-10-22)

The reasoning is in the module docstring of `src/platepu/benchmarks/split.py`. The fingerprints are
chirality-blind, so two stereoisomers carry the identical feature vector. A substrate that is in the
test side as one stereoisomer and in validation as another is therefore a leak. Validation is every
labelled row of the family whose acid and whose amine are absent from the test side, compared on the
flat key.

| quantity | value |
|---|---|
| labelled rows read in | 8,883 |
| validation reactions | 2,615 |
| validation positive rate | 29.14 % |
| test reactions | 1,313 (unchanged) |
| validation acids | 95 stereo keys, 94 flat |
| validation amines | 113 stereo keys, 112 flat |
| acids / amines shared with the test side (flat) | 0 / 0 |
| rows of the old validation set now excluded | 150 |

The flat counts (94 and 112) are the ones that matter statistically: one acid pair and one amine pair
collapse to a single feature vector. The old validation set (53 amines) is not a subset of the new one:
150 of its rows carried a stereoisomer of a test substrate (5 acids, 1 amine) and are gone.

    uv run python scripts/benchmark_split.py chemlex amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived_samples

Inventory-derived tables (library, its rejected containers, U) carry `licence = "internal"`, so
`write_table` refuses them outside `private/` (ADR 004, 2026-10-20). Chemlex-derived tables carry
`licence = "noncommercial"` for the same reason and under the same guard, the labels and the
validation and test tables included, which is why they sit under `private/benchmarks/` (ADR 009,
2026-10-22). No script writes a licence value any more: every stage reads it from its input and
merges with `io.strictest_licence`, so a substrate seen in both the inventory and validation comes
out `internal` and one seen only in validation comes out `noncommercial`.

## What the literature says (and what it does not)

**PAYN** (Boser, Spies, Glorius, *JACS* 2026, 148, 15066): one-shot *spy* PU. A CatBoost classifier
separates P from U with 20 % of P hidden in U as spies. The threshold keeps 95 % of the spies above
it, and U below it becomes "reliable negatives" (yield 0). A CatBoost regressor is then trained on P +
reliable negatives. Features: ECFP r=2, 2048 bits per component, concatenated. No class prior. P and U
are *simulated* by hiding negatives at random inside four fully labelled HTE sets (Buchwald–Hartwig,
Ni borylation, Suzuki), so SCAR holds by construction. P:U is 1:1. Evaluation is random 5-fold CV
inside the same set (yield MAE); there is no ranking metric, no amide coupling, no real literature
bias and no transfer to another source. Our setting differs on all of these points, so their settings
are a starting point, not an answer.

**Chemlex** (Zhong et al., *Nat. Commun.* 2025, 16:4522; CC BY-NC-ND 4.0):
- HATU 2.0 eq, base 3.0 eq, 0.08 M in DMF, 30 min at rt/30 °C. That is much shorter than typical
  literature conditions.
- Conversion = product area % / (1 − reagent area % − acid *or* amine area %) at 254 nm, with no
  response factors.
- Substrates were picked to match Pistachio's acid and amine type proportions. About 5,600 reactions
  were *designed as likely negatives* (weak nucleophiles, hindered pairs), which explains the
  sulfonamides, amidines and similar "amines" we found. The released file does not flag them.
- Both-unseen split: grid sorted, a corner block is test, off-diagonal blocks discarded (our 4,475
  unlabelled rows).
- Their best supervised model (DRFP + Bayesian NN) reaches AUC 0.79–0.81 on both-unseen over all
  conditions, with no per-condition result. That is a rough ceiling for a model that never saw Chemlex
  labels.

**PU theory** (Bekker & Davis, *Mach. Learn.* 2020):
- Our data are the *case-control* scenario: P from one source, U from another.
- A ranking needs no class prior if SCAR holds.
- Under SAR (propensity e(x) depends on attributes) the classifier can be reweighted, but the prior is
  not identifiable from PU data alone: it needs outside knowledge (here, Chemlex) or tuning.
- Propensity attributes should be *usage* statistics (literature frequency of a substrate,
  availability, year), not structure, or "gets reported" and "works" cannot be separated.

**Reporting bias** (Beker et al. *JACS* 2022; Strieth-Kalthoff et al. *Angew.* 2022): literature-trained
models often do no better than substrate popularity. Popularity is therefore a required baseline.

**Domain shift** (web research; to verify before citing: Hammoudeh & Lowd 2020; Kumagai et al. 2024,
2025): P substrates are literature molecules, U substrates are inventory molecules. A P-vs-U
classifier can learn "literature-like vs inventory-like" instead of reactivity. This is the main risk.

## Plan

**M0 — baselines** (no PU):
1. random;
2. substrate popularity: acid count × amine count in literature amide couplings (all ORDerly couplings,
   not only HATU);
3. nearest neighbour: Tanimoto of the pair to the closest P pair;
4. additive: acid score + amine score, each from how often similar substrates appear in P;
5. chemist rule: amine class (aliphatic primary > secondary > aniline > heteroaryl amine) and
   steric hindrance at the reacting atoms.

**M1 — PU models** (decided 2026-10-09: PAYN-style, including its yield-regression step):
1. P vs U classifier (CatBoost; the SCAR baseline);
2. PAYN: spy classifier (spy rate 0.2, tolerance 0.05) → reliable negatives at yield 0 → CatBoost
   regressor on P (literature yields) + reliable negatives;
3. bagging PU, as a variance-reduced version of (1).

The regressor predicts a yield on the literature scale; Chemlex measures LC-MS conversion. The two are
never put on one scale: the predicted yield is used only as a ranking score and evaluated against
Chemlex labels (≥ 20 %). Spearman with conversion is reported as secondary, with this caveat.

**M2 — shift checks and fixes**, run on M1:
1. *Adversarial validation*: how well a classifier separates literature, inventory and Chemlex
   substrates on the same features.
2. Correlation of the model score with a "literature-likeness" score.
3. *Pair shuffle*: retrain with the acid–amine pairing in P shuffled. If validation does not drop, the
   model learns single substrates, not pairs.
4. U composition (decided 2026-10-09): the main U is the lab inventory; on validation I compare it
   with U = inventory + literature acids × literature amines. The better one is used, and the
   difference is reported.

**Features:** Morgan r=2, 2048 bits per substrate, concatenated (as PAYN). Reacting-centre descriptors
(amine class, heavy-atom count and branching at the α-atoms) come in M2 if M1 learns single
substrates. Pretrained embeddings only later, because they may make literature vs inventory easier to
separate.

**Evaluation** (Chemlex validation; test once, with a configuration frozen and committed first):
- metrics: ROC-AUC, PR-AUC, enrichment and hit rate at k = 24 / 48 / 96 versus the base rate;
- uncertainty: bootstrap that resamples whole acids and amines;
- reported in groups: both substrates in P / one / none;
- caveat: Chemlex's designed negatives cannot be separated (not flagged), so absolute numbers are on a
  harder-than-typical set.

## Stages of the prototype (accepted 2026-10-20, ADR 006)

Each stage reads tables and writes one table, a report and, where rows can drop, a rejected table.
Nothing reads the Chemlex test side.

| stage | reads | writes | verified by |
|---|---|---|---|
| F features | positives (both widths), library, validation | one row per substrate: Morgan counts and reacting-centre descriptors | hand-checked descriptors on 10 substrates; same key gives same row |
| S scores | features, P choice, U choice, config in `configs/models/` | a score per validation pair and per inventory pair, with seed and config hash | baselines B0 to B7 reproduce the counted AUCs |
| E evaluation | scores, validation labels | ROC-AUC, PR-AUC, hits at k = 24 / 48 / 96, paired bootstrap differences, by amine class and by substrates in P | the success rule of ADR 006 evaluated once per run |

Splits of P keep one time block (`source_document`) together (ADR 006, item 4); ORDerly has no patent id.

### Stage F — features (agreed 2026-10-21)

Carries out ADR 006, item 2. Counted first (`scratch/count_f_features.py`, `scratch/count_tautomers.py`).

**Code.** `src/platepu/features/substrate.py`, `scripts/features.py`. `features` imports `classify`
only; nothing earlier imports it, stage S will.

**Inputs.** Positives (every family; HATU is a subset of `all`), the library, and the validation set
(never `test.parquet`, and the script refuses any file not named `validation.parquet`; test
substrates get the same code at the final evaluation). Only the open positives
(`<root>/positives/`) are read for now. When Pistachio is used, F must also read
`private/positives/`. Every input carries
`std_version`. Positives, library and benchmark measurements were changed on 2026-10-21 so it reaches
this stage. Mixed values stop the run.

**Output.** One table, `<root>/private/features/<class>/substrates.parquet`, plus `.report.json`. It is
under `private/` because it holds library rows (ADR 004). There is one row per `key_stereo`
(counted 2026-10-21: 11,787 acids, 10,204 amines; after the validation set grew, 2026-10-22: 22,050 rows,
11,796 acids and 10,254 amines, no duplicate `(key_stereo, role)`, no null descriptor, and 100 % key
coverage on validation, inventory and P at both widths).

| column | type | meaning |
|---|---|---|
| `key_stereo`, `key_flat`, `role`, `smiles` | string | identity; `smiles` is the lexicographically smallest of the key's SMILES |
| `n_smiles` | int16 | distinct SMILES seen for the key (2 for the tautomer pairs, rule 3) |
| `std_version` | string | one value |
| `licence` | string | `internal` if in the library, else `licensed` if any positive is, else `open` (Chemlex is `open` from stage 1) |
| `in_positives`, `in_library`, `in_validation` | bool | where the key was seen (for M2's adversarial validation) |
| `morgan` | `fixed_size_list<uint8>[2048]` | Morgan counts, radius 2, no chirality (counted max 51) |
| `morgan_centre` | `fixed_size_list<uint8>[2048]` | the same fingerprint rooted at the reacting atom (added 2026-10-22) |
| 10 descriptors | int16 | below |
| `subclass` | string | role subclass; null for a role without subclasses |

**Descriptors.** These are generic: one function serves both roles. R is the molecule atom at the `:1`
position of the role SMARTS. G is every atom of that match (acid: C, =O, OH; amine: N). The alpha atoms
A are the heavy neighbours of R that are not in G.
- `n_heavy_atoms`: heavy atoms in the substrate.
- `centre_in_ring`: 1 if R is in a ring, else 0.
- `n_alpha`: the number of atoms in A.
- `alpha_n_aromatic`, `alpha_n_in_ring`: how many atoms in A are aromatic, and how many are in a ring.
- `alpha_max_heavy_degree`, `alpha_sum_heavy_degree`: the largest and the summed heavy degree of the
  atoms in A. The bond to R counts, so methyl is 1 and the tert-butyl C is 4. Both are 0 when A is empty.
- `n_heavy_at_2_bonds`, `n_heavy_at_3_bonds`: heavy atoms not in G at topological distance 2 and 3 from R.
- `n_ewg_on_ring` (added 2026-10-21): electron-withdrawing substituents on the aromatic ring system that
  carries R, 0 when no neighbour of R is aromatic. The ring system is every aromatic ring fused to any
  aromatic neighbour of R, so both rings of a diarylamine count; a biaryl partner is a separate system. A substituent is counted once, by the atom that
  attaches it, so a ring-bound amide counts 1 although `C(=O)X` matches it twice. The SMARTS are in
  `configs/features.yml`, not in the class YAML, because what withdraws electrons does not depend on the
  reaction class.

A descriptor that is constant for one role stays in the table (acid: `n_alpha` = 1, `centre_in_ring` = 0);
S may drop it. Left out after the counts: H on R, the heavy degree of R (redundant with `n_alpha`), and
hetero atoms in A (nearly constant).

`n_ewg_on_ring` is the one descriptor the validation set cannot check. No validation aniline carries a
nitro, nitrile, trifluoromethyl or sulfonyl group at all; the whole observed effect is three anilines with
a ring carbonyl, which succeed in 2.6 % of reactions against 21.7 % for the other thirteen. It is in the
table because the chemistry is settled (Hammett rho -2.1 to -2.5 for aniline acylation) and about 10 % of
the lab amines carry such a group, not because validation shows it.

**The fingerprint rooted at the reacting atom** (`morgan_centre`, added 2026-10-22). The same Morgan
settings, but computed with RDKit's `fromAtoms=[R]`, so only the environments centred on the reacting
atom are counted: at most radius + 1 bits. Measured over the full table on 2026-10-22: **480** distinct
environments for the 10,254 amines and **308** for the 11,796 acids, against 2,048 bits ever set for the
whole-molecule fingerprint on either role. And **every amine environment falls inside a single
hand-written subclass** (acids 99.4 %, counted on 2026-10-21). It is therefore a generic and finer
replacement for a taxonomy, and it needs no per-class chemistry: only the `:1` mark the role SMARTS
already carries.

The 2026-10-21 table gave 459 and 314 for these two counts. The acid count fell although the table
gained acids, which cannot happen if the fingerprint is unchanged, so the root atom moved for some
acids between the two tables. I could not reconstruct which change did it, because none of this work is
committed yet. What I could check, I checked: recomputing `morgan` and `morgan_centre` from the stored
SMILES for 40 acids and 40 amines reproduces the stored vectors exactly, so the table on disk does match
the code as it stands. The earlier pair of numbers is not trustworthy and is kept here only so the
discrepancy is not silently lost.

It is not a subset of the whole-molecule fingerprint. The generator drops an environment whose bond set
is contained in another it has already recorded, and the sphere around a terminal atom sits inside the
sphere around its neighbour. An amine N is usually terminal, so its radius-2 environment is missing from
the full fingerprint and appears only here. The rooted column therefore adds information rather than
only pointing at it.

**Subclasses.** The class YAML key `roles.<role>.subclasses` is an ordered list of `{name, smarts}`. Each
SMARTS marks the reacting atom `:1`, and a subclass counts only when that atom is R; the first match wins
and no match gives `other`.

The amine has four: `heteroaryl_amine`, `aniline`, `aliphatic_secondary`, `aliphatic_primary`. These are
the SMARTS of `scratch/count_m0_anchors.py`, anchored at R. Anchoring changes the class of 821 of 9,973
positive amines and of 2 validation amines; the amine-class AUC on validation moves from 0.707 to 0.713.
They were left at four on 2026-10-21: the counts do not support splitting the primary amines (the
direction reverses and the last class holds one validation amine), and the cyclic-versus-acyclic split of
the secondary amines, which Mayr's nucleophilicities do support, is already `centre_in_ring`.

**The acid has none** (removed 2026-10-22). Nine were written on 2026-10-21 and taken out again the
next day under the principle of ADR 008: hand-written chemistry may live in the baseline and in the
reporting, never in the model's features. No baseline used them, so they were a feature only, and a
fingerprint rooted at the reacting atom recovers the same families generically. The nine SMARTS and
what they showed (success from 0.136 for an alpha-quaternary acid to 0.470 for a heteroaromatic one
with a pi-donor) stay reproducible in `scratch/count_subclass_evidence.py`; the acid side is still
the chemistry that matters, since Haas et al. (PNAS 2022) find the acid largely sets the rate and the
sharpest failures in Due-Hansen et al. (Org. Biomol. Chem. 2016) are acid-side.

`load_class` reads subclasses into `ClassRule`, so stage F and stage E share one definition, and the
key is optional: a new reaction class runs the whole pipeline without writing any taxonomy. Morgan
settings and the electron-withdrawing SMARTS live in `configs/features.yml`.

The acid `subclass` column is null for all 11,796 acids by design: `subclass_of`
(`src/platepu/classify/groups.py`) returns `None` for a role with no subclasses. Stage S2 must not
one-hot encode `a_subclass`; a null column encoded silently would be a bug.

**Rules (fail loudly; no rejected table).**
1. A role SMARTS with 0 or ≥ 2 reacting atoms gives `ValueError` (counted 0 in every input).
2. `ValueError` also for a key under both roles, mixed `std_version`, a SMILES that does not parse, or a
   Morgan count above 255.
3. A key with several SMILES takes its features
   from the smallest SMILES, and the report lists those keys. These are tautomers merged by the
   InChIKey: 33 in the full run, 29 inside the positives and 4 drawn differently in the library and
   the positives. Canonical tautomers in stage 6 were counted
   and rejected (Luka, 2026-10-21). RDKit's `Canonicalize` removed stereo in about 3,200 substrates and
   wrote wrong structures (indoline-2-carboxylate became an indole hemi-orthoester). It also took the
   amine role from 23 positive amines, all for 29 keys.

**Verification.** Tests in `tests/test_features.py` use hand-checked open SMILES. A full run must
reproduce the counts above, and two runs must give equal tables. A grid image of 20 open,
positives-only substrates with R and A highlighted is saved next to the table as
`substrates.grid.png`. It is drawn by `scratch/grid_features.py`, a one-off check, not part of the
stage.

**Report.**
- Rows in per input.
- Per input and role: keys, subclass counts, and min / median / max of each descriptor (the
  domain-shift record).
- Per role: rows out, the overlaps between inputs, and the keys with several SMILES.
- Morgan: max count, mean non-zero bits, and bits never set.

## Open questions

1. **Decided 2026-10-21:** no yield cut-off for P. A cut-off drops a pair reported at 15 % while
   keeping the 37.6 % that report no yield at all, and it changes P by at most 6.6 % (1,177 pairs
   with no cut-off, 1,145 at 10 %, 1,099 at 20 %). A 20 % variant is reported once as a robustness
   line.
2. **[open, goes to ADR 008 with the S2 inputs]** P:U ratio for the classifiers (PAYN used 1:1; ours
   is ~1:242): subsample U to 1:1 and compare with the full U on validation. Also open there: the U
   composition, and whether catboost is added or scikit-learn's `HistGradientBoostingClassifier` is
   enough.

Decided 2026-10-09: target as PAYN (classifier + yield regression, used for ranking); U compared as
above; Chemlex-derived tables are never shared, only the code that rebuilds them (CC BY-NC-ND, CLAUDE.md
§9); the plate's outcome is the same as Chemlex's (LC-MS conversion ≥ 20 % under a short HATU
protocol), so Chemlex is the right benchmark.

### Stage S — scores (agreed 2026-10-21)

Carries out ADR 006 item 5 and ADR 007 item 5. One score per pair and nothing else: no metric is
computed here (that is stage E) and no label is ever read.

**Two steps.** S1 are the baselines B0 to B7, which fit nothing. S2 are the PU models, which fit a
learner and need a U choice, a P:U ratio and a learner. They are two scripts over one shared
pair-assembly function, because one stage would need a branch between a pure function of the tables
and a fit-then-predict, and ADR 006 item 5 runs them weeks apart. They write the same columns, so
stage E reads one layout. **S1 is built first** (Luka, 2026-10-21); the S2 inputs go to ADR 008.

**Code.** `src/platepu/score/pairs.py` (pair rows and the feature matrix) and `score/baselines.py`;
`scripts/score_baselines.py`. `score` imports nothing from `platepu` at all; the script reads
the positives, library and benchmark tables and hands DataFrames in. Nothing imports `score`.

**Inputs.** The feature table `private/features/<class>/substrates.parquet`; the positives at both
widths (`family = "<family>"` and `"all"`, ADR 006 item 1); U, `private/unlabeled/<class>/<family>.parquet`;
and the validation file, read with an explicit column list so that `label` and `conversion_mean`
never enter the process. The script refuses any file not named `validation.parquet`. A pair whose
substrate key is missing from the feature table, or a mixed `std_version`, raises `ValueError`;
coverage was counted at 100 % for all four inputs, so a miss is input we have not seen.
P has **no yield cut-off** (Luka, 2026-10-21): the cut-off would drop a pair reported at 15 % while
keeping the 37.6 % of pairs that report no yield at all, and it changes P by 6.6 % at most.

**Output.** `private/scores/<class>/<family>/<model>.parquet` and `.report.json`, under `private/`
because the inventory pairs are internal (ADR 004); every row carries `licence`, so `write_table`
enforces the path from the column. One row per scored pair: `score_set` (`validation` or
`inventory`), `key_a_stereo`, `key_b_stereo`, `score` (float32, higher is better, comparable only
inside one `score_set`), `model`, `p_width`, `seed`, `config_sha256`, `std_version`, `licence`.
There is **no rank column** (Luka, 2026-10-21). A rank is derived, it is not comparable between
models, and its meaning depends on a tie convention; stage E and the plate step each compute the
ranking they need. The report records the tie structure instead, because the coarse baselines are
heavily tied: on the inventory 90.1 % of the aniline pairs share one B5 score (110,560 of 122,727,
`scratch/count_stage_s_inputs.py` section 3, re-counted 2026-10-22), so any ranking of them
is a decision about ties and not about chemistry.

**From substrate rows to a pair row.** `pair_matrix` joins the feature table twice, on
`key_a_stereo` and on `key_b_stereo`, prefixing the columns `a_` and `b_`. Block order, returned with
the matrix: the two Morgan blocks, the nine model descriptors of each side, then the one-hot subclass
columns, whose categories come from the class YAML order plus `other` (five for the amine; the acid has
no subclasses since 2026-10-22) and never from the data, so a column means the same in every run. `n_heavy_atoms` is **not** a
model feature: validation pairs average 38.66 heavy atoms against 23.73 for inventory pairs, so a model
given size can rank the inventory by size alone. It is kept for stratification and for B7.

Dropping that column is the cheapest step and not a cure. The sum of a Morgan count vector is about
(radius + 1) x heavy atoms, so **size is still in the features** through the fingerprint and cannot be
removed without changing what the fingerprint is. The real defences are B7, which says how far size
alone gets on validation, and the adversarial-validation check of M2, which stage E must report
beside every model: the correlation of the model score with pair size, and the AUC of a classifier
separating literature from inventory substrates on the same features. A model whose score correlates
with size as strongly as B7 does has not learned chemistry.

**The baselines.** None fits a parameter; B2 and B3 only look up P. B4 to B7 are defined on the amine
or on size alone (the acid subclasses were removed on 2026-10-22 from `configs/classes/amide_coupling.yml`,
see `docs/decisions/008-hand-written-chemistry-belongs-in-the-opponent.md`), so they give many pairs the same score; the report records the tie structure of each.
- **B0 random:** `numpy.random.default_rng` over the pairs of one `score_set`, sorted by both keys,
  with a seed derived from the config seed **and the set name**. One stream over both sets would make
  the validation scores change whenever U changes, which is not reproducible.
- **B1 popularity:** summed `n_reports` per `key_flat` from positives `family = "all"`, score
  `log1p(n_a) + log1p(n_b)`. It is weak here and must be read with that in mind: on the 2,615
  validation pairs, the amine is in no literature positive in 2,379 (91.0 %) and the acid in 1,789
  (68.4 %); these are shares of pairs, not of distinct amines. 1,615 pairs (61.8 %) tie at zero because
  both sides are unreported, and the score takes 43 distinct values
  (`scratch/count_stage_s_inputs.py`, section 2). On the old validation set the shares were 90.6 % and 68.6 %.
- **B2 nearest neighbour:** Tanimoto on the binarised Morgan counts, per side; the pair similarity is
  the **minimum** of the two sides; the score is the maximum over the P pairs of the chosen width.
  The minimum, not the mean (Luka, 2026-10-21): a pair that shares its acid with a known pair but has
  an unrelated amine is not an analogue, and the mean would give it a middling score. The minimum also
  keeps B2 and B3 asking different questions. Because
  the similarity is per side, the two substrate-to-substrate matrices are small (529 lab acids against
  790 P acids, and 538 against 680) and are computed once; what remains is a reduction over
  284,595 x 1,177 pair combinations at HATU width, and x 28,158 at class width, chunked at 5,000
  scored pairs. A concatenated 4,096-bit version was measured instead at about 2 s and about 0.6 min,
  with 96 GB in one piece at class width, so chunking is needed either way.
- **B3 additive:** per side, the Tanimoto to the single most similar P substrate of that role;
  score = acid score + amine score. One neighbour, not a mean over k (Luka, 2026-10-21): k would be an
  arbitrary constant, and with k = 1 the contrast with B2 is exactly the question worth asking, namely
  whether the pair matters or the two substrates on their own are enough. Needs only the two
  substrate-to-substrate matrices, so it is trivial.
- **B4 amine subclass:** the per-subclass score in the model YAML, `other` listed explicitly.
- **B5:** rank sum of B4 and `-b_n_heavy_at_2_bonds`. **B6:** B5 plus `-b_n_ewg_on_ring`. Measured
  2026-10-22 on the new validation set: **0.810** and **0.822** (0.791 and 0.802 on the old one).
  ADR 007 item 5 records 0.775 and 0.793 for B5 and B6; those figures come from the exploratory script,
  which built the steric part from two descriptors (`-(alpha_sum_heavy_degree + n_heavy_at_2_bonds)`)
  rather than one, and were measured on the old set. The two-descriptor variant of B5 gives 0.794 on the
  new set, so the discrepancy is the descriptor choice, not an error in either number. The two steric
  scores agreed at Spearman 0.982 on the old set but not exactly. The one-descriptor form is what is implemented,
  because it is simpler and scores higher, so it raises the bar rather than lowering it. ADR 007 is
  accepted and not rewritten; the correction is recorded here, in the journal, and in ADR 008.
- **B7 size only (added 2026-10-21):** `-(a_n_heavy_atoms + b_n_heavy_atoms)`. This is a control, not
  a chemistry baseline: it shows how much of a model's validation score is explained by size alone.
  Its sign comes from the prior that a less hindered pair reacts faster, not from validation, where
  it scores 0.598. The report also carries the acid-only and amine-only variants as diagnostics
  (acid-only 0.486, amine-only 0.645; 762 positives and 1,853 negatives; old set 0.575 and 0.628;
  `scratch/count_stage_s_inputs.py`, section 4). The amine alone carries almost all of it and the acid
  alone is at chance. The amine-only variant scores higher than the pair, but the pair is
  the pre-registered form, because picking the higher one would be a choice made on validation.

**Determinism.** One YAML per configuration in `configs/models/` with `model`, `p_width`, `seed` and
the baseline settings; `write_report` records the hash of it, of the class YAML and of
`configs/features.yml`, plus the git commit and package versions. Before the test side is read, the
model YAML and the validation report are committed.

**Stage S must not** compute a metric, read a label or any `test.parquet`, choose between
configurations, or search hyperparameters.

**Verification.** Tests first in `tests/test_score_pairs.py`, `tests/test_score_baselines.py` and
`tests/test_score_script.py`, on open hand-written rows. A full run (seed 20261021, `std_version` 2)
scores 2,615 validation pairs and 284,595 inventory pairs under ten (model, width) groups, 2,872,100
rows out, in 1 min 29 s (89.2 s wall, `/usr/bin/time -p`). A second run gave the identical table
(`pd.testing.assert_frame_equal` passed on the before and after parquet). P has 1,177 HATU pairs and 28,158
class pairs.

    uv run python scripts/score_baselines.py amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

Measured on validation, 2026-10-22 (the same command; this replaces the 2026-10-21 table, which was
on the 1,725-reaction set):

| baseline | ROC-AUC | distinct scores | largest tie |
|---|---|---|---|
| B0 random | 0.504 | 2,615 | 1 |
| B1 popularity | 0.524 | 43 | 1,615 |
| B2 nearest pair, HATU width | 0.590 | 350 | 154 |
| B2 nearest pair, class width | 0.632 | 392 | 101 |
| B3 additive, HATU width | 0.621 | 2,244 | 6 |
| B3 additive, class width | 0.621 | 1,385 | 71 |
| B4 amine subclass | 0.743 | 4 | 916 |
| B5 + sterics | 0.810 | 17 | 363 |
| B6 + electronics | 0.822 | 24 | 298 |
| B7 size only | 0.598 | 62 | 102 |

B0 at 0.504 is the floor, as it should be.

The two B3 columns give the same AUC to three decimals at both P widths. The two score vectors are
genuinely different (Spearman 0.414 between them; means 0.818 at HATU width and 1.213 at class width; `uv run python scripts/counts/count_b2_vs_b3.py`, section `width_coincidence`). I checked that they are not identical, so this is a
coincidence and not a bug.

The earlier table led me to write that B2 beats B3 by about 0.012 at either width. **I retract that
claim.** It was never supported. I now quote uncertainty from paired differences and not from each AUC
on its own: the scores are correlated, so the paired interval is far narrower than two single
intervals would be. Method: Owen's pigeonhole bootstrap, 2,000 resamples, seed 20261022, resampling
whole acids and whole amines by flat key (`uv run python scripts/counts/count_b2_vs_b3.py`).

| difference | estimate | 95 % interval | share of resamples ahead |
|---|---|---|---|
| B2 class width minus B3 class width | +0.010 | [-0.070, +0.084] | 57.6 % |
| B2 HATU width minus B3 HATU width | -0.031 | [-0.109, +0.044] | 18.6 % |
| B6 minus B4 | +0.080 | [+0.037, +0.134] | 100 % |
| B6 minus B1 | +0.298 | [+0.204, +0.383] | 100.0 % |

- B2 against B3 at class width is indistinguishable from zero. I cannot say whether knowing which acid
  went with which amine helps at all.
- Validation amine subclasses (113 stereo keys): aliphatic_secondary 34, aniline 30, heteroaryl_amine 27,
  aliphatic_primary 17, other 5.
- Pair overlap with P: no validation pair is in P at either width, on stereo or flat keys. Inventory
  pairs are: 406 on stereo keys and 669 on flat keys at the class width.
- Narrowing P from the class to the HATU family hurts B2: B2 at the HATU width is ahead of B3 at the
  HATU width in only 18.6 % of resamples. The interval still covers zero.
- A paired comparison here resolves about **0.05 AUC**. That is the size of edge a PU model has to
  show over a pre-registered baseline before I call it real.
- The amines carry most of the uncertainty. Resampling amines alone gives a half-width of 0.083 for
  B6 minus B2; resampling acids alone gives 0.029.

## Sources

Notes with page references in `docs/literature/notes/`: `boser-2026-payn.md`, `zhong-2025-chemlex.md`,
`bekker-2020-pu-survey.md`, `2026-10-09-pu-modeling-web-research.md` (agent summaries; see their headers). To read before ADR 003: Bekker,
Robberechts & Davis, ECML PKDD 2019 (SAR), and the domain-shift PU papers above.
