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
| unlabeled U | `library/unlabeled.py`, `scripts/unlabeled.py` | the library, the positives, the benchmark validation set | `private/unlabeled/<class>/<family>.parquet`: every role-A × role-B library pair not in P of the family (by stereo keys); report: overlap of lab and validation substrates with P (flat keys) | `tests/test_unlabeled.py`; 284,600 pairs, 2 lab pairs in P against the rebuilt P of 2026-10-29 (284,595 and 7 before it) |
| U literature | `positives/unlabeled.py`, `scripts/unlabeled_literature.py` | the positives at `family = "all"`, the U config | `unlabeled/<class>/<config>.parquet`: one row per drawn counter-pair, one side replaced from the same heavy-atom decile, P subtracted on flat pair keys | `tests/test_unlabeled_literature.py`; the joint size probe at ~0.5 against a uniform-U control |
| S2a reliable negatives | `score/pu.py`, `scripts/score_pu.py` | features, positives, U, the S2 config | `pu/<class>/<config>.negatives.parquet`: one row per U pair and per spy, with its spy score and whether it is a reliable negative | `tests/test_score_pu.py`; realised spy recall equals the configured 0.95 |
| S2b PU scores | `score/pu_scores.py`, `scripts/score_pu_scores.py` (written and run on all six configurations, 2026-11-02) | features, positives, the S2a negatives, the validation pairs, the inventory grid | the stage S score layout, one row per (model, feature block, scored pair) | two runs identical; B2 and B8 reproduce from the shared pair assembly |
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
| S scores | features, P choice, U choice, config in `configs/models/` | a score per validation pair and per inventory pair, with seed and config hash | baselines B0 to B8 reproduce the counted AUCs |
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
| `morgan_near` | `fixed_size_list<uint8>[2048]` | the same fingerprint rooted at every atom within `distance` bonds of the reacting atom (added 2026-11-02) |
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

**A third block, `morgan_near`** (agreed 2026-11-02). It is the Morgan count fingerprint with the same
settings as the other two (radius 2, 2,048 bits, no chirality). Its roots are every atom within
`distance` bonds of the reacting atom, and each root contributes its radius-2 environments.
`morgan_centre` is the same construction at distance 0. At a distance beyond the molecule's diameter it
equals `morgan` bit for bit. The distance lives in `configs/features.yml`:

    morgan_near:
      radius: 2
      n_bits: 2048
      distance: 6

The column name carries no number on purpose. ADR 030 selects the distance per reaction class: amide
coupling gets 6, another class may get 3. A column called `morgan_d6` would lie for the next class, and
stage S validates `feature_block` against a fixed set of names. The distance is recorded in the config
and hashed into every run instead.

Why 6 for this class is in ADR 030. In short: at distance 0 only 2.0 % of the 284,600 laboratory
candidate pairs have a feature vector no other candidate shares, and the first stage S2b run gave its top
96 pairs one single score shared by 1,300 pairs. The rule takes the smallest distance within 5 points of
the best achievable share, which is 6 at 89.1 % against a best of 91.0 %.

Containment is not what a reader will assume. The series is nested from distance 1 upward, but distance 0
is special and can carry bits that no larger distance has. RDKit drops an environment whose bond set lies
inside another it has recorded, so the radius-2 environment of a terminal reacting atom survives at
distance 0 and is swallowed once its neighbour becomes a root. It holds on 7 of 14 fixtures (pivalic
acid, tert-butylamine, aniline, benzylamine, 2-aminopyridine, 4-chloroaniline, ethylamine); on
ethylamine it is bit 317. Command: `uv run pytest tests/test_features.py -k terminal_centre`
(`test_near_loses_the_radius_two_bit_of_a_terminal_centre_at_the_first_step`). The far boundary and the
containment claim: `uv run pytest tests/test_features.py -k "near_beyond_the_diameter or only_gains_bits"`. It is the same quirk the `morgan_centre` comment in `configs/features.yml`
already records. I keep default RDKit and not `includeRedundantEnvironments=True`, because the redundant
setting would break the bit-for-bit match between a large distance and `morgan`. That match is what makes
the whole distance table comparable with the numbers already measured.

Verification (tests written first; all 61 that were red are green on 2026-11-02, whole suite 765 passed,
1 skipped, `ruff check .` and `mypy src` clean):
- `morgan_near` at distance 0 equals `morgan_centre` exactly on every fixture.
- Beyond the diameter it equals `morgan`.
- Containment holds from distance 1 upward.
- The terminal-atom exception is pinned by name.
- The uint8 count guard is unchanged.
- A missing or non-integer `distance` raises `ValueError` naming the file.
- The stage F table carries the column, and `block_matrix` returns it as an (n, 2048) float64 matrix.

The rebuild, run 2026-11-02 at 14:52:

    uv run python scripts/features.py amide_coupling ~/University/Phd/Dataset/Datasets/derived \
        --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

The file went from 1,853,789 to 2,693,543 bytes. The rebuild left the two existing blocks untouched. I
checked this and did not assume it, because every figure measured so far comes from those two columns, and
a rebuild that had quietly altered them would have invalidated all of it without any test failing. The
check was done once, at rebuild time on 2026-11-02, by comparing the rebuilt table against a copy of the
previous one taken immediately before the run. The previous table no longer exists, so this check is not
repeatable and is recorded on trust. If it is ever needed again, the way is to rebuild from the previous
commit's code and compare, which is a reconstruction and not the same thing.

The row count is 11,345 for the current table under `derived`. The 22,050 in the Output paragraph above
(2026-10-22) is the same root at an earlier date: it predates the 2026-10-29 rebuild of P with the 20 %
yield cut-off, which took the substrate table down with it. There is no feature table under
`derived_samples`.

| | before | after |
|---|---|---|
| rows | 11,345 | 11,345, same keys, same order |
| `morgan` | | unchanged, bit for bit |
| `morgan_centre` | | unchanged, bit for bit |
| `morgan_near` | absent | present, equal to neither |

Mean set bits per substrate: `morgan_centre` 3.0, `morgan_near` 26.8, `morgan` 34.2. These can be
reproduced against the table as it stands:

    uv run python -c "import numpy as np, pandas as pd; d = pd.read_parquet('~/University/Phd/Dataset/Datasets/derived/private/features/amide_coupling/substrates.parquet'); print({b: round(float((np.stack([np.asarray(v) for v in d[b]]) > 0).sum(1).mean()), 1) for b in ('morgan_centre', 'morgan_near', 'morgan')})"

The 26.8 is also the value the d = 6 row reports in the seven-block table that ADR 030's rule was applied
to:

    uv run python scripts/counts/count_centre_distance.py ~/University/Phd/Dataset/Datasets

The production code reproduces, by a different code path, the scratch measurement the
decision was taken on. So the block in the stage is the block that was selected.

Consequence: the feature table is rebuilt, and stages S2a and S2b are re-run under the new block. Every
figure measured under `morgan_centre` for a fitted model is superseded for the primary line. That
includes the 15,250 reliable negatives of the first S2a run and the four S2b score tables of 2026-11-02.
They are kept and reported, not deleted, because the comparison across blocks is part of the result.

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

### The substrate overlap is reported and never removed (written down 2026-10-29)

This was decided for stage U from the start and never said for P or F, so data-guardian looked for
the code that removes the overlap and could not find it. It does not exist, and it should not.

**Why nothing is dropped.** Filtering P, U or the feature table by membership in the validation split
would mean reading the validation set to decide what may be trained on. That lets the evaluation set
choose the training set — leakage turned around, and harder to spot because it looks like caution.
The same sentence already governs stage U below ("Stage U must not … subtract the validation pairs
from U — that last would use Chemlex labels to build training data"). The overlap is counted, named
in the report, and carried into the paper as a number.

**What *is* enforced** is validation against **test**: validation is every labelled family row whose
flat acid key and flat amine key are both absent from the test side, and `scripts/benchmark_split.py`
aborts if the two sides share one substrate (0 shared acids, 0 shared amines). Substrates shared
between P and validation are a different thing and are allowed: the task is a *pair*, no validation
pair is in P, and a real plate reuses known building blocks in new combinations.

**Three measurements, none comparable with another.** This is the trap, and on 2026-10-29 I fell into
it and reported a count of pairs as a count of molecules:

| where | key space | unit | measured against |
|---|---|---|---|
| stage F, `overlaps` | stereo | substrate keys | the family validation split |
| stage P, `leak_pairs_one_side` / `leak_pairs_both_sides` | flat | **pairs** | every labelled Chemlex row, test side included |
| `scripts/unlabeled.py`, `share` / `pair_groups` | flat | pairs | the family validation split |

So stage F's 17 acids and 6 amines and stage P's 215 are answers to three different questions, and
putting two of them in one sentence is a mistake. The stage P keys were `leak_substrate` and
`leak_pair` until 2026-10-29; both always counted pairs (ADR 010, "the counts now mean distinct
pairs"), so the rename implements that record rather than changing it. The three implementations are
left as they are, deliberately: one function would be tidier, but it would touch three stages and
their reports for no new number.

A worked example of the incomparability, so the point is not abstract: at the class width 21 of 94
validation acids are in P on **flat** keys, where stage F reports 17 on **stereo** keys. Flat keys
merge stereoisomers, so a flat comparison can only ever match more. Both numbers are right.

**The figure the paper needs** — how many validation pairs have *both* substrates in P at all — is
`validation_pairs_by_substrates_in_P_*` in the inventory-U report. Re-read against the rebuilt P on
2026-10-29:

    uv run python scripts/unlabeled.py amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived \
        --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

| width of P | both substrates in P | one | neither |
|---|---|---|---|
| class, all conditions (what the model trains on) | **18** | 602 | 1,995 |
| HATU family | 0 | 28 | 2,587 |

of 2,615 validation pairs; flat-key substrate overlap at the class width is 21 of 94 acids and 6 of
112 amines. This section carried "50 of 2,615" until 2026-10-29. That figure was measured on
2026-10-22, before the 20 % cut-off took P from 28,158 reported pairs down to 11,396 positives, and
it is 18 now. The same rebuild moved the inventory U from 284,595 pairs to 284,600 and the lab pairs
in P from 7 to 2, because a smaller P subtracts less.

## Open questions

1. **Decided 2026-10-21, reversed 2026-10-29:** P takes a 20 % yield cut-off
   (`min_yield_pct: 20` in `configs/classes/amide_coupling.yml`, with its three citations). What
   changed is the definition rather than the arithmetic below: a reaction counts as working at 20 %
   yield or above, so a pair reported at 15 % becomes a **negative** instead of a discarded row,
   and a pair with no usable yield becomes **unlabeled** instead of a positive. All three labels
   are written to `reported.parquet`, and a yield outside (0, 100] is rejected under its own reason
   (`yield_out_of_range`) before the cut-off is applied. Counts after the rebuild are in the
   journal, 2026-10-29; at the HATU width P went from 1,177 to 627.
   The 2026-10-21 reading is kept because the earlier numbers in this file assume it: a cut-off
   drops a pair reported at 15 % while keeping the 37.6 % that report no yield at all, and it
   changes P by at most 6.6 % (1,177 pairs with no cut-off, 1,145 at 10 %, 1,099 at 20 %).
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

**Two steps.** S1 are the baselines B0 to B8, which fit nothing. S2 are the PU models, which fit a
learner and need a U choice, a P:U ratio and a learner. They are two scripts over one shared
pair-assembly function, because one stage would need a branch between a pure function of the tables
and a fit-then-predict, and ADR 006 item 5 runs them weeks apart. They write the same columns, so
stage E reads one layout. **S1 is built first** (Luka, 2026-10-21); the S2 inputs go to ADR 008.

**Code.** `src/platepu/score/pairs.py` (pair rows, the feature matrix `pair_features`, the
fingerprint-only matrix `pair_matrix`, and `substrate_index`, the one key lookup, which
`score/similarity.fingerprints` now calls instead of holding its own copy), `score/similarity.py`
(the stage's one per-side substrate similarity, public since 2026-10-30 because B2/B3/B8, the pair
K-NN probe and ADR 023's base kernel are three callers) and `score/baselines.py`;
`scripts/score_baselines.py`. Stage S2b adds `score/pu_scores.py` and `scripts/score_pu_scores.py`, both written and run (2026-11-02); S2a's `scripts/score_pu.py` is written and run too. Stage S2a adds four modules: `score/pu.py` (the spy step: `spy_labels`,
`spy_threshold`, `reliable_negatives`), `score/pu_config.py` (`PuConfig`, its loader, `config_sha256`),
`score/pu_fit.py` (the one place the learner is constructed, shared with S2b) and `score/pu_audit.py`
(`score_deciles`, `selection_shift`). `score` imports `features.blocks` and, in `score/pu.py`,
`io.strictest_licence` (CLAUDE.md §5 puts `io` below all stages); the script imports `io` and
reads the positives, library and benchmark tables, and hands DataFrames in. Nothing imports `score`.

**Inputs.** The feature table `private/features/<class>/substrates.parquet`; the positives at both
widths (`family = "<family>"` and `"all"`, ADR 006 item 1); U, `private/unlabeled/<class>/<family>.parquet`;
and the validation file, read with an explicit column list so that `label` and `conversion_mean`
never enter the process. The script refuses any file not named `validation.parquet`. A pair whose
substrate key is missing from the feature table, or a mixed `std_version`, raises `ValueError`;
coverage was counted at 100 % for all four inputs, so a miss is input we have not seen.
P takes a **20 % yield cut-off** (Luka, 2026-10-29; the line said "no yield cut-off" from
2026-10-21 until then, see the open questions above). A pair whose largest usable yield is 20 % or
above is a positive, below 20 % a negative, and with no usable yield unlabeled; S1 scores the
positives, as before.

**Output.** `private/scores/<class>/<family>/<stem>.parquet` and `.report.json`, under `private/`
— `baselines.parquet` for S1, which holds every baseline in one table rather than one file per
model; the line said `<model>.parquet` until 2026-10-28 and never matched the code —
because the inventory pairs are internal (ADR 004); every row carries `licence`, so `write_table`
enforces the path from the column. One row per scored pair: `score_set` (`validation` or
`inventory`), `key_a_stereo`, `key_b_stereo`, `score` (float32, higher is better, comparable only
inside one `score_set`), `model`, `feature_block`, `p_width`, `seed`, `config_sha256`,
`std_version`, `licence`. `feature_block` is `morgan` for B2 and B3, `morgan_centre` for B8, **null**
for B0, B1 and B4 to B7 — null and not a placeholder string, so a baseline that reads no fingerprint
cannot be mistaken for one that does — and the configured block for a stage S2 model, so one row is one
(model, feature block, scored pair) and stage E needs no lookup of which model used what.
There is **no rank column** (Luka, 2026-10-21). A rank is derived, it is not comparable between
models, and its meaning depends on a tie convention; stage E and the plate step each compute the
ranking they need. The report records the tie structure instead, because the coarse baselines are
heavily tied: on the inventory 90.1 % of the aniline pairs share one B5 score (110,560 of 122,727,
`scratch/count_stage_s_inputs.py` section 3, re-counted 2026-10-22), so any ranking of them
is a decision about ties and not about chemistry.

**From substrate rows to a pair row.** Two functions in `score/pairs.py`, not one.
`pair_matrix` returns only the chosen fingerprint block, the acid's columns and then the amine's,
float64, one row per pair, and refuses any non-fingerprint block by name (ADR 008 item 1); the PU
models use it. `pair_features` is the wider function described next. It joins the feature table twice, on
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
  284,595 (before the 2026-10-29 rebuild; now 284,600) x 1,177 pair combinations at HATU width, and x 28,158 at class width, chunked at 5,000
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
- **B8 nearest positive pair on `morgan_centre`** (declared 2026-10-23, **before** stage S2 ran).
  B2's definition on the rooted fingerprint. Measured **0.7115** on validation against B2's 0.6317,
  from the same production code path, on the pre-cut-off positives; after the rebuild of 2026-10-29 the
  pair is **0.7221** and **0.632**. It sits below B4 (0.743), so it is not the binding bar; it
  exists to isolate what *fitting* adds over the same fingerprint with none. It is a **new** baseline
  and not a redefinition of B2, which ADR 012 item 4 forbids because B2 is named in a
  pre-registration.
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

**Stage S must not** compute a metric **against a benchmark label**, read a label or any
`test.parquet`, choose between configurations, or search hyperparameters. The adversarial AUC that
ADR 008 item 7 requires beside every model is computed from stage F's `in_positives` and
`in_library` columns and touches no label, so it belongs in the stage's own report.

**Verification.** Tests first in `tests/test_score_pairs.py`, `tests/test_score_baselines.py` and
`tests/test_score_script.py`, on open hand-written rows. A full run (seed 20261021, `std_version` 2)
scores 2,615 validation pairs and 284,595 inventory pairs (before the 2026-10-29 rebuild; now 284,600) under ten (model, width) groups, 2,872,100
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
  pairs are: 406 on stereo keys and 669 on flat keys at the class width (measured 2026-10-22 against the
  pre-cut-off P; superseded by the counts in "S2b", 2026-11-02).
- Narrowing P from the class to the HATU family hurts B2: B2 at the HATU width is ahead of B3 at the
  HATU width in only 18.6 % of resamples. The interval still covers zero.
- A paired comparison here resolves about **0.05 AUC**. That is the size of edge a PU model has to
  show over a pre-registered baseline before I call it real.
- The amines carry most of the uncertainty. Resampling amines alone gives a half-width of 0.083 for
  B6 minus B2; resampling acids alone gives 0.029.

### Stage U — the literature-built unlabeled space (agreed 2026-10-23)

Carries out ADR 008 item 4: U comes from the substrates that appear in the literature positives, never
from the inventory. The inventory enters only at scoring time, with the adversarial AUC beside the
score. `library/unlabeled.py` keeps building the inventory grid; since ADR 008 item 4 that grid is a
**scoring set**, not training data, and the two must not be confused in a table or a report.

**Why the stage is a sampler.** The class-wide cross-product is 11,543 acids × 9,959 amines =
**114,956,737** pairs against 28,158 positives (`scratch/count_u_space.py`), so enumeration is not an
option and *how* it is sampled is the whole design. Two shortcuts had to be closed, and both were
measured before anything was written.

**First, popularity.** 67.4 % of acids and 67.3 % of amines appear in exactly **one** reported pair.
Draw U uniformly and a substrate reported 321 times is in P 321 times and in U about as often as one
reported once, so a classifier separates P from U by how often a molecule gets reported rather than
by whether the pair fits. Matching U's per-role marginals to P's removes it: a probe that sees
nothing but each side's literature frequency scores **0.511** against a matched U and **0.892**
against a uniform one of the same size.

**Second, the pair structure.** The literature is strongly disassortative in size —
`corr(size_acid, size_amine) = −0.388` — because a complex fragment is coupled to a simple building
block. Drawing both sides independently destroys that (U comes out at +0.003), and a classifier given
**two numbers**, the heavy-atom count of each side, then separates P from U at **0.640**: above B2
(0.632) and B7 (0.598), reached with no chemistry at all. So each counter-pair replaces **one side**,
drawn from the **same heavy-atom decile** as the substrate it replaces, weighted by P frequency; half
the draws perturb the acid and half the amine. Measured: the joint probe falls to **0.510** and U's
correlation to **−0.390** (ADR 015 item 1).

This does not make U bias-free and nothing can — P's joint structure is shaped throughout by what
people chose to publish. It removes the shortcuts that could be **named and measured**; the rest are
reported.

**Code.** `src/platepu/positives/unlabeled.py`, `scripts/unlabeled_literature.py`. It sits in
`positives/` because its only input is the positives table and its output is a counterfactual of that
table built from its own marginals. It imports nothing from `platepu` but `io`, so the one-way rule
holds.

**Config,** one YAML per configuration in `configs/models/`: `p_width: all`, `k: 10`, `seed`,
`subtract_p_on: key_flat`, `match: size_stratified_one_side`, `size_bins: 10`. The k = 5 and k = 20
sensitivity runs are two more files and the same code; the config hash distinguishes them.

**Identity.** Draws are on **stereo** keys, because every pair table in this project joins on them.
**P is subtracted on the flat pair key** (ADR 013 item 2): the fingerprints are chirality-blind and
isotope-blind, so two stereo variants are one input to the model, and the error is asymmetric —
removing one pair too many costs one candidate, leaving a known positive in U teaches the model that
a published reaction is unknown. **U is not deduplicated against itself** (ADR 015 item 2): about
19 % of draws repeat a pair at k = 10 and repeats concentrate on popular substrates, so deduplicating
would pull exactly those below their P frequency and reopen the first shortcut in reverse. A repeat
is legitimate weight and is carried in `n_draws`.

**No top-up loop.** k × |P| draws are made once and rejects are not replaced: a drawn pair is already
a known positive **1.11 %** of the time, so the loop would be a branch for a case the data does not
have (CLAUDE.md §8). The report carries the realised deficit.

**Output.** `<root>/unlabeled/<class>/<config-stem>.parquet`, `.rejected.parquet`, `.report.json`,
**outside `private/`**: every substrate comes from the open positives, so the whole training side of
the model is publishable and only the scoring side is internal. Columns: `key_a_stereo`,
`key_b_stereo`, `key_a_flat`, `key_b_flat`, `substrate_a`, `substrate_b`, `n_draws` (int16),
`licence`, `std_version`. Rejected reasons: `pair_in_p_flat`, and nothing else — a key is never under
both roles, so an acid cannot be drawn as its own amine.

**Report.** The P side (pairs, acids, amines, cross-product, the single-partner share this stage
exists to neutralise); the draws (k, seed, requested, rejected by reason, out, realised P:U); marginal
fidelity per role at both key levels (total-variation distance, the largest per-substrate deviation
and which substrate, Spearman between P and U counts); the single-partner share recomputed on U, which
must land on P's; U pairs also in P on stereo and on flat keys, both of which must be 0.

**Verification,** tests first on a hand-written open P with a deliberately lopsided popularity:

1. **The gate: the marginal.** The **rank**-AUC of one side's raw literature frequency must be
   about 0.5 (measured **0.5036** on the real draw), and each substrate's U count must match k times
   its P count within a tolerance fixed before the run. This is what frequency matching promises.

   The measure is the rank-AUC of the feature itself, **not** a classifier fitted on it. On a single
   monotone feature with classes at 1:9.7 and near-identical distributions there is nothing to fit,
   so a gradient booster fits fold noise: on this same draw it returned **0.377**, an AUC below 0.5
   on a feature whose own rank-AUC is 0.504. That is a property of the instrument, not of U. A model
   is needed only where the probe has more than one feature and cannot be ranked directly.
2. **Diagnostics, reported and never pass/fail** (ADR 017). The **joint** popularity probe — both
   sides' frequencies — reads **0.597**; the **joint** size probe, the two sides' heavy-atom counts
   and **not their sum**, reads **0.536**. The sum is blind to this and reads 0.498 even on an
   independently drawn U, because anti-correlated sizes sum to about the same as uncorrelated ones.
   Both are classifier-based, because a two-feature probe cannot be ranked directly; a value near
   0.5 from a classifier therefore carries the noise described above and is read as "nothing much",
   not as a measurement to three decimals. The controls are what give them force: the same joint
   popularity probe reads 0.892 against a uniformly drawn U.
   These numbers do not gate the stage, because a U whose joint structure matched P's in every
   respect would **be** P: something must differ and a classifier with enough features will find it.
   They are reported so that what a model could reach without chemistry is on the page beside what it
   did reach. Part of the residual is unavoidable: subtracting known positives takes 5.8 % of the
   top-30 acids' draws and 9.1 % of the top-30 amines', against 0.6–1.2 % for single-partner
   substrates, because a popular substrate has more known partners to collide with.
3. **The marginals as an assertion**, per substrate against P's counts, under a tolerance fixed before
   the run.
4. **The shuffled-P probe.** Permute P's amine column to get P′, which has P's exact marginals and no
   real pairing, and fit the same classifier on P′ against U. If it separates P′ from U about as well
   as it separates P, then what the model learns is not the pairing.
5. **Determinism**, and a grid image of 20 drawn pairs for a human to confirm they are plausible
   unreported pairs rather than nonsense (§6).

**Stage U must not** read any benchmark table, read the inventory, or subtract the validation pairs
from U — that last would use Chemlex labels to build training data. The overlap is counted in stage E
instead; only 50 of the 2,615 validation pairs have both substrates in the literature at all.

### Stage S2 — the PU models (agreed 2026-10-23)

The method is **the spy technique**, as PAYN used it (ADR 013 item 3): hide a random subset of known
positives in U, train a classifier on P against U, set the threshold so about 95 % of the spies score
positive, call everything in U below it a **reliable negative**, and fit the final model on P plus
those negatives.

**Only the pair model is fitted.** ADR 008 item 5 asked for a substrate-level competitor; under
matched marginals a one-sided classifier has identical class-conditional inputs and scores **0.5 by
construction**, so it cannot answer the question. The question already has its answer: B2 sees the
pair, B3 sees the sides separately, and Owen's bootstrap could not separate them (+0.010, interval
[−0.070, +0.084]). On this data the pair cannot be shown to matter, and that is reported as the result
it is (ADR 015 item 3).

**Two sub-steps, two tables,** because the reliable negatives are the part most in need of auditing.
**S2a** fits the spy classifier and writes one row per U pair and per spy with its score and its flag;
**S2b** fits the final model on P plus those rows and writes the stage S score layout.

**Spy scores are taken out of fold** (5 folds), with in-sample run beside it as a secondary
configuration (ADR 016 item 1). A spy sits in its own training set labelled unlabeled, so a flexible
learner can partly memorise it; every U row is in there with the same label, so the direction of the
contamination is **not known and not bounded**. Out of fold removes the question instead of answering
it, and the in-sample run makes the size of PAYN's choice a measurement rather than an assumption.

**A plain P-against-U classifier runs as a secondary control** (ADR 016 item 2): the same classifier
with no spies and no threshold, its raw score used as a ranking. It answers what the spy step bought,
which nothing else can.

**Features are the fingerprint and nothing else.** ADR 008 item 1: hand-written chemistry lives in a
baseline and in the reporting, never in the model. The ten descriptors, the amine subclasses and
`n_heavy_atoms` are all out, and the only legal `feature_block` values are `morgan_centre` (primary,
ADR 014 item 2) and `morgan` (secondary). Each block gets its own full run, spies included, because
the reliable negatives depend on the block. Nothing is one-hot encoded: `a_subclass` is null for all
11,796 acids by design, so encoding it would add a constant column that looks informative.

**The learner** is `HistGradientBoostingClassifier` — scikit-learn is already a dependency and our
matrix has no categorical columns at all, so CatBoost's distinctive strengths do not apply (ADR 013
item 3). **`early_stopping` is written out as `false`** with a fixed `max_iter`: scikit-learn
otherwise enables it above 10,000 rows and carves an internal validation split with its own seed, and
our training set is about 287,000 rows. That is a tuning decision taken by the library on data we
chose with a seed we did not set, which §5 calls hidden state (ADR 016 item 3). `learner` is a
recorded value and not a branch; any other value raises.

**Config** in `configs/models/`, hashed into every row: `model`, `feature_block`, `p_width: all`,
`unlabeled` (the U config stem), `spy_fraction`, `spy_recall: 0.95`, `spy_scoring`
(`out_of_fold` | `in_sample`), `n_folds`, `learner`, `learner_params`, `seed`.

**S2a output.** `<root>/pu/<class>/<config-stem>.negatives.parquet`. Columns: the four
pair keys, `in_p` (true for a spy), `weight` (1.0 on a positives row and on a spy, `n_draws` on a U row; ADR 029), `fold`, `spy_score` (float32), `reliable_negative`, `threshold`,
plus `model`, `feature_block`, `seed`, `config_sha256`, `std_version`, `licence`. The table carries
**one** licence for all rows, the strictest of P's and U's (`io.strictest_licence`), as
`scripts/score_baselines.py` already does for stage S1. Today that is `open`, because P and U are
100 % `open` (12,023 rows in the whole positives table, both family widths, the class width the model trains on being 11,396; 102,200 rows in U; counted 2026-10-30). **Open question for Luka and
data-guardian, not a decision:** this does not close a gap that S1 shares. The feature table's own
licences are not considered, and of the 10,641 substrates the fit reads, 535 are `internal` and 16
`noncommercial`. The same reasoning applies (incomplete, not wrong, until a private-only substrate enters
P or U); it is written under "Licence" in the S2b section. Its report is the
audit record: spies, configured and realised recall, both thresholds, the number of reliable negatives
and their share of U by rows and, as `reliable_negative_share_weighted`, by draws (ADR 029), the spy and U score deciles, and **the shift the selection introduces** — the
marginal distance of the reliable negatives against U as a whole, their mean heavy-atom count against
U's, their single-partner share (the share of a set's distinct substrates that appear in exactly one of
its pairs, both roles pooled). U was built to match P's marginals and the threshold breaks that
match on purpose; how far it breaks it is the number a later reader will want.

### S2b — the final fit and the two arms (designed 2026-11-02, ADR 028, ADR 029)

**Two arms, both blocks, four runs.** The spy arm (primary, ADR 014 item 1) fits P against the reliable
negatives. The naive arm (secondary, ADR 016 item 2) fits P against all of U. They differ in one thing,
where the negative class comes from. Both run on `morgan_centre` and on `morgan`, so S2b is four runs
and four config files: `pu_score_spy_morgan_centre.yml`, `pu_score_spy_morgan.yml`,
`pu_score_naive_morgan_centre.yml`, `pu_score_naive_morgan.yml`. Each is one fit and one table. I chose
the extra fits on 2026-11-02 (ADR 028 item 6).

**Config.** S2b has its own record, `PuScoreConfig`, in `score/pu_config.py` beside `PuConfig`. It has
eight keys and none is inert in either arm. The naive arm needs no `spy_fraction`, `spy_recall`,
`spy_scoring` or `n_folds`, and a file that carried them would claim spies were drawn.

    model: pair_hgb_spy                    # pair_hgb_naive in the naive arm (stage E needs no lookup)
    feature_block: morgan_centre           # morgan | morgan_centre, as in S2a
    p_width: all                           # the class width, 11,396 positives
    unlabeled: unlabeled_literature_k10
    negatives: pu_spy_morgan_centre        # the S2a config stem; null in the naive arm
    learner: HistGradientBoostingClassifier
    learner_params:
      max_iter: 200
      early_stopping: false
    seed: 20261102

`negatives` picks the arm. A stem names the S2a config whose `<stem>.negatives.parquet` supplies the
negative class. `null` means this run reads no such table and trains against all of U, the same
convention S1 uses for `feature_block`.

**The pointer is checked.** S2b loads the named S2a config. It raises unless `feature_block`, `p_width`,
`unlabeled`, `learner` and `learner_params` agree with its own, and unless that record's value hash
equals the single distinct `config_sha256` in the negatives table. A table cut under `morgan` is never
fitted under `morgan_centre`. The spy settings stay traceable through that hash and are not copied.

**Training set, exactly.**

| arm | positive class | negative class | weight |
|---|---|---|---|
| spy | the 11,396 positives of P **plus the spies**, which rejoin P | the reliable negatives of the S2a table | 1.0 per positives row; `n_draws` per U-derived row |
| naive | the 11,396 positives of P | every row of U | 1.0 per positives row; `n_draws` per U row |

The spies rejoin P because the draw existed only to place a threshold, and a spy is a reported
positive. `reliable_negative` is never true on a spy, so no row enters twice; the script asserts it by
checking that the flat pair keys of P and of the reliable negatives are disjoint. Every fit is weighted
and a spy weighs 1.0 (ADR 029 items 1 and 2). The P:U ratio is quoted by rows and by draws.

**Scored sets.** `validation` (2,615 Chemlex pairs) and `inventory` (284,600 lab pairs): 287,215 rows per
run. The literature U is **not** scored. In the naive arm it is the training negative class; in the spy
arm it is the pool the negatives were cut from.

**One fit, no cross-fitting.** Some scored pairs are in training. Counts, `uv run python scratch/count_s2b_inputs.py ~/University/Phd/Dataset/Datasets`, run 2026-11-02
(output in `scratch/count_s2b_inputs.out.txt`):

| scored set | in | keys | pairs | share of the set |
|---|---|---|---|---|
| validation | P | stereo and flat, both widths | 0 | 0 |
| validation | U | stereo and flat | 1 | 0.038 % of 2,615 |
| inventory | P, class width | stereo | 208 | 0.073 % |
| inventory | P, class width | flat | 267 distinct pairs in 335 rows | 0.118 % |
| inventory | P, HATU width | stereo | 0 | 0 |
| inventory | U | stereo | 1,587 | 0.558 % |
| inventory | U | flat | 1,925 distinct pairs in 2,307 rows | not given |

The one validation pair in U carries `n_draws` 1 and matches on flat keys too. Its substrate keys
are deliberately **not** written here: a Chemlex row's identity is a row of a CC BY-NC-ND set, which
CLAUDE.md §9 and ADR 009 keep out of anything we share, and the count above is what a reader needs.
It is recoverable from the two tables with the command in this section. The flat inventory has
duplicate rows, which is why rows and distinct pairs differ.

The 406 and 669 of 2026-10-22 are superseded. They were measured against the pre-cut-off P and are not
comparable, not wrong. The likely cause, which I have not verified, is that `positives.parquet` was
rebuilt on 2026-10-29 and the 20 % cut-off took P from 28,158 reported pairs to 11,396 positives (the split is recorded, with its command, in ADR 024, Context), so a
smaller P subtracts less; the same file records that the rebuild moved the lab pairs in P from 7 to 2.
The old "about 0.14 %" is withdrawn; the figure is 0.073 % on stereo keys.

**These pairs are reported and never removed.** The validation pair sits in the training negative class
in the naive arm and in the pool the negatives are cut from in the spy arm. Filtering U by membership in
the validation split would use the evaluation set to decide what may be trained on, the same leakage
turned around as in "The substrate overlap is reported and never removed". The 1,587 inventory pairs in U
are scored by a model that trained on them as the negative class, 1,587 times the situation of the single
validation pair, on the set the plate is picked from. They are counted in the report, not removed.
**Effect on the plate, to report:** those pairs are trained on as the negative class, so their scores are
pushed down, and the plate is picked from the top of the inventory ranking. U was drawn from P's own
substrates, so these 1,587 are pairs with literature-like substrates, the ones a "looks publishable"
model would otherwise rank highest. The effect is therefore not a harmless nudge on an AUC: it steers the
plate systematically away from 0.56 % of the grid. **Open for Luka, not decided here:** whether anything
should be done about it. Removing them from U would use a scoring set to shape the training data, the same
class of error as removing the validation pair. For the validation pair the bias is conservative: a pair told "not known to work" is scored lower, so a Chemlex success among them
pushes the validation AUC down, not up. The inventory pairs in P get an in-sample score. Dropping any of
these would filter a scoring set; cross-fitting only them would put two kinds of score inside one
`score_set`.

**Output.** `<root>/private/scores/<class>/<family>/<config-stem>.parquet` and `.report.json`, under
`private/` because the inventory pairs are internal (ADR 004). The columns are **S1's eleven, in S1's
order** (`score_set`, `key_a_stereo`, `key_b_stereo`, `score`, `model`, `feature_block`, `p_width`,
`seed`, `config_sha256`, `std_version`, `licence`), `score` float32. `feature_block` is never null for
a PU model. There is no rank column.

**Licence.** Per scored set, widened by the training side through `io.strictest_licence`: the validation
rows carry `noncommercial` and the inventory rows `internal`, as `scripts/score_baselines.py` already
writes them. **Open, a question for Luka and data-guardian, not a decision:** the feature table's own
licences are not in the merge (535 `internal` and 16 `noncommercial` of the 10,641 substrates the fit
reads). The gap is the one S1 and S2a share, and it bites harder here, because the fitted model then
scores the plate. **Assessed 2026-11-02 (pr-reviewer): incomplete, not wrong,** for a structural reason.
The matrix the fit sees is only the fingerprint rows of the P and U pairs, and both tables are built from
the open positives, so every number entering the fit is a Morgan count of a molecule in an open corpus.
The `internal` (535) and `noncommercial` (16) stamps on those feature rows record co-occurrence with the
lab inventory or with validation; that fact never reaches the fit, because the script reads `key_stereo`,
the block, `n_heavy_atoms` and `std_version`, and not `in_library` or `in_validation`. **It becomes wrong
the day a substrate whose structure comes only from a private source enters P or U.**

**Report.** Provenance and the two config hashes (the S2b record's and, in the spy arm,
`spy_config_sha256`); `arm`, `negatives`; `n_positives`, `n_negatives`, `negative_source`; the P:U ratio
by rows and by draws and `n_draws_sum`; rows in and out per scored set, and the number of scored pairs found in P and in U on each key type (287,215 = 2,615 + 284,600, arithmetic on the two set sizes; the 284,600 comes from `scripts/unlabeled.py`, command at the 2026-10-29 rebuild note in "Stage U" of this file, around line 345). Beside every model, as ADR 008
item 7 requires: the adversarial ROC-AUC of the block used, per role, with its bootstrap interval and
size-adjusted value (intervals about ±0.04, so a gap under 0.04 is not a gap, ADR 012 item 2); the mean
heavy-atom count of each scored set; and B7 as a **pointer** to the baselines report, not a recomputed
metric. The label-free diagnostics: score deciles, the tie structure, the Spearman correlation of score
against pair heavy atoms, and the scored-in-training overlap on stereo and flat keys.

**S2b must not** compute a metric against a label, read a label or any `test.parquet`, choose between its
four configurations, or search hyperparameters. The validation ROC-AUC is read afterwards by a count
script, as `scripts/counts/count_baseline_aucs.py` does for S1.

**Verification,** tests first, on hand-written open rows:

1. A config with `negatives: null` loads; one carrying `spy_fraction` raises as an unknown key.
2. A spy-arm config that disagrees with its named S2a config raises; so does a negatives table whose
   `config_sha256` is not that config's.
3. A negatives table with two distinct `config_sha256` values raises.
4. A planted separable case in which every scored pair's side of the boundary is written by hand.
5. S1's eleven columns in order, `score` float32, and the two licences.
6. A P row that is also a reliable negative raises.
7. A missing substrate and a mixed `std_version` raise.
8. Two runs give identical tables and reports but for timestamps.
9. The full run reproduces 11,396 positives and 287,215 rows out.

**Status, 2026-11-02.** Code written: `score/pu.py`, `pu_config.py`, `pu_fit.py`, `pu_audit.py`,
`pu_scores.py` and `scripts/score_pu.py`, `scripts/score_pu_scores.py`. S2a run on all three feature
blocks: `<root>/pu/amide_coupling/pu_spy_{morgan_centre,morgan,morgan_near}.negatives.parquet`. S2b run on
all six configurations: `<root>/private/scores/amide_coupling/HATU_DIPEA_DMF/pu_score_{spy,naive}_{morgan_centre,morgan,morgan_near}.parquet`,
287,215 rows each. The validation read was spent once, by `scripts/counts/count_pu_aucs.py`; its six
numbers are in the journal entry of 2026-11-02 (the day's last entry) and are not repeated here.

**Stage S2 must not** compute a metric against a benchmark label, read a label or any `test.parquet`,
choose between configurations, or search hyperparameters. The strata, the comparisons and every
interval are stage E. The evaluation is ADR 008 item 6, a pre-registration, and is not touched.

**Verification,** tests first on hand-written open rows: a spy set whose threshold can be computed by
hand; a planted separable case where the reliable negatives must be exactly the planted ones; a matrix
that must contain no descriptor column, and an input carrying one, which must raise; realised spy
recall equal to 0.95 under the written tie rule; two runs identical.

### The information axis — the Setting-4 protocol (agreed 2026-10-30, ADR 023)

ADR 018 item 3 asked for the ROC-AUC of ranking P above U, cross-fitted over 5 folds grouped on the acid
flat key, and ADR 019 named it PAUC and added a bootstrap interval. The fold rule does not survive
measurement: under it **100 % of test pairs have an amine that appears in training**, at all 5 folds and
all 5 CV seeds, for P and for U, and **no** test pair has both sides unseen. 4,658 amines carry 113,596
pairs, so the rule cannot produce an unseen amine at any seed. That is Setting 3 of Viljanen, Airola and
Pahikkala, "new acid, known amine". Our deployment is their Setting 4, a new class with new substrates
on both sides. ADR 023 replaces the rule. This section is what gets built.

**This is not a pipeline stage.** It fits models to choose a column; it writes no stage table, and no
stage reads it. It is a protocol, run once, whose output is a decision recorded in an ADR and in the
journal.

**Inputs.**

| table | columns |
|---|---|
| `<root>/positives/amide_coupling/positives.parquet`, `family = "all"` | `key_a_flat`, `key_b_flat`, `key_a_stereo`, `key_b_stereo`, `n_reports`, `std_version`, `licence` |
| `<root>/unlabeled/amide_coupling/unlabeled_literature_k10.parquet` | the same four keys, `n_draws`, `std_version`, `licence` |
| `<root>/private/features/amide_coupling/substrates.parquet` | `key_stereo`, `role`, `smiles`, `std_version`, restricted to keys that occur in P or U, so no library row and no `internal` licence enters the protocol |

The seven candidate blocks are computed from `smiles` by the same code as
`scripts/counts/count_centre_distance.py`, so stage F is untouched until a `d` wins (ADR 018 item 6).
No benchmark table is opened: not `labels.parquet`, not `validation.parquet`, and never any
`test.parquet`. A mixed `std_version` across the three inputs raises.

**The split (ADR 023 item 1).** The distinct acid flat keys of P ∪ U go into 3 near-equal parts and the
distinct amine flat keys into 3, independently, under partition seed `20261030`. 3 × 3 is fixed, not
2 × 2 or 4 × 4. For each of the 9 cells `(i, j)` the **training block** is every pair whose acid is
outside part `i` and whose amine is outside part `j`; the **test cell** is every pair whose acid is in
part `i` and whose amine is in part `j`; every mixed pair is **discarded**. Counted at 3 × 3: 44 %
discarded, training block about 51,100 pairs, test cell about 12,400; over 20 partition seeds the
smallest cell held 1,025 P and 9,776 U pairs over 1,821 acids and 1,517 amines. Flat keys, because the
fingerprints are chirality-blind. A hyperparameter, if any survives ADR 023 item 3, is chosen on the
training block split again 75/25 **by the same rule**, and the inner split never sees the test cell.

**The estimand changes.** Because of the discard, about 44 % of P is never scored. PAUC here is measured
on the Setting-4 subpopulation of P and not on P, and ADR 018 item 3's "every positive is scored out of
fold" is no longer true. The paper states this. The report carries the share of P scored and compares
the scored with the discarded positives on literature frequency and heavy-atom count.

**U is reweighted (ADR 023 item 2), in the training block and in each test cell.** U's marginals are
matched to the **whole** of P (ADR 013 item 1, ADR 015), so an in-block P is a subsample and the match
degrades. Measured: ranking P above U by **in-block** frequency reads 0.538–0.570, where by **full-P**
frequency it reads 0.495–0.516, which is stage U's own gate (0.5049 acid, 0.5095 amine). 0.04–0.07 above
chance is three times the gate's 0.02 tolerance, in every cell and at every seed, so the shortcut the
sampler closed reopens inside a block. Raking U's `n_draws` (iterative proportional fitting, 200 passes,
target = each flat key's P count in the same rows, scaled to U's mass there) takes the marginal
total-variation distance from 0.17–0.24 to **≤ 0.0009** and the frequency rank-AUC to **0.4999–0.5000**
(measured on training blocks; the test cells are not yet counted). Both sets are raked: if only the block
were, the test cell would keep the mismatch and the figure would be partly a frequency reading. PAUC is
a weighted AUC on the raked weights, and the residual TVD is reported for both. The price is counted and
reported, never hidden: **23–29 %** of block U pairs get weight 0, and U's effective sample size falls
to about **0.68**. In sample at `d = all`, where 10,943 distinct fingerprints over 11,345 substrates give
near-perfect memorisation capacity, the reweighted arm reads **0.5001**. That shows the one-sided
marginal shortcut is gone. It does not show U is clean, and nothing yet shows the raking leaves the
real pair signal alone. That is why control 3 below is a gate.

The weights live in the protocol, not in stage U's table. A block weight is a property of a split and
depends on the partition seed; writing it into `<root>/unlabeled/` would make a stage table depend on a
fold seed and hand stage S2 a column it must not use.

**The estimator (ADR 023 item 3).** The Kronecker product pairwise kernel with a normalised, binarised
per-side Tanimoto base kernel, in weighted kernel ridge regression, P at 1 and U at 0, weights
`n_draws × rake weight`, solved by conjugate gradient with the generalised vec trick. Our pair set is not
a complete matrix (113,596 observed of 6,138 × 5,015 possible), so the closed form does not apply. The
implementation is our own, about one module and one test module. RLScore (MIT) is a reference only: the
fit is checked against it in an isolated environment on SciPy 1.14.1, and it is never a dependency,
because it installs on Python 3.13 only from GitHub master and then fails on SciPy 1.18.1. The estimator
here is an instrument comparing seven descriptors, so its sensitivity must not vary with what it
measures, and the ladder spans 3.0 to 34.2 bits and 615 to 10,943 distinct fingerprints. A boosted tree
underfits 4,096 sparse columns at 120 iterations (ADR 020), so its reading depends on a candidate's
sparsity as much as on its information. Poly2D adds the per-side second-order blocks the raking just
nulled. A Gaussian pairwise kernel is the same Kronecker form with a bandwidth that would need rescaling
per candidate. Tanimoto is a ratio and reads the same way along the ladder; normalised per side, the
Kronecker kernel has unit diagonal and no scale parameter. `λ` and the iteration budget are fixed across
all seven candidates, chosen once on the inner split and recorded before any PAUC is read; the
per-candidate inner-split values are a reported diagnostic. **Two claims are verified before the axis is
read, not asserted:** that a product of PSD kernels is PSD, and that Tanimoto on our vectors is PSD, by
the minimum eigenvalue of our own per-side and pair Gram matrices.

**The readings per candidate `d`.** PAUC per cell, averaged over the 9 cells, as the primary number;
IPM beside it (ADR 019 item 5); the model-free reading beside both (ADR 018 item 3), under the same split
and the same weights. The model-free reading has two forms: the existing **minimum** of the two sides'
Tanimoto (`score/pair_knn.py`, unchanged, so ADR 020's bar of 0.6843 does not move) and a new **product**
beside it, added the way B8 was added beside B2. A disagreement between the readings is reported as a
finding and not resolved by convenience. The interval is Owen's pigeonhole bootstrap, 5 repeats, 500
resamples, seed 20261028, resampling the acid and amine flat keys **of the test cell** (cells share no
substrates by construction) and bracketing the mean over cells. The eligibility and tie rules of ADR 018
item 4 and ADR 019 item 4 apply unchanged, parsimony by fewest distinct fingerprints as the final
tie-break.

**Controls, fixed before any selection number exists (ADR 023 item 4).** Run on the same cells through
the same code path; the axis is void if any fails.

1. Labels shuffled with their weights inside the test cell: **0.50 ± 0.02**.
2. The identity-only oracle, per-flat-key `P / (P + U)` from the training block, no fingerprint, scored
   on the test cell: **0.50 ± 0.02**. Under pair-level folds this same oracle read **0.189** on the acid
   side against 0.498 for shuffled labels, because a pair-fold removes a held-out pair's own contribution
   from its molecule's counts, so a held-out P looks less P-like. A Setting-4 split gives that artefact no
   channel, since no test-cell molecule is in the training block. That is a deduction, which this control
   measures anyway.
3. A planted cross-side interaction on the real fingerprints, two bits actually set at the candidate `d`:
   must read well above 0.5 at every candidate, threshold and bits fixed with the counts beforehand. It
   separates a real 0.5 from a dead instrument, and it is the check that the raking did not destroy pair
   signal.
4. P's amine column permuted inside the training block (stage U's shuffled-P probe, moved here): must
   fall relative to the real reading, or the candidate is not being read on the pairing.

Controls 1 to 4 gate the **instrument**, not the data, so ADR 017 is untouched: it governs probes that
judge our U, and these judge our measuring device. A first synthetic known-answer control exists (a pure
interaction with no marginal signal: `CGKronRLS` 0.9575, logistic regression on the concatenated sides
0.5658), but its script is not yet in the repository.

**Report keys,** per candidate and per cell: `pauc`, `pauc_ci_low`, `pauc_ci_high`, `ipm`,
`pauc_model_free_min`, `pauc_model_free_product`, `n_p_scored`, `n_u_scored`, `share_p_discarded`,
`u_pairs_zero_weight_share`, `u_effective_sample_fraction`, `marginal_tv_acid_block`,
`marginal_tv_amine_block`, `marginal_tv_acid_cell`, `marginal_tv_amine_cell`,
`frequency_rank_auc_before`, `frequency_rank_auc_after`, `lambda_inner_best`, `control_shuffled_labels`,
`control_identity_oracle`, `control_planted_interaction`, `control_shuffled_pairing`, `n_acids`,
`n_amines`, `partition_seed`, `std_version`, `config_sha256`, plus the package versions and the git
commit. The seed sensitivity (partition seeds `20261030 + 1 … + 4`) is reported, never selected on.

**Verification.** Tests first, on hand-written open pairs: a hand-made 6 × 6 pair set where the nine
Setting-4 cells and the discarded pairs can be written out by hand, and the split must reproduce them
exactly; a case where one side's parts are empty, which must raise; a hand-computed raking on a 2 × 2
marginal, converged in one pass; a Gram matrix on three hand-written fingerprints whose Tanimoto values
are computed by hand; our conjugate-gradient fit against a naive dense kernel on a small hand-made case
(and against RLScore in the isolated environment); the planted chessboard, which the estimator must
separate; the identity-only oracle on a block where no molecule crosses, which must read 0.5; and two
runs giving identical numbers.

Not decided here, and not built before it is: where the code lives under `score/`, whether the protocol
writes a table through `io`, and whether the winner's stage F column is added before or after the
Chemlex agreement check of ADR 018 item 5. The counts that come first are listed in ADR 023, Consequences.


## Sources

Notes with page references in `docs/literature/notes/`: `boser-2026-payn.md`, `zhong-2025-chemlex.md`,
`bekker-2020-pu-survey.md`, `2026-10-09-pu-modeling-web-research.md` (agent summaries; see their headers). To read before ADR 003: Bekker,
Robberechts & Davis, ECML PKDD 2019 (SAR), and the domain-shift PU papers above.
