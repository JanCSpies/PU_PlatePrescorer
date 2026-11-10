# 007 — Acid subclasses, one electronic descriptor, and success measured within a class

- **Status:** Accepted, 2026-10-21
- **Supersedes:** 006 in part (item 2, the feature list; item 3, the success rule) · **Superseded by:** —

ADR 006 is Accepted and is not rewritten (CLAUDE.md §8). This record supersedes the two parts of it
that changed.

## Context

On 2026-10-21 stage F was built with four amine subclasses and nine reacting-centre descriptors.
Before stage S I checked the taxonomy and the features against the literature and against our own
counts. Sources: `docs/literature/notes/2026-10-21-subclasses-and-features-research.md`,
`due-hansen-2016-difficult-amide-couplings.md`, `zhang-2025-amide-intermediate-knowledge.md`.
Counts (Chemlex validation side only, test side not opened):

    uv run python scratch/count_subclass_evidence.py ~/University/Phd/Dataset/Datasets
    scratch/count_subclass_candidates.py    (counted the candidate classes before the split; pre-split numbers)
    scratch/count_electronic_descriptor.py
    scratch/count_design_rule_proxy.py

Five findings drive this record.

1. The amine-class baseline gives every amine of a class the same score, so its ROC-AUC **inside** a
   class is exactly 0.5. ADR 006 item 3 therefore only asks a PU model to beat chance within a class,
   which is far weaker than intended.
2. If the chance that a success gets reported depends only on the amine class, every class is
   multiplied by its own constant and the ranking inside a class is unchanged. Within-class AUC is
   robust to class-level reporting bias; pooled AUC is not. This is my reasoning from the SAR
   definition of Bekker and Davis, not a cited result.
3. The acid side had no classes, yet it is where the sharpest failures are. Haas et al., PNAS 2022,
   119, e2118451119, find the acid largely sets the rate for CDI couplings. Due-Hansen et al., Org.
   Biomol. Chem. 2016, 14, 430, report hindered acids and arylacetic acids as the hard cases. On our
   validation set the acid classes run from 0.136 success (alpha_quaternary) to 0.470
   (heteroaryl_pi_donor) over the classes with at least 4 distinct acids. alpha_tertiary (0.550) and
   arylacetic (0.517) score higher but rest on 2 and 1 acids. Splitting the heteroaromatic class
   showed that the old single class was an average of two families that differ, 0.378 (azine)
   against 0.470 (pi donor).
   `uv run python scratch/count_subclass_evidence.py ~/University/Phd/Dataset/Datasets`.
4. Our cheap topological descriptors already capture sterics as well as a 3D buried-volume descriptor
   does. Single descriptor `amine.n_heavy_at_2_bonds`, ROC-AUC on validation:

   | score | ROC-AUC |
   |---|---|
   | `amine.n_heavy_at_2_bonds` | 0.711 |
   | occupied van der Waals volume, 5 Å sphere | 0.733 |
   | subclass baseline | 0.713 |

   0.711 and 0.713: `uv run python scratch/count_subclass_evidence.py ~/University/Phd/Dataset/Datasets`.
   0.733: `uv run python scratch/count_design_rule_proxy.py`.
   No conformer generation and no new dependency is needed.
5. Mayr nucleophilicities contradict the order inside ADR 006 item 3. Cyclic secondary amines are the
   strongest nucleophiles (pyrrolidine N 18.58, piperidine 17.35); tert-butylamine (12.35) is at
   aniline level (12.64). The chemist's order is a convention for the baseline, not a fact.

## Decision

1. **Acid subclasses.** `configs/classes/amide_coupling.yml` gains nine, first match wins:
   heteroaryl_azine, heteroaryl_pi_donor, aryl, alpha_vinyl, alpha_quaternary, alpha_heteroatom,
   arylacetic, alpha_tertiary, alkyl; anything else is `other`. Counted over 11,787 acids:

   | class | acids |
   |---|---|
   | aryl | 4,207 |
   | alpha_heteroatom | 2,274 |
   | heteroaryl_azine | 1,144 |
   | alkyl | 982 |
   | heteroaryl_pi_donor | 869 |
   | alpha_tertiary | 807 |
   | arylacetic | 483 |
   | alpha_vinyl | 456 |
   | alpha_quaternary | 419 |
   | other | 146 |

   Three boundaries follow one reason: a class that mixes opposite electronics is worse than no
   class. `heteroaryl_azine` (1,144 acids, 25 in the library, validation success 0.378) and
   `heteroaryl_pi_donor` (869 acids, 26 in the library, validation success 0.470) were one class
   with a `[n,o,s]` SMARTS, which put 2-furoic, 2-thenoic and indole-2-carboxylic acid with picolinic
   acid, although a ring O, S or N-H donates electrons and an azine ring N withdraws them. Luka
   agreed to the split on 2026-10-21. Azine is tested first, so thiazole-2-carboxylic acid, whose
   ring carbon touches both n and s, is azine. `alpha_quaternary` excludes a halogen on the alpha
   carbon: degree counted halogens, so trifluoroacetic acid was filed as hindered although it is
   electronically activated; those 37 acids now fall to `alkyl`. `alpha_vinyl` is a C=C only, for
   the same reason (below).

   Counts: the stage F run (Consequences), `uv run python scripts/features.py amide_coupling
   ~/University/Phd/Dataset/Datasets/derived --validation
   ~/University/Phd/Dataset/Datasets/derived_samples/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet`.
   It writes `private/features/amide_coupling/substrates.parquet` and `substrates.report.json`; the
   per-class counts come from the `subclass` column of the table.
   `alpha_vinyl` is a C=C only. The 81 alpha-keto and oxamic acids and 23 oximes and hydrazones fall
   to `other`, together with the alkynyl and carbamic acids (counted 2026-10-21).
   Every named class holds at least 21 lab compounds, so every named class can reach a plate; `other`
   holds 19. Sterics is ordered
   before the heteroatom, so 2-aminoisobutyric acid stays alpha_quaternary; Z-Aib-OH is the standard
   hindered benchmark (Valeur and Bradley, Chem. Soc. Rev. 2009, 38, 606, p. 610). `arylacetic` is
   kept although only one validation acid carries it. Due-Hansen report that class failing through a
   suspected ketene route, which no steric or electronic descriptor can express, and the library holds
   21 such acids.
2. **One electronic descriptor**, `n_ewg_on_ring`, computed by the same function for both roles:
   electron-withdrawing substituents on the aromatic ring system of the reacting atom, counted once
   per attachment atom, zero when no neighbour of the reacting atom is aromatic. The SMARTS (nitro,
   nitrile, trihalomethyl, sulfonyl, carbonyl) are in `configs/features.yml`, not in the class YAML,
   because what withdraws electrons does not depend on the reaction class. **Validation cannot check
   this descriptor.** No validation aniline carries a nitro, nitrile, trifluoromethyl or sulfonyl
   group. The whole observed effect is three anilines with a ring carbonyl, 2.6 % success against
   21.7 % for the other thirteen. It is in the table because the chemistry is settled (Hammett rho
   -2.1 in water and -2.5 in acetonitrile for aniline acylation, Brotzel, Chu and Mayr, J. Org. Chem.
   2007, 72, 3679) and about 10 % of the lab amines carry such a group.
3. **Amine subclasses stay at four plus `other`.** The counts do not support splitting the primary
   amines: success by alpha-carbon degree runs 0.49, 0.57, 0.67, the opposite of the steric
   expectation, and the last class holds one validation amine. The cyclic-versus-acyclic split of the
   secondary amines, which Mayr's values support (0.52 against 0.17 on validation), is already the
   existing descriptor `centre_in_ring`, so it needs no new class.
4. **Success is measured within a class** (replaces ADR 006 item 3).
   - Primary estimand: the pair-weighted within-class ROC-AUC over the amine subclasses, the sum over
     classes of n_positive(g) x n_negative(g) x AUC(g) divided by the sum of
     n_positive(g) x n_negative(g).
   - Secondary: the plain mean of the per-class AUCs over classes with at least 10 positives and 10
     negatives, and the pooled AUC. Both are reported, neither is decisive.
   - The amine-class baseline scores 0.5 inside every class by construction, so within a class the
     model is compared with the popularity baseline and with the steric descriptor, not with the
     class rule.
   - Uncertainty: Owen's pigeonhole bootstrap (Owen, Ann. Appl. Stat. 2007, 1, 386,
     doi:10.1214/07-AOAS122), resampling acids and amines separately rather than reactions, 10,000
     resamples with a fixed seed, every reaction weighted by the product of its two multiplicities.
     Model and baseline are scored on the same resample, so the difference is paired.
   - "Beats every baseline and passes in every class" is an intersection-union test, so each part is
     tested at the same alpha with no Bonferroni correction.
   - Power: validation holds only 5 to 16 distinct amines per class, so per-class intervals will be
     wide. The requirement is a 95 % interval above 0 on the pair-weighted within-class difference,
     plus a point estimate above 0.5 in each class, with every per-class interval reported.
5. **Baselines.** B0 to B4 stay as in ADR 006. Two more are fixed now, before any PU run:

   | baseline | definition | ROC-AUC on validation |
   |---|---|---|
   | B5 | amine subclass plus topological steric rank sum | 0.775 |
   | B6 | B5 plus `n_ewg_on_ring` | 0.793 |

   B5 and B6: `uv run python scratch/count_subclass_evidence.py ~/University/Phd/Dataset/Datasets`.

   Both were put together after looking at validation, so they are optimistic. They are written down
   anyway because they raise the bar, which is the safe direction.
6. **Zhang's activated-ester template is not adopted.** Replacing the acid with its HOAt ester before
   featurisation lifted a transformer from R2 0.39 to 0.84 at their "full novelty" split, but it made
   XGBoost on ECFP worse, and their novelty split reuses the same 136 substrates. On our 69
   validation acids the template adds 27 bits that are set in every product and raises the mean
   pairwise Tanimoto between acids from 0.145 to 0.384. Under one fixed condition it compresses the
   differences between acids instead of adding information.
7. **What the Chemlex label measures is recorded in the paper.** Its positives are reactions fast
   enough to pass 20 % conversion in 30 minutes at 30 °C. Due-Hansen report 17 % after four days at
   room temperature with HATU and DIPEA in DMF for a hindered acid with an aminothiazole, so a
   Chemlex failure can mean "slow", not "impossible". I keep Chemlex as the validation benchmark
   because its protocol is the one our plate copies.

## Alternatives considered

- **Leave ADR 006 item 3 as it stands:** a model could pass by beating a baseline that scores 0.5
  inside every class.
- **Split the amine classes further** (cyclic secondary, alpha-electron-poor primary, electron-poor
  aniline, azole against azine): the counts leave 1 to 3 validation substrates in several of these,
  and the strongest of them is already a descriptor.
- **Three-dimensional sterics (buried volume from conformers) or computed charges:** our topological
  descriptors match the 3D one on validation, and Gasteiger charges are not usable here (rank
  correlation with MMFF charges on the amine nitrogen is -0.11).
- **Reconstruct Chemlex's designed-negative flag** to score the two halves separately: the released
  file carries no such flag, only Acid, Amine, Reagents, Solvent, Products, Conversion and three
  split columns.

## Consequences

- Stage F re-run: 21,991 substrates in about 5 seconds, 10 descriptors, acid subclasses filled:

      uv run python scripts/features.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --validation ~/University/Phd/Dataset/Datasets/derived_samples/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

  Gates: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q`
  gives 247 passed, 1 skipped. `docs/design/modeling.md` was updated in the same piece of work.
- No new dependency. catboost is still open and is decided when the PU models start.
- Stage E must compute the within-class estimand and the pigeonhole bootstrap. Stage S must report B5
  and B6 beside B0 to B4.
- The acid classes and `n_ewg_on_ring` enter the model as features, and the amine classes also as
  evaluation strata. How a category is encoded per model type is written in
  `docs/design/modeling.md`, not here.
- Chemlex stays the validation benchmark; the test side is still unread.

## References

- `docs/design/modeling.md`
- `docs/literature/notes/2026-10-21-subclasses-and-features-research.md`
- `docs/literature/notes/due-hansen-2016-difficult-amide-couplings.md`
- `docs/literature/notes/zhang-2025-amide-intermediate-knowledge.md`
