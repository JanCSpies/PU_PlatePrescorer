# Lab journal

Append-only. Newest entry on top; never rewrite an old entry. Each entry, written in the first person: context → done → result →
interpretation → next. Every number can be regenerated with the commands in its entry.

---

## 2026-10-23 — Review and optimisation pass before the first commit: a yield filter that passed every missing yield, a faster B2, an ADR number that does not reproduce

**Context.** Nothing is committed yet. Before the first commit I ran four reviews in parallel: a full project code review, a silent-failure hunt, a performance review and a comment-accuracy review. This entry records what they found and what I changed.

**Done.**
- **Performance.** Baseline B2's reduction selected rows before columns and so built a 5,000 × 11,543 intermediate it never needed. It now selects the positive columns first. A reference-oracle test, `tests/test_score_baselines_oracle.py`, freezes the old reduction and asserts `np.array_equal`, not `allclose`.
- **Not taken.** A threaded variant would have reached 1.3 s. I left it out. It would be the only concurrency in the file that produces a published number. It would also read the CPU count, so the same code would behave differently on my laptop and on the cluster. That is hidden state (§5).
- **Yield filter.** It was `notna() & (yield < minimum)`, so a reaction with no recorded yield survived every cut-off. I added a rejection reason, `no_yield`, applied only when a cut-off is set.
- **Seven more silent failures closed, tests first** (16 failing tests before any code; no command for that count):
  1. The atom-mapper's output was matched to its input by position and never checked. A reordered output would have attached one reaction's mapping to another's hash permanently and invisibly.
  2. Cache part files were numbered from zero on every run, so a second source silently replaced the first.
  3. An all-failed shard was cached as "mapping failed" forever. The cache keys on the reaction hash and is never retried.
  4. `subprocess.run` discarded the child's stderr and had no timeout.
  5. An empty benchmark reported leakage as 0/0, indistinguishable from "checked, none found".
  6. A misspelled split value would have put every labelled row into validation while the seal looked intact.
  7. The labels report still averaged `positive_share` over the test side. I had believed this fixed the day before and had written a comment saying so.
- **Three blockers in my own work of 2026-10-22:**
  1. I moved `build_chemlex_like_fixtures.py` from `scratch/` into `scripts/counts/`. Its path depth broke (`parents[1]` where the new location needs `parents[2]`). The script that ADR 011 names as the evidence that no CC BY-NC-ND row is in the repository died before writing anything. That is the §9 promise unkept for the second time in one day.
  2. `scripts/counts/count_b2_vs_b3.py` still read the pre-ADR-009 benchmark path.
  3. Five known HATU-family positives sit in the unlabeled space under flat keys. U is subtracted from P on stereo keys, while our fingerprints are chirality-blind.
- **Stereo question, counted** (below).
- **Feature blocks measured on two axes** (below).
- **Re-ran the adversarial validation quoted in ADR 008 and 009** (below).

**Result.**

Stage S1, B2 reduction. Command: `uv run python scripts/score_baselines.py amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet`

| | before | after |
|---|---|---|
| time, stage S1 | 89.2 s | 23.9 s |
| peak memory | about 4.3 GB | about 0.2 GB |
| score rows | 2,872,100 | 2,872,100, byte-identical |

Byte-identity checked with `pd.testing.assert_frame_equal` on the old and new score tables.

Yield filter, positives of the amide-coupling class. Command: `uv run python scripts/positives.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --sources orderly --benchmark-labels ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/labels.parquet`

| cut-off | before | after |
|---|---|---|
| none | 28,158 | 28,158 |
| ≥ 10 % | 27,822 | 12,636 |
| ≥ 20 % | 27,033 | 11,843 |

In ORDerly 20,662 of 38,459 reactions (53.7 %) have no yield at all.

**Interpretation of the yield result.** A 20 % cut-off looked like it cost 4 % of the positives. It costs 58 %. This is the most consequential finding of the pass, and it is not about speed. `configs/classes/amide_coupling.yml` still has `min_yield_pct: null`, with the note that these sizes are reported before choosing. So this was exactly the number that was going to make that decision, and it was wrong.

Stereo collapse of class-wide pairs (`family == "all"`, 28,158 pairs). A group is a set of rows that share a flat pair key but differ on the stereo pair key. Command: `uv run python scratch/count_stereo_collapse.py` (no argument)

| | groups | rows | share |
|---|---|---|---|
| groups, all | 352 | 732 | 2.6 % of rows |
| size 2 | 325 | | |
| size 3 | 26 | | |
| size 4 | 1 | | |
| safe to merge | 200 | | 57 % of 352 |
| not safe to merge | 152 | | 43 % of 352 |

Safe to merge, 200 groups:
- 104: one record drawn flat, another with a wedge; partner achiral.
- 96: single assigned centre on the varying side (a true enantiomer pair); partner achiral.

Not safe to merge, 152 groups:
- 43: single R/S on the varying side, partner carries an assigned centre (diastereomeric pair).
- 40: two or more assigned centres on the varying side, partner achiral (epimers within one molecule).
- 19: flat-vs-wedge, partner carries an assigned centre.
- 18: multi-centre R/S, partner carries an assigned centre.
- 14: single R/S, partner has only unassigned centres.
- 7: flat-vs-wedge, partner has only unassigned centres.
- 7: E/Z only (cis/trans is not an enantiomer relationship).
- 4: both sides vary.

200 + 152 = 352.

My judgement, not a measurement: 21 of the 152 (the 14 and the 7 whose partner has only unassigned centres) are counted as not safe by my reading. The counting agent called them ambiguous or probably safe, which would give 145 not safe. An unassigned centre does not mean the molecule is achiral. It means it is chiral and was drawn flat. The partner is therefore chiral, a diastereomeric transition state is possible, and the symmetry argument that makes the other 200 safe does not apply.

HATU family: 7 groups, 14 rows, all of size 2, all symmetry-safe. None has chirality on both sides. Merging would take the family from 1,177 pairs to 1,170, and the class width from 28,158 to 27,778.

Two of the five leaking U pairs are not stereo at all. They are isotopologues: ¹³C-benzoic acid and d₅-benzoic acid against plain benzoic acid. The inventory seems to hold labelled analytical standards that probably should not be on a discovery plate.

Adversarial validation, which does not reproduce. ADR 008 and 009 quote 0.984 for the fingerprint and 0.559 for the hand descriptors. The re-run today gives the numbers below. Command: `uv run python scripts/counts/count_adversarial.py ~/University/Phd/Dataset/Datasets` (the dataset root is its one argument; the script moved from `scratch/` to `scripts/counts/` under ADR 011 because its numbers go into the paper).

| | fingerprint | hand descriptors |
|---|---|---|
| ADR 008 / 009 | 0.984 | 0.559 |
| re-run today | 0.864 / 0.870 | 0.730 / 0.776 |

On amines the hand-written descriptors carry more provenance than the rooted fingerprint. That is the opposite of what I had been saying. I could not trace why. The feature table was rebuilt under ADR 009, the old one was deleted, and nothing was committed. ADR 012 records the correction and the new framing.

Feature blocks on two axes. Command: `uv run python scratch/count_block_two_axes.py` (seed 20261022). The dataset root is hard-coded at line 61 of the script, so this command is correct only on my machine; on the cluster the root must be changed in the script first.

- The floor is size, at about 0.67, not chance. Literature molecules are genuinely bigger.
- ECFP4 is dominated. `morgan_centre` finds analogues better (0.711 against 0.632) and carries less provenance on both roles.
- ECFP4 wins only the supervised upper bound. That bound is fitted on the validation labels. It is a diagnostic, never a result.
- No block sits in the useful corner. `morgan_centre` is the most defensible compromise, not a win.
- The reimplementation reproduces B2 at 0.632 and B7 at 0.598 exactly. That is what makes the rest trustworthy.

No command exists for: the citation map, the 0-of-6 Chemlex fixture join, and the count of failing tests before a fix. They are the same gap as in the entries of 2026-10-22.

**Interpretation.** Three times in two days a discrepancy was unresolvable because there was no version to compare against: the rooted-fingerprint environment count that fell from 314 to 308, the journal entries a sweep had rewritten, and now 0.984. Each time the cause was the same. That is why this is the last entry before the first commit.

**Next.**
- Commit, in topic commits.
- Open, deliberately not done: the one-line fix for U on flat keys (proposed, not yet agreed).
- Open, deliberately not done: deduplicating P on flat keys (counted, left on stereo keys with the reason recorded).
- Open, deliberately not done: ADR 012 item 4, whether to declare a new baseline B8 on `morgan_centre`. It must be accepted or dropped before stage S2 runs, never after. B2 is pre-registered, and swapping its fingerprint now would be the selective choice the pre-registration exists to prevent.

---

## 2026-10-22 — Cleanup of redundant files; one finding recorded before deletion

**Context.** I asked for the repository to be cleaned of redundant files. A citation map showed that 15 of the 19 files in `scratch/` are cited from an ADR, the design doc, a config comment or this journal as the command that regenerates a number. They stay. There is no command for the citation map: an agent grepped `CLAUDE.md`, `README.md`, `LAB_JOURNAL.md`, `docs/`, `configs/`, `src/`, `scripts/`, `tests/` and `.claude/` for each `scratch/` filename and its stem. This is the third figure today with no script behind it, after the 0-of-6 fixture check and the 44 failing tests in the entry below. It is the same gap as the open question further down, seen from the other side. Three uncited scripts and one stale plan file go. My rule for the orphans: record the finding first, then delete the file.

**Done.**
- Ran `scratch/handcheck2.py` before deleting it. It asks how much `n_ewg_on_ring` changes when electron-withdrawing groups are counted over the union of all aromatic rings on the reacting nitrogen, instead of one arbitrarily chosen ring. Population: open literature positives, single-component, not in the library, not in validation.
- Deleted from `scratch/`: `handcheck.py`, `handcheck2.py`, `export_explainer_data.py`, `claude_code_plan_steps_1-3.md`.
- Deleted `tests/data/uspto_4rows.parquet` (no test loads it) and `graphify-out/` (knowledge-graph export from 2026-10-09, git-ignored, regenerable). Measured before deleting with `du -sh graphify-out/`: 3.7 MB in 119 files. `du -sh scratch/` gives 252 KB, so the stale export was about fifteen times the size of all the exploration scripts together.
- Deleted three published claude.ai pages: a project guide, a stage-S plan and an explainer. All three carried numbers that today superseded, and one pre-empted a decision that ADR 008 item 6 now writes differently. The deletions are irreversible. Three other project-related published pages are left alone, because I did not ask for them to go.

**Result.** I ran `uv run python scratch/handcheck2.py` (no arguments, from the repository root). The script is deleted in this same piece of work, so no command in this entry can regenerate the output. For this one finding the three rows below are the record, not a pointer to a script.

| diaryl amine | class | stored `n_ewg_on_ring` | single-ring counts |
|---|---|---|---|
| `COc1cc(Nc2ccccc2)cc(OC)c1OC` | aniline | 0 | [0, 0] |
| `O=S1c2ccccc2Nc2cc(Cl)ccc21` | aniline | 0 | [0, 0] |
| `Cc1cn(C)c2c1Nc1ccccc1NC2=O` | aniline | 1 | [0, 1] |

Total: 3 diaryl amines. Only the third differs between the union and a single ring.

What the deleted files held:
- `handcheck.py` was an eyeball check of alpha-halo acids, alpha-quaternary acids and diaryl amines in the private feature table. The fixes it led to (the `alpha_quaternary` halogen fix, the `alpha_vinyl` narrowing) are already in the 2026-10-21 entry. Nothing is lost.
- `export_explainer_data.py` only fed an explainer page that no longer exists. Every number it exported is printed by one of the cited counting scripts.
- `claude_code_plan_steps_1-3.md` was a pasted plan from 2026-10-09, superseded by `docs/design/rebuild.md`.

**Interpretation.** The union is the right definition: the result must not depend on an arbitrary choice of ring. The honest scale is 3 amines out of thousands, with 1 affected. It is a correctness point, not a modelling point. I should not spend more time on it.

**Open question, not decided.** All 15 cited scripts live in `scratch/`, which is git-ignored. Every number in the ADRs and in this journal therefore cites a command that does not exist in a fresh clone. ADR 005 requires the published work to rebuild from open sources alone. The sharpest case is `scratch/build_chemlex_like_fixtures.py`: CLAUDE.md §9 says that for Chemlex we publish only the code that rebuilds the tables from the original file, and that code cannot be published from where it sits. I have not decided what to do about it.

**Next.** Nothing is committed. The architecture review of dead code inside `src/` is still running.

---

## 2026-10-22 — Pre-commit audit: data-guardian BLOCK, a Chemlex mislabel, ADR 009

**Context.** Nothing had been committed in two days. Before asking about a commit I ran data-guardian over the whole working tree, 88 paths (`git status --short | wc -l`).

**Done.**
- **data-guardian returned BLOCK.** Most of the tree was clean. The three `tests/data` fixtures are hand-written: it joined `chemlex_like_6rows.parquet` against the raw `Chemlex_Acidamine_Wetlab_Data.xlsx` and printed only booleans. Against Chemlex's 11,669 rows it found:

  | fixture items in Chemlex | found |
  |---|---|
  | acid/amine pairs | 0 of 6 |
  | products | 0 of 6 |
  | acids | 0 of 6 |
  | amines | 0 of 6 |

  There is no script for this check yet. The numbers stand on this entry, not on a command. That is a gap: if a licence argument rests on it, it should become a test.

  `.claude/` holds no secret. The Chemlex split leaks nothing. Four HIGH findings stopped it:
  1. **Two holes in the licence guard** (`src/platepu/io.py`). The `private/` test was lexical: `root/private/../library/x.parquet` contains the word `private`, but the bytes land in `root/library/`, and a `private` symlink can point anywhere. The licence test was a deny-list naming `licensed` and `internal`, so a null licence, `"Internal"`, `"internal "` and any unrecognised value passed. Fixed, tests first: `_under_private` now resolves both paths, and the check is an allow-list that asks every value to be `open`.
  2. **A test-side label read.** `scripts/benchmark.py` published `both_unseen_test_positive_share`, the mean of the test labels, in a counting report. §6 says never tune on TEST. I removed it; only the row count is reported.
  3. **An append-only violation.** A sweep of `LAB_JOURNAL.md` removed "— Luka" from the two 2026-10-02 headings and deleted the line "**Next.** Jan reviews ADR 001 in PR #2". It also rewrote the status line of the Accepted ADR 001. data-guardian caught it before any commit. I restored all of it from `git show HEAD:LAB_JOURNAL.md`.
  4. **`.gitignore` was too broad.** `!tests/data/**` would have let a stray CSV or image in `tests/data/` be committed. Narrowed to `!tests/data/*.parquet`.
- **What I found myself, which was worse.** `src/platepu/ingest/chemlex.py` hard-coded `licence="open"` for Chemlex, which is CC BY-NC-ND 4.0. The guard reads the label, not the chemistry, so a wrong label disarms it for every path. Also, every `*.rejected.parquet`, the RXNMapper shards and the benchmark labels carry no licence column, so the guard never sees them.
- **ADR 009** (`docs/decisions/009-chemlex-is-noncommercial.md`, Accepted, supersedes ADR 004 in part). Four decisions:
  - a third licence value `noncommercial` for Chemlex;
  - `LICENSED_SOURCES` renamed `PRIVATE_SOURCES`, because all four of its readers mean "withhold from the open outputs";
  - the benchmark labels and the validation and test tables gain a licence column, so the guard protects them by content, not by path;
  - the existing Chemlex mappings in the open RXNMapper cache stay, as a recorded and accepted exception. I do not purge them.
- **Executed the same day, tests first.** 44 tests failed before I wrote any of the code. That number cannot be re-measured now: it was `uv run pytest -q` at a state where the tests existed and the code did not, and the same command now gives 329 passed. The licence precedence lived in two places and one copy did not know about `internal`. It is now one function, `io.strictest_licence`, ordered `internal`, `licensed`, `noncommercial`, `open`, called by both `features` and `positives`. No script writes a licence value any more: `features/substrate.py` and `scripts/score_baselines.py` read it from their input instead of stamping `open`.
- **CLAUDE.md.** §10 gained `score/` (it was missing, which §4 forbids). §9 gained the `noncommercial` taxonomy and the allow-list. §5 gained the list of agreed dependencies: rxnutils, scikit-learn, openpyxl, pyyaml.
- **Re-ran every stage** (commands below).

**Result.** Every stage reproduced exactly, with the licence as the only changed cell. I compared every other column against the pre-change tables.

| stage | rows | licence before → after |
|---|---|---|
| ingest | 11,669 records, 0 rejected | open → noncommercial |
| rsmi | 11,662, 7 rejected | open → noncommercial |
| mapped | 11,662, 0 failed | open → noncommercial |
| clean | 11,662, 0 rejected | open → noncommercial |
| classify | 9,370 amide couplings, 2,292 rejected | open → noncommercial |
| standardise | 9,370 | open → noncommercial |

| labels and split | value |
|---|---|
| labelled | 8,735 |
| positive | 28.1 % |
| validation | 2,615 |
| test | 1,313 |

Unchanged.

| stage F | value |
|---|---|
| substrates | 22,050, unchanged |
| moved `open` → `noncommercial` | 188 |
| `internal` | 1,067, unchanged (the strictest licence wins) |

The 188 substrates are those seen only in validation. That 188 is the size of the mislabel.

Commands:

    uv run python scripts/ingest.py chemlex ~/University/Phd/Dataset/Datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx ~/University/Phd/Dataset/Datasets/derived
    uv run python scripts/rsmi.py chemlex ~/University/Phd/Dataset/Datasets/derived
    uv run python scripts/mapped.py chemlex --paths configs/paths/local.yml
    uv run python scripts/clean.py chemlex ~/University/Phd/Dataset/Datasets/derived_samples
    uv run python scripts/classify.py chemlex ~/University/Phd/Dataset/Datasets/derived_samples
    uv run python scripts/standardise.py chemlex ~/University/Phd/Dataset/Datasets/derived_samples
    uv run python scripts/benchmark.py chemlex ~/University/Phd/Dataset/Datasets/derived_samples ~/University/Phd/Dataset/Datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx
    uv run python scripts/benchmark_split.py chemlex amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived_samples
    uv run python scripts/features.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

CLAUDE.md length: `wc -l CLAUDE.md` prints 202. The file's own rule is under 200 lines.

**Stage S1 re-run.** It reproduced exactly.

| item | value |
|---|---|
| score rows | 2,872,100, same count as before |
| columns other than `licence` | identical, including every one of the 2,872,100 scores (`pd.testing.assert_frame_equal` on the common columns) |
| validation set licence | `open` → `noncommercial` |
| inventory licence | `internal`, unchanged |
| run time | 1 min 29 s |

    uv run python scripts/score_baselines.py amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

This was the expected outcome, so on its own it is a weak check. The baselines read no licence column. A score could only have moved if something else had changed by accident. The value of the check is that nothing did.

**Interpretation.** Two things worth keeping.
- The guard reads a label, so a mislabel at the source disarms it everywhere at once. That is a stronger argument for the allow-list than any single rule: an allow-list refuses what it does not recognise instead of waving it through.
- Moving a table under `private/` is not protection unless the table carries its own licence. A path convention is exactly what hole 1 defeated.

**Honest notes.**
- I wrote code before tests twice today: the `shared_substrates` guard and `load_baselines_config`. chem-verifier covered both afterwards, which is not the same as tests first.
- The append-only violation came from my own brief to a subagent. I let a preference about how to write from now on be applied backwards to historical records.

**Next.** Nothing is committed; I have not asked for a commit. Open:
- whether `derived/chemlex_labels/` (52 KB, `du -sh ~/University/Phd/Dataset/Datasets/derived/chemlex_labels`; an orphan no current script writes) is deleted;
- the two CLAUDE.md lines: it is 202 lines against "under 200", and I have to decide what gives;
- a test for the `scripts/positives.py` open/private split, now that `chemlex` is in `PRIVATE_SOURCES`;
- then stage S2, the PU models, with U built from literature substrates.

---

## 2026-10-22 — Stage F and S1 built; acid subclasses added and removed; a larger, leak-free validation set; three confounds measured

**Context.** Yesterday ended with ADR 007 (acid subclasses, Accepted 2026-10-21) written and stage S agreed on paper. Today I wanted stage F finished with a rooted fingerprint, the eight baselines B0 to B7 run, and an answer to one question I could not yet answer: how much can a validation set of 53 amines tell me?

**Done.**
- **Stage F** (substrate features) built and tested: 10 descriptors, a whole-molecule Morgan count fingerprint (`morgan`) and a second one rooted at the reacting atom (`morgan_centre`). I had assumed the rooted bits were a subset of the whole-molecule bits. The tests disproved it. RDKit drops an environment whose bond set is contained in another, so the radius-2 environment of a terminal amine nitrogen is absent from the whole-molecule fingerprint. The tests now pin the real behaviour. The stage F report now also summarises `morgan_centre`.
- **Stage S1**, the eight baselines B0 to B7, built, run and reported (table below).
- **Acid subclasses added, counted, then removed the same day.** I wrote SMARTS for nine acid subclasses and counted the evidence (`uv run python scratch/count_subclass_evidence.py $D`). The data does not support them as features. ADR 008 (`docs/decisions/008-hand-written-chemistry-belongs-in-the-opponent.md`) supersedes item 3 of ADR 007, the acid subclasses, and item 5, the success rule; I removed them from `configs/classes/amide_coupling.yml`. B4 to B6 are defined on the amine only. One SMARTS bug worth remembering: `[CX3:1](=O)[OX2H1,OX1-]-c` attaches the substituent to the oxygen, not the carbon. It needs branch parentheses: `[CX3:1](=O)([OX2H1,OX1-])`. I caught it by testing 25 named acids.
- **The validation set was redefined.** Chemlex's own train side shared substrates with its test side as stereoisomers (5 acids, 1 amine, same flat key). Our fingerprints are chirality-blind, so that is a leak. Validation is now every labelled row in the family whose acid and amine appear nowhere in the test side, compared on the flat InChIKey. data-guardian BLOCKed until this was fixed.
- **Three confounds measured** (below).
- **Four review findings fixed** after chem-verifier wrote failing tests first: B2 and B3 raise on an empty positives table; `pair_features` refuses pairs whose `std_version` disagrees with the feature table; the split report records `shared_acids` and `shared_amines`; the stage F report summarises `morgan_centre`.
- **Two claims of mine retracted or corrected.** I had written that B2 beats B3 by 0.012. The bootstrap interval covers zero, so the claim is withdrawn. I had also said the validation set was too small to resolve anything. The paired half-width turned out to be 0.049, so that was too pessimistic.

**Result.** Validation set (`uv run python scripts/benchmark_split.py chemlex amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived_samples`):

| | old | new |
|---|---|---|
| labelled rows read in | | 8,883 |
| validation reactions | 1,725 | 2,615 |
| validation positive rate | 32.8 % | 29.14 % |
| test reactions | 1,313 | 1,313 |
| distinct amines | 53 | 113 stereo keys, 112 flat |
| distinct acids | | 95 stereo keys, 94 flat |
| shared acids / amines with test, flat keys | 5 / 1 (Chemlex train side) | 0 / 0 |
| old validation rows now excluded | | 150 |

The old set is not a subset of the new one. One acid pair and one amine pair collapse on the flat key; the flat counts are the ones that matter statistically.

S1 baselines on the new validation set (seed 20261021, `std_version` 2; P has 1,177 HATU pairs and 28,158 class pairs; 2,615 validation pairs and 284,595 inventory pairs scored, 2,872,100 rows out in 1 min 29 s (89.2 s wall, `/usr/bin/time -p`); `uv run python scripts/score_baselines.py amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived --validation ~/University/Phd/Dataset/Datasets/derived_samples/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet`):

| model | AUC | distinct scores | largest tie |
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

The two B3 columns agree to three decimals although the score vectors differ (Spearman 0.414 between them; means 0.818 at HATU width and 1.213 at class width; `uv run python scratch/count_b2_vs_b3.py`, section `width_coincidence`). A coincidence, not a bug. Re-running the whole of S1 reproduced the previous table byte for byte (`pd.testing.assert_frame_equal` passed on the before and after parquet), which checks the seed.

Paired differences, Owen's pigeonhole bootstrap, 2,000 resamples, seed 20261022, whole acids and amines resampled by flat key (`uv run python scratch/count_b2_vs_b3.py`):

| difference | estimate | 95 % interval | ahead in |
|---|---|---|---|
| B2 class minus B3 class | +0.010 | [-0.070, +0.084] | 57.6 % |
| B2 HATU minus B3 HATU | -0.031 | [-0.109, +0.044] | 18.6 % |
| B6 minus B4 | +0.080 | [+0.037, +0.134] | 100 % |
| B6 minus B1 | +0.298 | [+0.204, +0.383] | 100.0 % |

Resampling amines alone gives a half-width of 0.083 for B6 minus B2; resampling acids alone gives 0.029 (same command).

The B5 and B6 figures differ from ADR 007 (0.775, 0.793). Those came from a two-descriptor steric variant; the implemented B5 uses `-b_n_heavy_at_2_bonds`. The two-descriptor variant of B5 gives 0.794 on the new set (`uv run python scratch/count_b2_vs_b3.py`).

Confound 1, provenance. Adversarial validation, 5-fold cross-validated ROC-AUC, literature substrates against lab substrates (`uv run python scratch/count_adversarial.py`):

| features | AUC |
|---|---|
| Morgan fingerprint | 0.984 |
| hand-written descriptors | 0.559 |

Confound 2, domain shift. Mean heavy atoms per pair: validation 38.66 (sd 11.83, median 37), inventory 23.73 (sd 7.12, median 22) (`uv run python scratch/count_stage_s_inputs.py`, section 4). The 41.4 I first carried was the old 1,725-row set.

Confound 3, statistical power: 53 amines became 113. A paired comparison resolves about 0.05 AUC (B6 minus B4, half-width 0.049, `uv run python scratch/count_b2_vs_b3.py`).

Gate: ruff, ruff format, mypy and pytest green, 263 passed, 1 skipped (`uv run ruff check . && uv run ruff format . && uv run mypy src && uv run pytest`).

**Interpretation.**
- B0 at 0.504 is the floor, as it should be.
- B2 against B3 is indistinguishable from zero. I cannot say whether knowing which acid went with which amine helps. Narrowing P to the HATU family hurts B2.
- The amine baselines beat popularity by +0.298, and B6 beats B4 by +0.080 with an interval that excludes zero. Chemistry on the amine does carry signal.
- A model that separates literature from lab substrates at 0.984 on fingerprints but 0.559 on descriptors can learn "looks like a patent molecule" instead of "reacts". Add the size gap, 38.66 against 23.73 heavy atoms, and fingerprints carry size too.
- The principle I settled on: hand-written chemistry may live in a baseline and in the reporting, never in the model's features. ADR 008 carries the reasoning.
- A PU model has to beat B6 (0.822) by about 0.05 before I call it better. Nothing is committed yet.

**Next.** ADR 008. Then stage S2, the PU models, with U built from literature substrates.

---

## 2026-10-21 — Acid subclasses, an electronic descriptor, and a success rule measured within a class

**Context.** Stage F was finished this morning (entry below) with four amine subclasses and nine reacting-centre descriptors. Before building stage S I wanted to know three things. Is the taxonomy chemically right? Are these the right features? How should a categorical class enter a PU model and its evaluation? Three Claude research agents searched the literature. Luka supplied two PDFs (Due-Hansen 2016, Zhang 2025) and two that were not what we were after.

**Done.**
- Three literature notes: `2026-10-21-subclasses-and-features-research.md`, `due-hansen-2016-difficult-amide-couplings.md`, `zhang-2025-amide-intermediate-knowledge.md`. They are agent summaries, not yet checked by Luka.
- Counted first, agreed the rule with Luka, then tests, then code (CLAUDE.md section 8).
- Nine acid subclasses in `configs/classes/amide_coupling.yml`. A tenth descriptor `n_ewg_on_ring` with its SMARTS in `configs/features.yml`. The subclass machinery was already generic, so the acid classes needed no new code.
- ADR 007 (Proposed). It supersedes ADR 006 in part, items 2 and 3. ADR 006 is untouched apart from its "Superseded by" line.
- A code review found three chemistry errors. All three are fixed before the run that counts.

**Result.** Acid classes over 11,787 acids (`uv run python scratch/count_subclass_evidence.py $D`):

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

Every named class holds at least 21 lab compounds. The counting that preceded this decision came from `uv run python scratch/count_subclass_candidates.py`; that script predates the heteroaromatic split, so its class names are not these.

Amine classes, same command (`uv run python scratch/count_subclass_evidence.py $D`):

| class | amines | of them in the lab library |
|---|---|---|
| aliphatic_primary | 3,934 | 184 |
| aliphatic_secondary | 2,926 | 72 |
| aniline | 2,592 | 232 |
| heteroaryl_amine | 699 | 41 |
| other | 53 | 9 |

Validation success by acid class (`uv run python scratch/count_subclass_evidence.py $D`):

| class | success | support (reactions / acids) |
|---|---|---|
| alpha_quaternary | 0.136 | 110 / 5 |
| alpha_heteroatom | 0.230 | 653 / 28 |
| alpha_vinyl | 0.276 | 58 / 2 |
| alkyl | 0.362 | 105 / 4 |
| heteroaryl_azine | 0.378 | 111 / 4 |
| aryl | 0.401 | 367 / 14 |
| heteroaryl_pi_donor | 0.470 | 202 / 8 |
| arylacetic | 0.517 | 1 acid |
| alpha_tertiary | 0.550 | 2 acids |

Zero-training ROC-AUC on the Chemlex HATU validation set (`uv run python scratch/count_subclass_evidence.py $D`):

| baseline | AUC |
|---|---|
| popularity | 0.54 |
| amine subclass | 0.713 |
| amine `n_heavy_at_2_bonds` alone | 0.711 |
| B5: subclass + topological sterics | 0.775 |
| B6: B5 + `n_ewg_on_ring` | 0.793 |

B5 and B6 were put together after I had looked at validation. They are optimistic. I write them down only because they raise the bar.

`n_ewg_on_ring` is non-zero for 1,761 of 21,991 substrates (`uv run python scratch/count_subclass_evidence.py $D`). Tests: 247 passed. The stage F re-run gives an identical table, 21,991 substrates in about 5 s.

The three errors the code review caught, each now pinned by a test in `tests/test_classify.py` or `tests/test_features.py` (no script produced them):
- The electronic descriptor searched only one aromatic neighbour. A diarylamine's value depended on atom order in the SMILES.
- `heteroaryl_alpha` lumped 869 pi-donor acids (2-furoic, 2-thenoic, indole-2-carboxylic) with 1,144 azine acids, whose ring nitrogen pulls the other way.
- `alpha_quaternary` counted halogens towards the degree. Trifluoroacetic acid was filed as sterically hindered.

I also narrowed `alpha_vinyl` to a C=C after the grid image (`uv run python scratch/grid_features.py`) showed an alpha-keto acid in it.

**Interpretation.**
- The acid side was the real gap. Haas (PNAS 2022) finds the acid largely sets the rate. The sharpest failures in Due-Hansen are acid-side, including arylacetic acids that fail through a suspected ketene route no descriptor can express.
- Cheap topological descriptors already match a 3D buried-volume descriptor on validation (0.711 against 0.733, the latter from `uv run python scratch/count_design_rule_proxy.py`). Sterics needs no conformers and no new dependency.
- The amine-class baseline scores exactly 0.5 inside a class. ADR 006's third condition therefore only asked the model to beat chance there. This is the main reason for ADR 007.
- Splitting the heteroaromatic acids showed the old single class was an average of two families, 0.378 against 0.470. A class that mixes opposite electronics is worse than no class. The same reason moved alpha-keto acids out of `alpha_vinyl` and halogenated ones out of `alpha_quaternary`.
- `n_ewg_on_ring` is the one descriptor validation cannot check. No validation aniline carries a nitro, nitrile, trifluoromethyl or sulfonyl group. The whole observed effect is three anilines with a ring carbonyl, 2.6 % success against 21.7 %. It stays in the table because the chemistry is settled and about 10 % of the lab amines carry such a group.
- We stay with Chemlex as the validation benchmark. Its label means "fast enough in 30 minutes at 30 °C", not "feasible". Due-Hansen report 17 % after four days at room temperature with HATU and DIPEA for a hindered acid with an aminothiazole.
- Not adopted: Zhang's replacement of the acid by its activated ester before featurisation. It helped a transformer but hurt a tree model on fingerprints. On our 69 validation acids it adds 27 bits that every product shares and raises the mean pairwise Tanimoto between acids from 0.145 to 0.384 (`scratch/count_subclass_evidence.py`).

**Next.** Stage S: baselines B0 to B6 from this table, then the PU models. Luka still has to accept ADR 007 item 4, the within-class success rule. One open point: the `.gitignore` rule `literature/**/*.pdf` is anchored at the repository root and does not cover `docs/literature/`. A paper PDF dropped there could be committed.

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scratch/count_subclass_evidence.py $D
uv run python scratch/count_subclass_candidates.py
uv run python scratch/count_electronic_descriptor.py
uv run python scratch/count_design_rule_proxy.py
uv run python scripts/features.py amide_coupling $D/derived --validation $D/derived_samples/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet
uv run python scratch/grid_features.py
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

---

## 2026-10-21 — Stage F: substrate features; std_version carried; amine subclasses anchored at the reacting N

**Context.** Stage F of ADR 006 (item 2) had one line in `docs/design/modeling.md`. I counted first, then agreed the rule with Luka (section "Stage F — features (agreed 2026-10-21)").

**Done.** Stage F (`src/platepu/features/substrate.py`, `scripts/features.py`). Before it, I fixed upstream: `std_version` was missing in positives, library and benchmark measurements. It is now carried; mixed values stop `build_positives` and stage F. I re-ran benchmark, split, positives, library and unlabeled. Amine subclasses moved into `configs/classes/amide_coupling.yml`, anchored at the reacting N. The 29 tautomer keys are handled in stage F, not in stage 6 (see Result).

**Result.** Counts before coding (`uv run python scratch/count_f_features.py $D`):

| | |
|---|---|
| distinct substrates by `key_stereo` (P ∪ library ∪ Chemlex validation) | 21,991 |
| acids / amines | 11,787 / 10,204 |
| substrates with 0 or 2+ reacting atoms | 0 |
| Morgan count maximum (fits uint8) | 51 |
| bits set on average / bits never set | 35.2 / 0 |

Half of the acid candidate descriptors were constant (degree, H count, ring of the carboxyl C). I dropped them.

Domain shift, same script:

| | P | library | validation |
|---|---|---|---|
| acid heavy atoms | 20.6 | 13.2 | 21.7 |
| amine heavy atoms | 17.3 | 10.6 | 19.8 |
| amine alpha sum of degrees | 3.3 | 3.2 | 4.6 |

The library substrates are smaller. The validation amines are more crowded.

Re-runs (commands below): every old column identical, only `std_version` added ("2"). Library and U also gained `licence`; the files on disk predated the ADR 004 change of 2026-10-20. Row counts unchanged:

| | rows |
|---|---|
| labels | 8,883 |
| validation | 1,725 |
| positives (all / HATU) | 28,158 / 1,177 (29,335 in total) |
| library | 529 + 538 |
| U | 284,595 |

Tautomers (`uv run python scratch/count_tautomers.py`, `scratch/count_tautomers_keepstereo.py`, 28,577 substrates): 29 `key_stereo` in positives carry two SMILES, because the InChIKey merges tautomers (2-hydroxypyridine/2-pyridone, 1H/2H-pyrazole). I first chose canonical tautomers in stage 6. RDKit `Canonicalize` with defaults removed stereo in about 3,200 substrates, drew wrong structures (indoline-2-carboxylate became an indole hemi-orthoester; odd DHPs, oxindoles, hydantoins) and took the amine role from 23 positive amines (27 fail the single-group rule). All of that to fix 29 keys; the overlap with validation did not change. Dead end, dropped. Instead stage F keeps the smallest SMILES per key and lists the keys in the report. Full run: 33 keys (29 within positives, 4 drawn differently in library vs positives). Stereo is not used by the model (Morgan without chirality); it stays in `key_stereo` for identity.

Subclasses (heteroaryl_amine, aniline, aliphatic_secondary, aliphatic_primary, else other):

| | n |
|---|---|
| positive amines changed by anchoring at the reacting N (of 9,973) | 821 |
| library / validation amines changed | 3 / 2 |
| amine-class AUC on validation, before / after anchoring | 0.707 / 0.713 |
| same AUC from the stage-F `subclass` column | 0.713 |
| P amines: aliphatic primary / secondary / aniline / heteroaryl / other | 3,873 / 2,892 / 2,474 / 678 / 42 |

"Other" in validation is N on P, Si or N+ (5 of 53), probably Chemlex designed negatives. Side observation for `classify`: `!$(N-[N,O])` excludes only aliphatic N, so an N-aminopyridinium passes the amine role. I did not change it.

Stage F full run: 21,991 rows in 4 s. Overlaps of keys:

| | P∩L | P∩V | L∩V |
|---|---|---|---|
| acid | 340 | 14 | 4 |
| amine | 336 | 4 | 6 |

A second run gives an identical table. I checked a grid image of 20 open positives-only substrates (R and alpha atoms sit on the right atoms) and 10 rows by hand. Tests: 194 passed, 1 skipped (RXNMapper environment). Review found no blockers; I added a guard so stage F refuses any file not named `validation.parquet`.

**Interpretation.** The table is small and fast, so stage S can read it directly. The library lies outside the P and validation range in size, so scores on the library are an extrapolation in size; how much that matters is for M2 (adversarial validation). Anchoring at the reacting N is the more honest definition of the subclass and helps a little (0.707 to 0.713). Canonical tautomers would fix 33 keys but damage more structures than that, so the smallest SMILES per key is enough for now.

**Next.** Stage S: baselines B0 to B4 from this table. Then stage E.

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scratch/count_f_features.py $D
uv run python scratch/count_tautomers.py
uv run python scripts/benchmark.py chemlex $D/derived_samples $D/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx
uv run python scripts/benchmark_split.py chemlex amide_coupling HATU_DIPEA_DMF $D/derived_samples
uv run python scripts/positives.py amide_coupling $D/derived --sources orderly --benchmark-labels $D/derived_samples/benchmarks/chemlex/labels.parquet
uv run python scripts/library.py amide_coupling "$D/Export Inventory Aktuell - Group.xlsx" $D/derived
uv run python scripts/unlabeled.py amide_coupling HATU_DIPEA_DMF $D/derived --validation $D/derived_samples/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet
uv run python scripts/features.py amide_coupling $D/derived --validation $D/derived_samples/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

---

## 2026-10-20 — First model prototype: ADR 006, scikit-learn, source document in stage 1

**Context.** The literature review for the first model (note `docs/literature/notes/2026-10-20-first-model-findings.md`) led to ADR 006. I wanted splits of P by patent.

**Done.** ADR 006 (proposed): P at two widths, reacting-centre descriptors, success rule fixed before the first run. scikit-learn added. Stage 1 now keeps `source_document` (ORDerly's `extracted_from_file`; `None` for Chemlex), with a test on the fixture.

**Result.** ORDerly has no patent field. Counted on the full file (625,697 reactions):

| | |
|---|---|
| `extracted_from_file` values | 489 |
| reactions per value, median / max | 707 / 6,462 |
| grant years per value | 1 |
| HATU positives: files holding them / share in the 5 largest | 123 / 29 % |

Stage 1 re-run for full ORDerly, full Chemlex and the 2,000-row sample: all old columns identical, new column filled (ORDerly) or empty (Chemlex). 147 tests pass.

**Interpretation.** The key is a time block of the patent corpus, not a patent. A split by it is a coarse time split. I chose it over no document split (Luka, 2026-10-20) to test whether a model holds across time; it does not separate analogues from one patent.

**Next.** Baselines B0 to B4 and the evaluation harness (`docs/design/modeling.md`, "Stages of the prototype").

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scripts/ingest.py orderly $D/orderly_condition_train.parquet $D/derived
uv run python scripts/ingest.py chemlex $D/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx $D/derived
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

---

## 2026-10-20 — Open points from the review: no Chemlex in tests, internal licence, loud stage 6

**Context.** The review earlier today left six open points. I agreed to the internal licence (ADR 004)
and to keeping the prototype history private (ADR 005); the other four change no result.

**Done.**
- **Fixtures without Chemlex.** The 6 rows cut from the Chemlex file and the 6 Chemlex reactions in
  the mapped fixture are replaced by hand-written rows in Chemlex's layout with the same cases (two
  HATU drawings, EDCI, PyBrOP with PF6 as a cation, a disagreeing replicate pair). Their reactions
  are mapped for real with RXNMapper; the mis-mapped case is written by hand (N on the ketone C,
  4 dynamic bonds, `no_class`, as for the real row). Builder: `scratch/build_chemlex_like_fixtures.py`.
- **Stage 6 and benchmarks stop on a failed substrate** (`standardise.required`) instead of writing a
  null molecule. Counted first: 0 failures in full ORDerly stage 5 (38,459 reactions) and in full
  Chemlex (23,338 substrates), so there is no rejected table to build.
- **Licence `internal`** on the library, its rejected containers and U; `io.write_table` refuses
  `licensed` and `internal` rows outside `private/` (ADR 004).
- **RDKit's log** is no longer switched off at import. Data molecules are read inside `BlockLogs`.
  The switch also came from rdchiral, imported by rxnutils.chem, so substrates are now neutralised
  with RDKit's Uncharger directly, with the same steps as rxnutils' `neutralize_molecules`.
- **Condition text** has its inverse, `condition_from_text`, which the benchmark split uses.
- ADR 005: no history rewrite; the paper's code is published from a fresh repository. The 35 data
  files on `origin/main` come from ORD, HiTEA 2024, Merck 2025, Roche 2023, AZ ELN 2023, Dreher 2016
  and JNJ HTE 2024; none from Pistachio, Chemlex or the inventory.
- CLAUDE.md §9, `rebuild.md` (stage 6, principles, step 10), `modeling.md`, the guardian and
  verifier agents updated. 12 new tests (146 pass; ruff, format and mypy clean).

**Result.** Stages 5, 6, 7b, the split, 7a, library and U re-run against the baseline of this morning:
all 14 tables and 9 reports identical, except the new `licence` column in the three inventory
tables.

| | baseline | now |
|---|---|---|
| stage 5, full ORDerly | 168 s | 91 s |
| stage 6, full ORDerly | 46 s | 40 s |
| library: acids / amines / rejected containers | 529 / 538 / 8,723 | same |
| U / P HATU_DIPEA_DMF | 284,595 / 1,177 | same |

**Interpretation.** Nothing in the repository now holds Chemlex rows, and the inventory is guarded by
the same check as Pistachio. Still guarded by path only: the RXNMapper cache (ADR 004,
consequences), until Pistachio is mapped.

**Next.** Commit by topic when I decide to; then M0 baselines on the Chemlex validation set.

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scratch/build_chemlex_like_fixtures.py ~/University/Phd/envs/rxnmapper/bin/python
R=$(mktemp -d); mkdir -p $R/orderly; ln -s $D/derived/orderly/4_clean.parquet $R/orderly/
uv run python scripts/classify.py orderly $R && uv run python scripts/standardise.py orderly $R
uv run python scripts/library.py amide_coupling "$D/Export Inventory Aktuell - Group.xlsx" $R
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

---

## 2026-10-20 — Code review: one parse per reaction, one function per rule, docs match the code

**Context.** Before the model step (M0/M1, ADR 003) I reviewed `src/` and `scripts/` against CLAUDE.md
§5–§7 and fixed what changes no result. Anything that would change a result is listed as open below.

**Done.**
- Stage 5 parses each reaction once. It used to parse every rejected reaction a second time to find
  the reason, and 94 % of ORDerly is rejected there. An unreadable molecule is now a rejected row
  (`unparsable`) instead of a crash without a record id.
- One function per rule: `classify/groups.group_counts` (positives and the lab library),
  `standardise.in_family` (positives and the benchmark split), `standardise.condition_text`
  (benchmark labels and the stage-6 report), one `REJECTED_SCHEMA` in `io.py` instead of four.
- `standardize()` returns `no_inchikey` for a molecule RDKit writes no InChIKey for (a dummy atom
  `*`), instead of a result with an empty key that never matches.
- `scripts/classify.py` stops when it finds no class files (run outside the repository it rejected
  every row as `no_class` and exited 0); its report adds `reactions_matched`. The positives writer
  takes five parameters.
- 14 new tests, named after the cases (134 pass, ruff, format and mypy clean).
- Docs: `rebuild.md` (stage-5 reasons, largest fragment in stage 6, mapping function names, step 8
  status), `modeling.md` (new section "Model inputs" for library, U and validation/test, which were
  built before they were described), CLAUDE.md §5 import order, §6 `.yml`, §10 subpackages.

**Result.** Stages 5, 6, 7b, the split, 7a, library and U re-run with the old and the new code into two
fresh roots (inputs linked: full ORDerly stage 4, full Chemlex stages 1 and 4). All 14 tables and 9
reports are identical cell by cell; the classify report only gains `reactions_matched`.

| | before | after |
|---|---|---|
| stage 5, full ORDerly (625,630 reactions) | 168 s | 89 s |
| stage-5 rows / reactions matched | 38,459 / — | 38,459 / 38,454 |
| P all conditions / HATU_DIPEA_DMF | 28,158 / 1,177 | same |
| U | 284,595 | same |
| Chemlex labelled / positive; validation / test | 8,735 / 28.1 %; 1,725 / 1,313 | same |

**Interpretation.** The numbers of 2026-10-09 stand; the code now says the same thing as the docs.
Open, each a decision rather than a refactor:
- stage 6 writes no rejected table: a substrate that fails standardisation stays as a null row
  (count on the samples first, then decide whether it leaves the stage);
- inventory-derived tables have no `licence` column, so `write_table`'s guard does not see them; they
  are private by path only (ADR if the guard changes);
- `rdBase.DisableLog("rdApp.*")` runs on import in three modules and also hides RDKit's reason for an
  unparsable row;
- benchmark labels store the condition as `"A + B"` text and the split parses it back (a list column
  would change the labels schema);
- `origin/main` still tracks 24 prototype data files (e.g. `final_datasets/test.parquet`); their
  licence is unchecked.

**Next.** M0 baselines (random, popularity, nearest neighbour, additive, chemist rule) on the Chemlex
validation set, then M1 (P vs U, PAYN, bagging PU).

```bash
D=~/University/Phd/Dataset/Datasets
R=$(mktemp -d); mkdir -p $R/orderly; ln -s $D/derived/orderly/4_clean.parquet $R/orderly/
time uv run python scripts/classify.py orderly $R     # 89 s; compare $R/orderly/5_classified.parquet
                                                    # with $D/derived/orderly/5_classified.parquet
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

---

## 2026-10-09 — Full ORDerly through the pipeline; P, U and how far they reach

**Context.** The first model needs P (ORDerly, HATU_DIPEA_DMF), U (lab pairs) and an idea of how much of
the lab and Chemlex substrate space P actually covers.

**Done.** Full ORDerly through stages 3–6 on the Mac (RXNMapper on CPU ~2.5 min per 10 k reactions,
63 shards, ~4 h; stage 4 ~31 min on one core). Positives (`scripts/positives.py`), then
`library/unlabeled.py` + `scripts/unlabeled.py`: U = lab acids × lab amines minus P, plus an overlap
report on stereo-agnostic keys. ADR 003 and `docs/design/modeling.md` written from the literature
(notes in `docs/literature/notes/`).

**Result.**

| | |
|---|---|
| mapped / mapping failed | 625,649 / 15 |
| amide couplings (true amines) | 38,459; 33,692 at 2 dynamic bonds |
| rejected for positives | dynamic bonds 4,767; several groups 5,059 |
| P pairs, all conditions / HATU_DIPEA_DMF | 28,158 / **1,177** (≥ 10 %: 1,145; ≥ 20 %: 1,099) |
| U (lab pairs not in P) | **284,595** (7 lab pairs are in P) |

Overlap (share of unique substrates found in P):

| | P HATU_DIPEA_DMF | P all conditions |
|---|---|---|
| lab acids (517) | 12 % | 71 % |
| lab amines (520) | 15 % | 65 % |
| validation acids (69) | 6 % | 25 % |
| validation amines (53) | 2 % | 8 % |
| validation pairs both / one / none in P | 2 / 125 / 1,598 | 28 / 514 / 1,183 |

Lab and Chemlex validation share almost no substrates (~1 %).

**Interpretation.** The narrow P covers few of the substrates we will score: 93 % of the validation pairs
have neither substrate in it, so the model has to generalise through structure, not recall. The same
class under any condition covers most lab substrates (65–71 %), which makes "P of the family vs P of the
whole class" the first narrowness comparison on validation. The yield cut-off removes at most 7 % of P.

**Next.** Stage M0 (baselines) and M1 (PU models) on Chemlex validation, per ADR 003.

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scripts/positives.py amide_coupling $D/derived --sources orderly --benchmark-labels $D/derived_samples/benchmarks/chemlex/labels.parquet
uv run python scripts/unlabeled.py amide_coupling HATU_DIPEA_DMF $D/derived --validation $D/derived_samples/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet
```

---

## 2026-10-09 — Narrow first: HATU/DIPEA/DMF to the end; lab library; Chemlex validation and test

**Context.** With Jan I decided to stay narrow: amide coupling (acid + true amine) under HATU + DIPEA +
DMF (extras allowed) taken to a validated model first, with the unlabeled space U built from the
substrates we have in the lab. P comes from ORDerly only in this round; the modeling approach is
researched before any model code.

**Done.**
- `standardize()`: substrates are reduced to their largest organic fragment (the inventory writes salts
  and `[H+]`; reactions never have multi-fragment SMILES, so the pipeline is unchanged), and a
  neutralisation that yields an unreadable SMILES now returns `neutralise_failed` instead of crashing
  (cyclopentadienides). `std_version` 2.
- `library/inventory.py`, `scripts/library.py`: the group's inventory export
  (`Datasets/Export Inventory Aktuell - Group.xlsx`, internal, outputs under `derived/private/`) as a
  substrate library: one row per compound with its role in the class.
- `benchmarks/split.py`, `scripts/benchmark_split.py`: Chemlex both-unseen `train` side = validation,
  `test` side = test (written to its own file, read only in the final evaluation).

**Result.** Samples unchanged after the standardisation change (ORDerly 141, Chemlex 9,370; labels
8,735 / 28.1 %). Inventory: 10,212 containers, 9,635 with a SMILES → **529 acids** (804 containers)
and **538 amines** (685 containers); rejected containers: no role 7,204, unparsable 621, no SMILES 577,
both roles 150, several groups 149, neutralisation failed 22. Chemlex HATU_DIPEA_DMF: validation 1,725
reactions (32.8 % positive), test 1,313; 3,139 labelled HATU reactions carry no both-unseen label and
are not used.

**Interpretation.** The lab offers ~285 k acid × amine pairs for this class. Whether the model can say
anything about them depends on how many lab substrates (and Chemlex substrates) are near the literature
positives: next entry, once full ORDerly is through the pipeline.

**Next.** Positives from full ORDerly; U = lab pairs minus P; overlap report; modeling research
(`docs/design/modeling.md`, ADR 003).

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scripts/library.py amide_coupling "$D/Export Inventory Aktuell - Group.xlsx" $D/derived
uv run python scripts/benchmark_split.py chemlex amide_coupling HATU_DIPEA_DMF $D/derived_samples
```

---

## 2026-10-09 — Stage 7b: Chemlex labels with roles from Chemlex's own columns

**Context.** The test set must not depend on the mapper, and must use the same class definition and
the same standardisation as the positives.

**Done.** `benchmarks/measurements.py` (roles from `reactants[0]`/`[1]`, class checked with
`classify.fits_roles`, molecules through `standardize()`, a reaction = substrate keys + condition key),
the label rule moved to `configs/benchmarks/chemlex.yml`, `scripts/benchmark.py`. The mapping is only
checked (`n_mapped_in_class`). Deleted `scripts/chemlex_labels.py`, `configs/test_labels.yml` and
`evaluate/`.

**Result.** Old check (HATU, all N-nucleophiles): 8,454 rows and 28.3 % positive, as before; 7,898
reactions instead of 7,923, because two amines Chemlex writes as two SMILES of one molecule now merge
(a sulfonate and its acid; acarbose). Class amide_coupling on the full Chemlex:

| | reactions | labelled | positive |
|---|---|---|---|
| all conditions | 8,883 | 8,735 | 28.1 % |
| HATU + DIPEA + DMF | | 6,177 | 30.9 % |
| PyBOP + DIPEA + DMF | | 816 | 43.8 % |
| TCFH + NMI + DMF | | 795 | 11.5 % |
| both-unseen test, all conditions | | 1,942 | 39.2 % |

Statuses: single 8,398, all 0 % 191, all ≥ 10 % 146, inconsistent 148. Mapping check: 13 reactions not
confirmed by the mapping (6 mis-mapped, 7 PyBrOP rows never mapped), all kept.

**Interpretation.** These replace the 7,761 / 28.3 % HATU labels of the earlier entry (which mixed in
sulfonamides, amidines, thioamides, hydrazines and hydroxylamines). Chemlex's `one_unseen` split has a
small leak: acarbose drawn two ways counts as two amines, one in train and one in test (4 reactions);
those get no `one_unseen` label here. Results stay on the both-unseen split, which is not affected.

**Next.** Stage 7a, positives: condition families, filters (dynamic bonds, single reactive group,
yield cut-off), cross-source dedup, leakage flags.

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scripts/benchmark.py chemlex $D/derived_samples $D/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx
```

---

## 2026-10-09 — Stage 6: standardise; exploration scripts retired

**Context.** Molecules need one form (substrates neutral, HATU drawings unified) and conditions one
key, the same for every source.

**Done.** `standardise/molecule.py` (`standardize`, `condition_key`; substrates neutralised with
rxnutils' `neutralize_molecules`, agents with synonyms from `configs/standardise.yml`),
`standardise/table.py` (condition key from the reagent side of the mapped reaction),
`scripts/standardise.py`. Check against the old result: the crude amide rule on full ORDerly with the
new `condition_key`. Then I deleted `explore_condition_bins.py`, `recount_condition_bins.py`,
`run_rxnutils.py`, their tests and `configs/rxnutils_std.yml` (last at commit `7161120`; the
2026-10-02 entries' commands run there).

**Result.** HATU + DIPEA | DMF = 1,376, then 547 (HBTU/TBTU + DIPEA | DMF), 387, 326, 301: identical to
2026-10-02. Samples: ORDerly 141 couplings, 62 with an empty condition key (51 report no agent or
solvent, 11 only HCl); Chemlex 9,370, top condition HATU + DIPEA + DMF with 6,765 rows (both drawings
now one key).

**Interpretation.** The new condition key reproduces the old bins exactly, so the rebuild changes the
roles (from the mapping) but not how conditions are compared. Couplings without reported conditions
can only enter class-level positives, not a condition family.

**Next.** Stage 7b: Chemlex labels from the standardised tables, roles from Chemlex's own columns.

```bash
R=~/University/Phd/Dataset/Datasets/derived_samples
for s in orderly chemlex; do uv run python scripts/standardise.py $s $R; done
```

---

## 2026-10-09 — Stage 5: generic class engine, amide coupling = acid + true amine

**Context.** Classes are YAML files (ADR 002); the engine must not know about acids or amines.

**Done.** `classify/engine.py`: a formed bond is a product bond between atoms of two different
reactants; a class matches when the bond joins role A's reacting atom (`:1` in its SMARTS) to role
B's. `configs/classes/amide_coupling.yml`, `scripts/classify.py`. Before fixing the amine SMARTS I
counted what Chemlex calls amines.

**Result.**

| | ORDerly sample | Chemlex |
|---|---|---|
| amide couplings (true amines) | 141 | 9,370 |
| … at 2 dynamic bonds | 121 | 9,356 |
| not this class: sulfonamides, amidines/guanidines, thioamides | — | 1,697 (33 compounds, 11 % ≥ 20 %) |
| not this class: hydrazines, hydroxylamines | 5 | 589 |

The old crude rule (137 couplings in the sample) counted 6 reactions that are not couplings (reductive
methylation with acetic acid as solvent, alkylation, Boc protection, oxadiazole and imide formation)
and missed acid + NH4+ → primary amide.

**Interpretation.** Chemlex's "Amine" column mixes N-nucleophiles. I decided the amine role is a true
amine only (no N on C=O/C=S/C=N/SO2/C#N, no N–N or N–O), for literature positives and benchmark alike;
the others are counted and could become their own classes. Benchmarks take their roles from their own
columns, so mapping errors (6 Chemlex rows) never remove test reactions; Chemlex labels must be
recomputed on 9,370 reactions (the 7,761 HATU labels of the earlier entry included the other
N-nucleophiles).

**Next.** Stage 6, standardise, in the pipeline; reproduce HATU + DIPEA | DMF = 1,376 on full ORDerly.

```bash
R=~/University/Phd/Dataset/Datasets/derived_samples
for s in orderly chemlex; do uv run python scripts/classify.py $s $R; done
```

---

## 2026-10-09 — Stages 2–4 on samples: reaction SMILES, RXNMapper cache, rxnutils clean

**Context.** Stage 1 was done; the next stages turn records into mapped, cleaned reactions with roles
from the mapping (ADR 002).

**Done.** Stage 2 (`reactions/smiles.py`): every molecule left of `>>`, canonical, unmapped, sorted;
sha1 as cache key. Mapping (`reactions/mapping.py`, `scripts/map_missing.py`, `map_shard.py`): missing
reactions as shards, RXNMapper through `python -m rxnutils.data.mapping` in its own env, results as
append-only cache parts. Stage 3 (`scripts/mapped.py`) joins the cache. Stage 4
(`reactions/clean.py`, `configs/clean.yml`): rxnutils actions in-process. Run on ORDerly `samples_2k`
and the full Chemlex in `Datasets/derived_samples/`.

**Result.**

| | ORDerly sample | Chemlex |
|---|---|---|
| stage 2 rejects (`unparsable`) | 0 (full file: 33) | 7 (PyBrOP's PF6) |
| unique reactions to map | 2,000 | 11,170 |
| mapping failures | 0 | 0 |
| mapping speed (CPU) | ~65/s | ~38/s |
| stage 4 rejects | 1 (`no_product`) | 0 |
| reactants after `reactants2reagents` | 1: 484, 2: 1,485, 3+: 30 | 2: all 11,662 |
| dynamic bonds | 2: 1,070 of 1,999 (all classes) | 2: 11,642; 1: 2; 4: 4; 6: 9; 10: 5 |

**Interpretation.** On the full Chemlex, RXNMapper gives acid + amine as the two reactants every time
and 99.8 % have exactly 2 dynamic bonds; the 20 others are mapping errors, kept (benchmark) and
counted. `remove_unsanitizable` did nothing after stage 2 and was left out. Stage 4 is slow (~150/s on
Chemlex, CGR via SDF), so it needs sharding on the cluster like the mapping.

**Next.** Stage 5, classify: `configs/classes/amide_coupling.yml`; reproduce 116/137 on the ORDerly
sample and 500/500 roles on Chemlex.

```bash
D=~/University/Phd/Dataset/Datasets; R=$D/derived_samples; P=configs/paths/local.yml
uv run python scripts/ingest.py orderly $D/samples_2k/orderly_condition.parquet $R
uv run python scripts/ingest.py chemlex $D/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx $R
for s in orderly chemlex; do uv run python scripts/rsmi.py $s $R; done
uv run python scripts/map_missing.py orderly chemlex --paths $P --shard-size 2000
for f in $R/cache/rxnmapper-0.4.3/todo/shard-*.parquet; do uv run python scripts/map_shard.py $f --paths $P; done
for s in orderly chemlex; do uv run python scripts/mapped.py $s --paths $P; uv run python scripts/clean.py $s $R; done
```

---

## 2026-10-09 — One pipeline for every source: RXNMapper, CGR check, stage 1

**Context.** More sources (ORD, USPTO, Pistachio, CHORISO, CJHIF) and classes (Buchwald–Hartwig next)
are coming; Chemlex and HiTEA are benchmarks only. I wanted every source to go through the same steps,
with only the adapter source-specific, so the atom mapping had to be the same for all of them.

**Done.** In a scratch run (scripts not kept; stages 3–5 of the new pipeline will reproduce the numbers)
I mapped reactions built from `ReactionRecord`s with RXNMapper 0.4.3 through rxnutils (separate
Python 3.12 env, `~/University/Phd/envs/rxnmapper`), then counted CGR dynamic bonds (rxnutils
`cgr_dynamic_bonds`: bonds formed + broken). I read Lin et al. 2022 (AAM benchmark) and Gimadiev et
al. 2021 (reaction curation). Decision: ADR 002 (`docs/decisions/002-one-pipeline-rxnmapper-cgr.md`);
architecture: `docs/design/rebuild.md`. Then stage 1: `record_id`, `source_rxn` (was `mapped_rxn`),
records ↔ parquet with an explicit schema, `platepu.io` (licence guard, stage report),
`scripts/ingest.py`.

**Result.**

| | Chemlex (500 random) | ORDerly sample (2,000) |
|---|---|---|
| mapping failures | 0 | 0 |
| speed (one CPU) | ~20 reactions/s | ~60 reactions/s |
| acid and amine identified | 500/500 | — |
| carboxyl C–N bond exactly right | 498/500 | — |
| same reactants as the source mapping | — | 1,835/2,000 (amide products 728/779) |
| dynamic bonds | 498 at 2; the 2 wrong at 4 | acid + amine → amide: 116 of 137 at 2 |

The 21 ORDerly couplings above 2 are coupling + ester hydrolysis or ketal removal in one step, imide
formation, data errors, Boc protections caught by the crude rule, a 2:1 bis-amide, and 2 RXNMapper
errors (the source mapping had 2). Stage 1 on the full files: ORDerly 625,697 records in 9.8 s,
Chemlex 11,669, 0 rejected; `source_id` is unique in the full ORDerly file.

**Interpretation.** RXNMapper is good enough on amide couplings to be the one mapper for every source,
and the dynamic-bond count catches both its errors and the reactions I do not want as positives
(multi-step, wrong class). Allowed value for amide coupling: exactly 2, applied to positives only;
benchmarks are never filtered by mapping. Lin 2022: 28 % of RXNMapper errors keep the same count; for
couplings these are wrong-atom choices among equal groups, which the single-reactive-group filter
removes.

**Next.** Stage 2: canonical reaction SMILES and hash (mapping cache key).

```bash
D=~/University/Phd/Dataset/Datasets
uv run python scripts/ingest.py orderly $D/orderly_condition_train.parquet $D/derived
uv run python scripts/ingest.py chemlex $D/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx $D/derived
```

---

## 2026-10-09 — Chemlex test set: adapter and test labels

**Context.** Chemlex (Zhong et al., *Nat. Commun.* 2025) is the held-out amide test set: 11,669
measured acid–amine couplings in DMF, failures included. Before any model, I need a tested reader and
one label per reaction I can trust.

**Done.** `ReactionRecord` got `yield_kind` (`"isolated"` for ORDerly, `"lcms_uv_ratio"` for
Chemlex) so the two kinds of number are never put on one scale. Chemlex adapter
`src/platepu/ingest/chemlex.py` (no chemistry; both HATU drawings kept as written); the three splits
live in a side table. Before writing the label code I counted the HATU replicates: of 101 conflicting
pairs, the median gap is 62 conversion points and 70 have one replicate at exactly 0 % (the reaction
worked once and not at all the other time), so a noise ceiling at the 20 % cut-off would describe the
wrong thing. Rule instead (`configs/test_labels.yml`, `src/platepu/evaluate/labels.py`): replicates
are kept only when all are 0 % or all are ≥ 10 %; kept reactions get the mean conversion, positive at
≥ 20 % (the paper's cut-off). Both HATU drawings count as one condition until Stage 2.

**Result.**

| | rows | reactions | positive |
|---|---|---|---|
| HATU, all | 8,454 | 7,923 | 28.7 % (rows) |
| HATU, labelled | | 7,761 | 28.3 % |
| HATU both-unseen test, raw | 1,509 | | 36.8 % (rows) |
| HATU both-unseen test, labelled | | 1,490 | 36.9 % |

HATU replicate groups (515): 189 all 0 %, 164 all ≥ 10 %, 162 inconsistent (removed). 7,408
reactions are measured once; their noise cannot be seen. The random split puts replicates of one
reaction on both sides in 221 HATU reactions; both stratified splits never do.

**Interpretation.** The replicate rule hardly moves the positive rate, but it removes 162 reactions
whose label would be a coin flip. Results will be reported on the both-unseen split (1,490 reactions,
36.9 % positive); the random split is leaky.

**Next.** Stage 2, `standardize()`: unify the HATU drawings so the explicit list in the script can go.

```bash
uv run python scripts/chemlex_labels.py ~/University/Phd/Dataset/Datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx ~/University/Phd/Dataset/Datasets/derived/chemlex_labels
```

---

## 2026-10-02 — Decision: rxnutils is the standardisation layer — Luka

**Context.** The entry below left the standardisation tool open (rxnutils vs chython, planned as ADR 001).

**Done.** Decided to use rxnutils as the main standardisation tool for every source, without the
chython comparison; written up as `docs/decisions/001-rxnutils-standardisation.md`; work plan and open
questions in `docs/design/rebuild.md` updated.

**Result / interpretation.** Basis is the ORDerly run below: rxnutils left the condition-bin counts
unchanged and gave usable duplicate keys. Our own `standardize()` (stage 2) only fills its gaps
(covalent metal salts, substrate salt rules, tautomers).

**Next.** Jan reviews ADR 001 in PR #2; run the Pistachio sample through the same pipeline.

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
