# Lab journal

Append-only. Newest entry on top; never rewrite an old entry. Each entry, written in the first person: context → done → result →
interpretation → next. Every number can be regenerated with the commands in its entry.

---

## 2026-11-02 — The single validation read was spent, and every PU configuration lost to every baseline (the day's last entry)

**Context.** ADR 031 fixed the protocol before I saw any number: the primary configuration, the metric, the
interval, and that all six configurations are reported whatever they score. It also records, before the read,
that a null result would be a contribution and not a failure, given the three published results in the memo
note of the same date.

**Done.** One command, run once:

    uv run python scripts/counts/count_pu_aucs.py ~/University/Phd/Dataset/Datasets \
        ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

2,615 validation pairs, 762 positive (29.1 %), 2,000 substrate resamples, seed 20261102. The script refuses any
file not named `validation.parquet`. Before the read I checked that refusal by handing it `test.parquet`. The
test side is untouched.

**Result.** AUC on the validation pairs, same command as above:

| configuration | distinct scores | AUC | 95 % interval |
|---|---|---|---|
| naive, `morgan` | 2,251 | 0.4618 | [0.4108, 0.5106] |
| naive, `morgan_centre` | 478 | 0.4825 | [0.4377, 0.5242] |
| naive, `morgan_near` | 2,126 | 0.4697 | [0.4179, 0.5212] |
| spy, `morgan` | 2,608 | 0.5243 | [0.4653, 0.5805] |
| spy, `morgan_near` (primary) | 2,603 | 0.5503 | [0.4791, 0.6237] |
| spy, `morgan_centre` | 595 | 0.6233 | [0.5654, 0.6798] |

Baselines, read on 2026-10-29 and not re-run today (point values; I do not have their intervals at hand):

    uv run python scripts/counts/count_baseline_aucs.py ~/University/Phd/Dataset/Datasets \
        ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

| baseline | AUC |
|---|---|
| B6 | 0.8221 |
| B4 | 0.7426 |
| B8 (class width) | 0.7221 |
| B2 (class width) | 0.632 |
| B7 (molecular size) | 0.5982 |

Spy minus naive on the same block (computed from the table above, no new run):

| block | difference |
|---|---|
| `morgan` | 0.0625 |
| `morgan_near` | 0.0806 |
| `morgan_centre` | 0.1408 |

**Interpretation.** Five readings.

1. Every PU configuration loses to every baseline. The best, 0.6233, sits below B2 (0.632), and its interval
   tops out at 0.680, under B8 (0.7221). This is an interval-to-point comparison, because the baselines'
   own intervals are not in hand.
2. Five of the six are below molecular size alone (B7, 0.5982) outright. The sixth, spy `morgan_centre`
   (0.6233), is above it as a point value, but its interval [0.565, 0.680] contains 0.5982, so it cannot be
   shown to be above size alone either.
3. The naive arm is at chance: 0.46 to 0.48, every interval covers 0.5. Ranking P against U carries no
   validation signal.
4. My prediction failed, in the opposite direction. ADR 031 records, written before the read, that I
   expected `morgan` to beat `morgan_centre` for fitted models. `morgan_centre` is the best of the six, and
   `morgan` is the worst of the spy arm. I wrote the prediction down so that it could fail, and it did.
5. The spy step bought something, and the question in ADR 016 item 2 now has a number. The spy arm beats
   the naive arm on all three blocks, by 0.06 to 0.14. It is a consistent gain from a low base, and not
   enough to matter.

One qualification. The block that the rule in ADR 030 selected (`morgan_near`, 0.5503) scored below the block
the rule disqualified (`morgan_centre`, 0.6233). Their intervals overlap, [0.479, 0.624] against
[0.565, 0.680], so by this project's own standard that is not a difference. The rule solved the tie problem it
was written for. The read does not show that it chose a worse block, and it does not show that it chose a
better one.

**Next.** Nothing is tuned and nothing is re-run in search of a better figure. That is what ADR 031 was
written to prevent. What follows is my decision, taken knowing the number: whether the project continues as
framed, and in what form the result is written up. The mechanism behind the null is already measured and
recorded. The P-against-U objective cannot separate reported failures from reported successes (AUC 0.46 to
0.48, ADR 027), so the model learned "looks publishable" and not "works".

## 2026-11-02 — The primary S2b model could not rank the plate; the block is the cause (the day's fifth entry)

**Context.** S2b ran for the first time today. The top 96 inventory pairs all had score 0.993091, shared by
1,300 pairs. That is a tie-break, not a prediction.

**Done.** Counted why. Then swept the class ratio. Then measured how many candidate pairs each block can tell
apart.

    uv run python scratch/count_top_tie_cause.py ~/University/Phd/Dataset/Datasets
    PYTHONPATH=scratch uv run python scratch/count_resolution_bar.py ~/University/Phd/Dataset/Datasets

**Result.** The 1,300 tied pairs have one feature vector between them. The ratio sweep did not help:

| effective class ratio | distinct scores in top 96 | note |
|---|---|---|
| 1:1.4, 1:2.8, 1:5.7, 1:10.0, 1:19.9 | at most 2 | top tie 5,292 at 1:2.8 |

Command for the sweep: the same run as the first command above (one script, both halves share a fit
setup, so they are comparable). Share of the 284,600 inventory pairs with a count vector no other pair
shares (second command above):

| d | inventory | validation | literature positives |
|---|---|---|---|
| 0 | 2.0 % | 27.5 % | 10.1 % |
| 3 | 68.3 % | 99.7 % | 71.9 % |
| 6 | 89.1 % | 99.7 % | 90.3 % |
| all | 91.0 % | 99.7 % | 97.7 % |

Full table in ADR 030, Context 4.

**Interpretation.** My hypothesis that the near-balanced spy ratio saturated the probabilities was wrong. The
representation is the cause: at d = 0 only 2 % of the plate's candidates are distinguishable. Validation
saturates at d = 3 and the inventory only at d = 6, so a rule built on validation would have missed it. I
decided, before any validation read, that the block is chosen by a label-free rule (smallest d within 5
points of the best resolution on the set to be scored). It gives d = 6 for amide coupling, and also at a
2-point margin. ADR 030 records it, and edits only the `Superseded by:` lines of ADR 014 and ADR 018. The
rule has been applied to one class only.

**Next.** Tests first, then add the block to `configs/features.yml` and `features/substrate.py`. Rebuild the
feature table, re-run S2a and S2b, and keep the `morgan_centre` tables for the comparison.

## 2026-11-02 — Stage S2a ran on the real data for the first time (the day's fourth entry)

**Context.** Until today the spy step was tested code with no entry point. `scripts/score_pu.py` did not
exist, so no S2a number existed either. The `n_draws` fix (ADR 029) is in, so the fit is weighted.

**Done.** Wrote `scripts/score_pu.py`, tests first (`tests/test_score_pu_script.py`, 14 cases). Gate:

    uv run ruff check .
    uv run ruff format --check . && uv run mypy src && uv run pytest -q

ruff and `ruff format --check` clean on 167 files, `mypy src` clean on 39, pytest 623 passed, 1 skipped. Then both
feature blocks, once each:

    uv run python scripts/score_pu.py amide_coupling ~/University/Phd/Dataset/Datasets/derived \
        --config configs/models/pu_spy_morgan_centre.yml
    uv run python scripts/score_pu.py amide_coupling ~/University/Phd/Dataset/Datasets/derived \
        --config configs/models/pu_spy_morgan.yml

Each wrote `<root>/pu/amide_coupling/<stem>.negatives.parquet` and `<stem>.negatives.report.json`, 104,479 rows
each = 102,200 U pairs + 2,279 spies. Time and memory are not in the report. I measured them by putting
`/usr/bin/time -l` in front of each `uv run` line above (shown here for the first):

    /usr/bin/time -l uv run python scripts/score_pu.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --config configs/models/pu_spy_morgan_centre.yml

| | morgan_centre | morgan |
|---|---|---|
| wall clock, real | 244.32 s | 262.36 s |
| maximum resident set size | 10,880,483,328 bytes | 11,731,271,680 bytes (of 51.5 GB) |

**Result.** Every figure below, except the 23.73 and the 0.1455 sourced after the table, is in the `.report.json` next to the table (the paths above). Both runs:
`spy_fraction` 0.20, `spy_recall` 0.95, `spy_scoring: out_of_fold`, `n_folds` 5, seed 20261102.

| | morgan_centre | morgan |
|---|---|---|
| spies | 2,279 | 2,279 |
| realised spy recall | 0.9504 | 0.9504 |
| threshold, out of fold | 0.0429 | 0.0426 |
| threshold, in sample | 0.0451 | 0.0449 |
| reliable negatives, out of fold | 15,250 | 14,031 |
| reliable negatives, in sample | 16,952 | 19,646 |
| share of U rows / of draws | 0.1492 / 0.1464 | 0.1373 / 0.1355 |
| marginal TV, acid / amine | 0.3405 / 0.3855 | 0.3754 / 0.3717 |
| mean heavy atoms, reliable / all U | 33.78 / 32.36 | 36.61 / 32.36 |
| single-partner share, reliable / all U | 0.3559 / 0.0000 | 0.3629 / 0.0000 |

Inventory grid, mean heavy atoms per pair, 284,600 pairs: 23.73 (validation 2,615 pairs: 38.66). Re-counted
today, `uv run python scratch/count_s2b_inputs.py ~/University/Phd/Dataset/Datasets`, output in
`scratch/count_s2b_inputs.out.txt`, count 3.

Spy-sweep share for this block at `spy_fraction` 0.20, tolerance 0.05, out of fold, mean of two seeds:
0.1455 (0.148268 at seed 20261030, 0.142818 at seed 20261031). Command, run 2026-10-30 on the cluster:
`bash scratch/run_spy_fractions.sh ~/University/Phd/Dataset/Datasets 4`; read from
`scratch/private/cluster/spy_sweep_full_f0.20_morgan_centre.csv`, column `rn_share_pairs`. The Mac run of the
same day, `scratch/private/spy_sweep_full_morgan_centre.csv`, gives 0.1455 for the same cell.

**Interpretation.**
1. The mechanism does what it was designed to do. Realised recall is 0.9504 against a configured 0.95: never
   below, barely above, which is the written tie rule (recorded in ADR 029). The out-of-fold share, 0.1492,
   is close to the 0.1455 the sweep of 2026-10-30 gave for this block at this fraction and tolerance. Two
   independent runs (cluster and Mac) agreed on that cell before the stage reproduced it, so the stage
   reproduces the instrument it was built from.
2. The cost of PAYN's in-sample choice is now a number, and on `morgan` it is large. ADR 016 item 1
   deviated from PAYN by scoring the spies out of fold and said the difference would be measured. In sample
   would call 19,646 rows reliably negative against 14,031 out of fold, 40 % more, same data, same threshold
   rule. On `morgan_centre` it is 16,952 against 15,250, about 11 %. ADR 016 can be cited with a figure.
3. The reliable negatives are not a random slice of U, in two ways that bear on the plate. The marginal
   distance from U is 0.34 to 0.39 on a scale where 0 is no shift, so the threshold undid much of stage U's
   matching. The design predicted this; now it is quantified. Second, the selected rows are larger (33.78
   heavy atoms against U's 32.36, and 36.61 on `morgan`, +4.3 there), while the inventory grid the plate is
   picked from averages 23.73. And they are thin: no substrate in all of U appears in exactly one pair,
   while 36 % of the selected set's substrates do. So the final model's negative class is systematically
   bigger and sparser than the pool it came from and the grid it will score. That is the strongest argument
   so far for the naive arm of ADR 016 item 2 being a serious candidate and not a courtesy: it never
   touches the reliable negatives, so it inherits neither bias.

**Next.** Stage S2b is unblocked, since the table exists: four runs, two arms on two blocks (ADR 028), tests
first. ADR 028's input list does not yet say that the S2a table carries the `weight` column (ADR 029), so
the spy arm reads its training weights from it and never joins back to U. It should say so.

Still open, and mine to decide: the 1,587 inventory pairs in U that steer the plate away from 0.56 % of the
grid; the feature table's licence gap; and whether stage S1's `config_sha256` moves from the file hash to
the value hash, so the column means one thing.

## 2026-11-02 — I ignored `n_draws` when I wrote `score/pu.py` on 2026-10-30, and fixed it (the day's third entry)

**Context.** Stage U writes one row per distinct pair and carries the repeats in `n_draws`. The spy step I
wrote on 2026-10-30 did not read that column. I found it today while writing the S2b design.

**Done.** Counted the table, listed where the code treats the draw as the unit, and made every S2 fit
weighted (ADR 029). Wrote ADR 028 (S2b config and its two arms) and the S2b subsection of
`docs/design/modeling.md` the same day.

Five places say the draw is the unit: `positives/unlabeled.py:151`; the stage U gate,
`scripts/unlabeled_literature.py:67` and `:71`; `scripts/counts/count_u_probes.py:88`;
`scratch/count_spy_threshold.py:111, 130, 140, 181` (so ADR 026 and the entries of today rest on a weighted
fit); and the P:U ratio 1:9.7, which is by draws.

**Result.** Command, run 2026-11-02:

    uv run python -c "import pandas as pd; u = pd.read_parquet('~/University/Phd/Dataset/Datasets/derived/unlabeled/amide_coupling/unlabeled_literature_k10.parquet', columns=['n_draws']); p = pd.read_parquet('~/University/Phd/Dataset/Datasets/derived/positives/amide_coupling/positives.parquet', columns=['family']); print(len(u), u['n_draws'].sum(), (u['n_draws'] > 1).sum(), u['n_draws'].max(), (p['family'] == 'all').sum())"

| quantity | value |
|---|---|
| rows | 102,200 |
| draws | 110,585 |
| rows with `n_draws` above 1 | 5,735 |
| largest `n_draws` | 58 |
| positives at the class width | 11,396 |

Fixture test, `uv run pytest tests/test_score_pu.py -q` (hand-built, every P-like pair shares one vector):

| | P-like spy score | reliable-negative share, rows | by draws |
|---|---|---|---|
| all weights 1 | 0.6394 | 0.500 | 0.500 |
| one P-like U pair at `n_draws` 50 | 0.3769 | 0.500 | 0.145 |

`uv run pytest -q`: 609 passed, 1 skipped. `uv run ruff check .` and `uv run mypy src` clean.

Rank-AUC of the U draws against P's marginal, weighted and not. Command, run 2026-11-02 (output in
`scratch/count_s2b_inputs.out.txt`; it loads the gate's own `rank_auc` from `scripts/unlabeled_literature.py`):

    uv run python scratch/count_s2b_inputs.py ~/University/Phd/Dataset/Datasets

| role | weighted by `n_draws` | gate certified | unweighted rows | drift |
|---|---|---|---|---|
| acid | 0.5049 | 0.5049 | 0.5211 | 0.0211 |
| amine | 0.5095 | 0.5095 | 0.5332 | 0.0332 |

A second, independent computation agreed to within 0.0003 (weighted 0.5052 and 0.5094; unweighted 0.5211
and 0.5332).

**Interpretation.**
- The unweighted code would have fitted a U the gate never certified. The gate measures its rank-AUC on the
  draw-expanded table, so its 0.5049 (acids) and 0.5095 (amines) certify the marginal of the draws. An
  unweighted fit sees the marginal of the rows, which the gate never measured, so the certificate does not
  transfer to the data the model would have been given. Measured afterwards, same day (command below), the
  row marginal drifts from 0.5 by more than the gate allows.
- The drift is a property of the row marginal, not of the fit. `gate_rank_auc_tolerance` is 0.02
  (`configs/models/unlabeled_literature_k10.yml`), so U as an unweighted fit would have seen it is U that
  `scripts/unlabeled_literature.py:125` refuses to write.
- No S2a run existed, so no reported number is wrong. The unweighted code would not have reproduced the
  spy-sweep measurement.
- Open: `score/pu_audit.selection_shift` still counts over rows (ADR 029, Consequences).

**Next.** Write `scripts/score_pu.py` and run S2a once. Then `score/pu_scores.py` for S2b (ADR 028). The
inventory-in-P counts of 2026-10-22 were re-run the same day with the command above; the new figures, and
the 1 validation pair and 1,587 inventory pairs found in U, are in `docs/design/modeling.md`, S2b.

---

## 2026-11-02 — A classifier trained on the question itself separates the 1,121 negatives; my reading of 2026-10-30 was wrong (the day's second entry)

**Context.** The entry below this one shows the P-against-U model cannot tell the 1,121 known negatives from
positives (AUC 0.46 to 0.48). On 2026-10-30 I read that as "the features carry no yield information". Today I
asked what a plain classifier trained on the question itself would do.

**Done.** Fitted positive against negative: 11,396 reported positives, 1,121 reported negatives,
`HistGradientBoostingClassifier(max_iter=200, early_stopping=False)`, 5 folds, both blocks, seeds 20261102 and
20261103. Only 333 of the 1,121 negative pairs have both substrates in stage F, so I featurised the other 792
substrates in the script, as `scratch/count_spy_threshold.py` does.

Command:

    uv run python scratch/count_pos_vs_neg.py ~/University/Phd/Dataset/Datasets <block> <seed>

**Result.** ROC-AUC, positive against negative:

| block | protocol | seed 20261102 | seed 20261103 |
|---|---|---|---|
| morgan | acid-grouped (Setting 3) | 0.7006 | 0.7077 |
| morgan | both sides unseen (Setting 4) | 0.6846 | 0.6872 |
| morgan_centre | acid-grouped | 0.6354 | 0.6121 |
| morgan_centre | both sides unseen | 0.5967 | 0.5684 |

Controls (same command): permuted labels mean 0.508 (null standard error about 0.009); summed heavy atoms
alone 0.4760.

**Interpretation.**
- My 2026-10-30 reading was wrong. It showed that a model trained on a different question cannot see the
  information. It did not show the information is absent.
- Caveat I cannot remove: a low-yield entry sits in the same scope table as its neighbours, and Setting 4
  removes shared substrates, not shared papers. Part of the margin may be "I recognise this scope table".
- Decision, mine, 2026-11-02, before any validation read: keep the pre-registered plan. The spy technique
  and `morgan_centre` stay primary; the negatives stay an instrument. I rejected switching to `morgan` (better
  by about 0.10 here, but the larger literature-against-lab shortcut, 0.859 and 0.879) and rejected training
  on the negatives. The finding goes into the paper as a result about the method.
- `configs/models/pu_spy_morgan_centre.yml` and `configs/models/pu_spy_morgan.yml` now exist, with
  `spy_fraction: 0.20`. The entry below says they do not; that was true when it was written. I fixed the value
  later today, after that entry. The reason: the measurement cannot choose the fraction, because the lift is far
  below 1 at every value (0.26 to 0.36 on morgan_centre) and the differences are at or near the seed spread.
  What does discriminate is the threshold's footing: 114 spies at 0.01, so the 95 % cut rests on about five
  rows, against 2,279 at 0.20, where it rests on about 114. The cost is nil, because stage S2b fits the final
  model on all of P; a larger fraction only thins the positives the spy classifier sees, leaving 9,117. So
  0.20, for that reason and not because PAYN used it.
- Reasons and the narrowing of ADR 024 item 1 and ADR 026 item 2: `docs/decisions/027-what-the-literature-negatives-can-and-cannot-measure.md`.

**Next.**
- `scratch/count_pos_vs_neg.py` is git-ignored. Whether it joins the published count scripts is open (ADR 011).
- A sentence on the scope-table confound for the paper's limitations.
- Whether a yield-aware objective is worth designing. Not decided.
- The validation read, naive arm beside the spy arm.

---

## 2026-11-02 — I kept the spy technique as the primary method before any validation read; the spy-fraction measurement is done and the check is weaker than ADR 026 says

**Context.** Two dates. The measurements below were run on **2026-10-30** (the cluster sweep between 18:09 and
18:32, the Mac run of the quartile file at 15:48). Only the choice of the method order was made on
**2026-11-02**, and that is what this entry's date stands for. The sweep is not today's work. ADR 026 item 5 owed a measurement of `spy_fraction` at our real P:U ratio, about 1:9.7. The
Chemlex validation read has not happened yet. Before it does, I had to settle one thing, because it cannot be
settled honestly afterwards.

**Done.**
- **Choice of the primary method, made today, before any Chemlex read.** I was offered two options. A: keep
  the spy technique as the primary method, as ADR 014 item 1 says. B: write an ADR superseding ADR 014 item 1,
  so that the naive P-against-U ranking could become the headline. I chose **A**.
- (Run on 2026-10-30.) Ran the spy-fraction measurement on the group cluster (reaction.uni-muenster.de, 32 cores, `OMP_NUM_THREADS=4`).
- Added `scratch/run_spy_fractions.sh`. It takes the Datasets root as an argument and refuses to run without
  it. That is the fix ADR 021 made for the published count scripts. `scratch/private/run_spy_all.sh` never got
  it: it hard-codes a Mac path and so runs only on the Mac.
- Added the mode `sweep_full_f<fraction>` to `scratch/count_spy_threshold.py`. The existing `sweep_full` mode is
  untouched, so every number already recorded from it still reproduces.

Command (eight processes: two blocks x four fractions, both seeds, both scorings; 28.5 minutes wall clock):

    bash scratch/run_spy_fractions.sh ~/University/Phd/Dataset/Datasets 4

**Result.**

*(a) The choice.* It was made before the validation read. That is the only thing that separates it from
choosing a headline on the strength of a result (ADR 008 item 6). So the spy technique stays primary. The naive
control of ADR 016 item 2 runs beside it. If the naive arm wins, I report it as "the choice of PU method changed
the answer by X", not as a substitution. ADR 014 item 1 stays as written; no superseding ADR is needed.

*(b) Lift* (share of the 1,121 known literature negatives caught, divided by the share of U called reliably
negative), tolerance 0.05, out of fold, seeds 20261030 / 20261031:

| block | f=0.01 | f=0.05 | f=0.10 | f=0.20 |
|---|---|---|---|---|
| morgan_centre | 0.364 / 0.330 | 0.277 / 0.299 | 0.316 / 0.348 | 0.259 / 0.275 |
| morgan | 0.296 / 0.236 | 0.182 / 0.228 | 0.164 / 0.217 | 0.221 / 0.288 |

Command: the one above.

| | value |
|---|---|
| seed-to-seed spread per cell, morgan_centre | 0.016 to 0.034 |
| seed-to-seed spread per cell, morgan | 0.046 to 0.067 |
| spies at f=0.01 | 114 |
| spies at f=0.20 | 2,279 |

Reliable-negative pairs, tolerance 0.05, out of fold, mean over the two seeds and both blocks (four rows per fraction):

| f | reliable-negative pairs |
|---|---|
| 0.01 | 20,071 |
| 0.05 | 19,123 |
| 0.10 | 17,798 |
| 0.20 | 14,190 |

Command: the one above. At f=0.01 the 95 % threshold rests on about five rows (5 % of 114).

*(c) Out-of-fold score quartiles, morgan_centre* (25 % / median / 75 %). These are from the **Mac run of
2026-10-30**, not the cluster run: `scratch/private/spy_sweep_full_morgan_centre.csv`, the row with
`seed == 20261030` and `scoring == "oof"`, columns `q_u`, `q_spy`, `q_neg` (a property of the fit, so they repeat
across the three tolerances in that file).

| set | quartiles |
|---|---|
| U | 0.054 / 0.070 / 0.081 |
| spies | 0.069 / 0.081 / 0.106 |
| the 1,121 known negatives | 0.069 / 0.081 / 0.114 |

Command for the quartiles (the Mac-only runner that `run_spy_fractions.sh` replaces):

    bash scratch/private/run_spy_all.sh

which for this file runs:

    uv run python scratch/count_spy_threshold.py ~/University/Phd/Dataset/Datasets morgan_centre sweep_full 20261030 20261031

`auc_n_vs_ptrain` (known negatives against training positives), out of fold, over all rows of the eight
`sweep_full_f*` CSVs of the cluster run: 0.4596 to 0.4802. Command: `bash scratch/run_spy_fractions.sh
~/University/Phd/Dataset/Datasets 4`, as above.

**Interpretation.**
- Every lift is far below 1. No fraction rescues the step.
- f=0.01 is highest on morgan_centre, 0.08 above f=0.20, against a seed spread of 0.03. On morgan there is no
  consistent order. Two seeds is thin. Every lift is far below 1, and the differences between fractions are at or near the
  seed-to-seed spread, except f=0.01 on morgan_centre. My own reading of what to do with `spy_fraction` is still owed.
- **This corrects how I read the lift on 2026-10-30.** I reported 0.27 as "three to four times worse than
  drawing negatives at random" and stopped. The arithmetic was right and the reading was incomplete. The known
  negatives are scored like hidden positives (their quartiles match the spies', not U's), and the model cannot
  tell a reported failure from a reported success (AUC 0.46 to 0.48). The low lift follows from the known
  negatives *being* published chemistry. The check cannot tell "the threshold cut the wrong subspace" from "the
  negatives look like positives".
- So the check of ADR 024 role 2 and ADR 026 item 2 is weaker than ADR 026 states. ADR 026 item 2 calls it "the
  only available check". It is available, but it does not discriminate. A superseding record is owed. It is not
  written: the interpretation, and what replaces the check, are mine to give.
- What the data does not show: that the spy step is wrong, or that it is right.

**Next.**
- Write the superseding ADR for ADR 024 role 2 and ADR 026 item 2 once I have settled my reading. [owed]
- `spy_fraction` is not frozen, so no `configs/models/pu_*.yml` exists yet.
- The cluster now holds a copy of `derived/` including `private/` (612 MB, `drwx------`, no Chemlex test
  split). Whether it stays is undecided.
- Run the validation read, with the naive arm beside the spy arm, as in (a).

---

## 2026-10-30 — The literature gave no value for `spy_fraction`; it said the spy step is a heuristic (the day's fourth entry)

**Context.** This is the fourth entry of today. Before choosing `spy_fraction` I asked for the literature on
PU learning without SCAR to be read. Two papers were read in full by an agent and written up as notes, which I
have not checked against the PDFs: `docs/literature/notes/bekker-2019-beyond-scar-propensity.md` and
`docs/literature/notes/kumar-2024-pulsnar-class-proportion-without-scar.md`. ADR 013 item 3 chose the spy
technique over estimating the class prior. ADR 019 established that SCAR fails here but named `e(x)`, SAR and
SNAR with no source.

**Done.**
- Read the two notes against ADR 013, 019 and 024.
- Checked each repository figure I cite against its record (ADR 008, 018, 020, 024; `docs/design/modeling.md`; the stage U entry).
- Wrote ADR 026. It supersedes nothing. I did not edit ADR 013, 019 or 024.

**Result.** No value for `spy_fraction`. Two findings.

Both principled routes out of SCAR need P and U to be separable. My sampler removes that on purpose.

| | figure | source |
|---|---|---|
| rank-AUC of literature frequency, acids | 0.5049 | stage U gate |
| rank-AUC of literature frequency, amines | 0.5095 | stage U gate |
| U pairs kept | 102,200 | stage U |
| literature-against-lab adversarial AUC, `morgan`, acids · amines | 0.859 · 0.879 | `scripts/counts/count_adversarial.py` |

Commands:

    uv run python scripts/unlabeled_literature.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --config configs/models/unlabeled_literature_k10.yml
    uv run python scripts/counts/count_adversarial.py <datasets root>

(The second takes its path argument as in ADR 020; I did not rerun either today. The values are those in the stage U entry and in ADR 018 and 020.)

- PULSNAR (Kumar and Lambert 2024) estimates α, the positive fraction among the unlabeled only, from `α_max = min(D_u_i / D_p_i)`. It needs the two densities distinguishable. If they nearly coincide α goes toward 1, and I could not tell that from "my matching worked".
- Bekker, Robberechts and Davis (2019) fit a propensity model by EM and say it "will work best if the classes are separable" (p. 9).
- Their U is not mine. They draw one sample and labelling acts on it (p. 2). I built 102,200 pairs around the published ones, so my U rows had probability zero of being labelled. Their weighted estimator is unbiased for the empirical risk on whatever set I hold (p. 5). That risk is under my sampler's marginal, not the plate's and not nature's. An EM on my tables would report my sampling design as publication propensity.

The size of the SCAR failure and the counterweight, both from Bekker et al., ROC-AUC, simulated bias (no command, authors' figures, via the note, Fig. 1b, p. 12):

| | SCAR-KM2 | SAR-EM | naive (U as negative) |
|---|---|---|---|
| diabetes | 0.500 | 0.659 | 0.632 |
| 20 Newsgroups | 0.603 | 0.891 | 0.859 |
| breast cancer | 0.971 | 0.983 | 0.992 |
| mushroom | 0.924 | 1.000 | 1.000 |

Naive beat SCAR-KM2 on 6 of 8 datasets. On four of the eight the whole question is worth under 0.05 AUC. The large wins are in two datasets.

SCAR-based α estimators usually underestimate for α of 15 % and above (Kumar and Lambert, § SNAR data, via the note). My α may be high:

| | figure | source |
|---|---|---|
| validation positive rate | 29.14 % of 2,615 | ADR 008; `docs/design/modeling.md` line 18 |
| PAYN unlabeled sets | 58–77 % positive | PAYN note, SI pp. S4–S6 |

No command for the last two rows: the first is from ADR 008's record, the second is the authors' figure via an agent note.

**Interpretation.**
- The spy step stays, and I now call it a heuristic for choosing reliable negatives, not an estimator. ADR 013 item 3 already said π was not honestly estimable. What is new is that the two methods built for the non-SCAR case both need the separability I removed.
- The failure Bekker et al. describe (p. 11) is a SCAR method choosing the wrong subspace to hit the right ratio. A spy threshold chooses how much of the space is reliably negative, so it could cut the wrong part while looking calibrated. Their data is simulated and not chemistry. I have not shown this happens here.
- The same paper says correcting the bias is often not worth much. The naive control of ADR 016 item 2 therefore stays, and is better motivated than when I wrote it.
- The 1,121 negatives of ADR 024 change standing. They were prudence. They are now the only check I have on the reliable-negative set.
- `spy_fraction` has to be measured. PAYN tuned 0.2 and 0.05 at P:U = 1:1 and mine is about 1:9.7. The research found nothing on how the threshold responds to that ratio. That is a statement about the search, which is not recorded in the repository.
- No code from either paper. Neither states a code licence in the paper. Reading a licence is not agreeing to a dependency.

**A correction from the same session.** The first reading of the recommender-systems literature on negative sampling had it that my sampler must be unusually rich in false negatives "by that literature's own logic". That does not follow, and the source was checked only after the claim was made. Hard negative sampling selects by **model score** — it takes what the model ranks highest, which is exactly why those items are often true positives; arXiv:2602.18759 names this the "False Negative Problem" and cites PinSage for it. My sampler matches on a **covariate**, the heavy-atom decile. Matching on size does not make a pair more likely to work; it makes it harder to tell apart **by size**. Those are different things. The closer analogy is the property-matched decoys of DUD-E, which the Chen note already covers (`docs/literature/notes/chen-2019-hidden-bias-dude.md`) — though that note's own wording is "closer to their control than to their failure", so reading it as "not hard-negative mining" is my inference and not Chen's sentence.

What that literature does give is the vocabulary and one clean decomposition: `y = o · r`, click equals exposure times relevance (Entropy 2025, 28(1):41, doi 10.3390/e28010041, CC BY 4.0, read through PMC12839574). For us that reads *published = someone tried and chose to report × it worked*. Both of those papers were read in this session — arXiv:2602.18759 as HTML, the Entropy one through PMC, where the fetch covered the first 100,000 of about 131,000 characters — but **neither has a literature note**, so they are session reading and not part of the record yet. Two further pointers were **not** read at all and are ids from the search alone: arXiv:2401.06452 (AutoML for PU learning, which treats the spy rate and tolerance as searched hyperparameters) and arXiv:1808.02180 (He et al., propensity falling with the probabilistic gap).

**Next.**
- Measure `spy_fraction` and the tolerance, and fix both before any Chemlex read, as ADR 023 fixed its split. Needs a count first; the sweep is not designed.
- Read the licence of both codebases at source. I will say separately whether I agree to either as a dependency.
- Open, not decided: α on the inventory grid (284,600 lab pairs, not marginal-matched; PULSNAR's support assumption is the trap); He et al., arXiv:1808.02180, for the propensity shape; whether SCAR can be tested (arXiv:2404.00145, pointed to in ADR 019). Nobody has read the first or the third.
- Check the two notes against the PDFs.

---

## 2026-10-30 — "44 % of P is never scored" was my error: every positive is scored once (the day's third entry)

**Context.** This is the third entry of today. ADR 023 says, as its headline, that about 44 % of P is never scored under the Setting-4 split, that "every positive is scored out of fold" becomes false, and that PAUC measures a subpopulation. I wrote that. When I explained the record I made it the headline, which went further than the drafting note did. ADR 023 also demanded eight counts before any code. I ran them today.

**Done.**
- Counted the scored and never-scored share of P at 3 × 3, at five partition seeds (20261030–20261034, nine cells each) and at 4 × 4. Then checked the zero with an independent fifteen-line computation.
- Compared scored against discarded P per cell, against a random-pair-subset null.
- Raked the test cells and measured what is left (ESS, zeroed rows, TVD, floor, key counts).
- Per-side sizes, base rate per cell, distinct fingerprints per candidate.
- The boosted tree at four budgets; the PSD checks in float64.
- Wrote ADR 025. I did not edit ADR 023 or `docs/design/modeling.md`.

**Result.** Commands:

    uv run python scratch/count_setting4_adr023.py
    uv run python scratch/count_setting4_adr023_extra.py
    uv run python scratch/count_tree_budget.py <seed> <stem>      # seeds 20261029-20261031 literature U, 20261029 both-sides control
    uv run python scratch/count_psd_kernels.py ~/University/Phd/Dataset/Datasets

Outputs are under `scratch/private/`.

The estimand (first two scripts):

| | measured |
|---|---|
| positives scored exactly once | 11,396 of 11,396 |
| never scored, 5 seeds and 4 × 4 | 0 |
| pairs discarded per cell | 0.4444 (= 4/9), range 0.4366 to 0.4593 |

The split is clean. Scored against discarded P, per cell:

| | median | range |
|---|---|---|
| CLES | 0.49–0.503 | 0.44–0.575 |
| KS | 0.04–0.07 | max 0.13 |
| null CLES (1/9 against 4/9 of P) | | 0.478–0.519 |
| null KS median | 0.02 | |

The raking is the real selection effect. 3.2 % of cell P pairs have no non-zero-weight U partner on a role (1,864 acid-side, 1,732 amine-side, of about 58,980). They are singletons and bigger: mean heavy atoms 19.6 against 18.4 (acid), 17.7 against 13.4 (amine).

Three figures of ADR 023 Context 4 were wrong, all optimistic (min / median / max over 45 sets):

| | ADR 023 | block | raked cell |
|---|---|---|---|
| U ESS | about 0.68 | 0.511 / 0.577 / 0.636 | 0.176 / 0.218 / 0.338 |
| U rows zeroed | 23–29 % | 21.2 / 25.7 / 30.5 % | 49.4 / 57.3 / 63.6 % |
| residual TVD | ≤ 0.0009 | ≤ 0.0011 | 0.011–0.063 acid, 0.013–0.051 amine |

The 0.68 had P in the numerator. The cell TVD has a floor, the share of P with no non-zero-weight U. Excess over the floor: at most 0.0074 (acid), 0 (amine). 2,000 raking passes instead of 200 change nothing to four decimals.

Bootstrap units after raking: cells 711 / 816 / 962 acids and 594 / 683 / 750 amines; blocks 2,637 / 2,843 / 2,999 and 2,159 / 2,287 / 2,448. ADR 023 cited 1,821 and 1,517, which are pre-raking.

Base rate per cell, P / (P + Σ raked U weight): 0.0882 / 0.0934 / 0.0970. On the ESS reading, 0.315 / 0.370 / 0.429.

Fingerprints per candidate inside the block (median): 438, 1,866, 3,573, 4,928, 5,740, 6,517, 6,879 for d = 0, 1, 2, 3, 4, 6, all. Strictly increasing in all 90 sets. No change to ADR 018 item 4.

Boosted tree (`count_tree_budget.py`): `morgan` is below `morgan_centre` at 120 and at 200 iterations in all four runs (0.008–0.017 at 120), mixed at 500, ahead at 1,500 in all four (0.005–0.026). `make_model` uses 200. The 120-iteration gaps are of the order of the fold SD (0.005–0.023). No paired fold differences were computed. Logistic regression on the same pool: 0.4780 and 0.4896.

PSD (`count_psd_kernels.py`, float64):

| | minimum eigenvalue |
|---|---|
| per-side Tanimoto, full sets, 4 cases | −6.4e-12 to −3.2e-15 (all at the floor) |
| Kronecker product, 12 samples | −3e-13 to +2.6e-2 |
| `min(k_A, k_B)`, d=0, n=1,000 | −0.21 to −0.28 |
| `min(k_A, k_B)`, d=all, n=4,000 | −7.83 to −7.97 (−2.5 % of λmax, 260–270 negatives) |

Known-answer controls passed (identity +1.000; `[[1,2],[2,1]]` −1.000; random 500×500 −16.98). Acid Gram rank at d=0 is about 258 (5,731 zero eigenvalues of 5,989); at d=all about 5,957.

**Interpretation.**
- The error was about a procedure and not a number. I counted that 44 % of pairs are discarded and did not ask what it is a fraction of. In ordinary *k*-fold nothing is discarded. Here each fold discards the mixed pairs for that fold, but the nine cells partition P, so every positive is tested once. What shrinks is the training set per fold, 4/9 instead of 8/9. The estimand does not change. The sentence for the paper is: each positive is scored once, by a model trained without its acid and without its amine.
- I caught it because ADR 023 asked for the count (its number 3). A claim about a procedure needs checking exactly as hard as a measured number does. I did not apply that to my own headline when I wrote it.
- The other three misses have a common cause: I carried a training-block measurement to the test cell. They are different objects. The cell is smaller, so the raking zeroes more of U and a floor appears that the block does not show.
- Because the cell TVD has a floor, a raw tolerance would measure singletons and not the raking. Luka's decision: block 0.002; cell, report the floor and set 0.01 on the excess over it (observed maximum 0.0074).
- The tree sentence changes from "underfits" to "its ordering depends on its budget". The 200 iterations in `make_model` sit in the inverting range, so the argument against the tree as instrument now rests on a printed number. It is still not paired and still not converged.
- A negative eigenvalue on a sample proves non-PSD. No negatives on a sample proves nothing. So the Kronecker form stands on the theorem and `min` is refuted by counterexample. The minimum stays K-NN-only.
- A trap: `score/baselines._side_similarity` is float32, whose rounding would show up as a negative eigenvalue. The next PSD check must build float64 Gram matrices.

**Next.**
- Add ADR 025 to the `Superseded by` field of ADR 023 (me, by hand).
- Put the corrected figures into the Setting-4 subsection of `docs/design/modeling.md`. Outstanding.
- Implement the Kronecker–Tanimoto kernel ridge fit; time one vec-trick product there, on the sizes in ADR 025 Context 6.
- ADR 023 number 9 (our fit against RLScore and a dense kernel, as a script) goes with the implementation.

---

## 2026-10-30 — Do the 1,121 literature negatives belong in training? No: they are an instrument (the day's second entry)

**Context.** This is the second entry of today. Reading that the 20 % cut-off had produced 1,121 pairs we know do not work, I asked how PAYN handles this, and whether it would be better to say "above 20 % is positive" and keep no negatives. Nothing was measured today for this entry. I read the PAYN note and checked the code. That is all of "Done".

**Done.**
- Read `docs/literature/notes/boser-2026-payn.md`. It is an agent summary I have not yet checked against the PDF line by line.
- Read `build_positives` and `build_reported` in `src/platepu/positives/build.py`. Nothing was run.
- Wrote ADR 024.

**Result.** These are readings, not measurements.

Counts, from the stage P run of 2026-10-29:

    uv run python scripts/positives.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --sources orderly --benchmark-labels ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/labels.parquet

| | n |
|---|---|
| positive | 11,396 |
| negative | 1,121 |
| total | 28,158 |

1,121 of 28,158 is 4 % (division of the two table values).

What the code does: `build_positives` writes P alone. The three labels live in `reported.parquet`. The docstring of `build_reported` names two purposes: stage U subtracts every reported pair, and the spy step reads the known negatives. So "no negatives in training" is already the design.

What PAYN does (note, page references as the note gives them):

| | PAYN |
|---|---|
| data | HTE, every well measured, true negatives known (p. 15069) |
| negatives | discarded on purpose, then rebuilt from U by spies (p. 15068–15069; SI p. S11) |
| spy rate / tolerance | 0.2 / 0.05 (SI p. S25) |
| RN yield | synthetic 0.0 % (SI p. S14) |
| class prior | not estimated (p. 15068–15069) |
| P:U | 1:1, excess discarded (SI pp. S11, S16) |
| "U = negative" baseline | lost to positives-only (SI p. S14) |
| RN false negatives, Stevens | about 50 % (SI p. S24) |

The one thing PAYN has that I do not: ground truth over U, so it can measure RN precision (p. 15071). My U is constructed.

**Interpretation.**
- The simplification I proposed would cost exactly one thing. Stage U would still subtract the 1,121, because it subtracts every reported pair whatever its label. What would go is role 2: using them to check the spy threshold. They are the only real failures I have, and they stand where PAYN's HTE truth stands.
- Reasons against training on them are in ADR 024 item 3, all from `uv run python scratch/count_2x2_control_levels.py`: 68 % and 63 % of reports missing temperature and time, and the spread of conditions. That last one I first wrote as `[n]`, because I had a figure in hand but no written source for it — and that placeholder is what saved it. Looking for the source, in that run's own saved output `scratch/private/control_levels_out.txt` line 116, I found that the number I remembered, 2,164, counts distinct keys among all **28,633 kept reports**, while the population my argument is about — the **12,517 observed pairs** — spans **1,531**. Right field, right file, wrong population. The distribution is the better argument anyway: one key holds 3,488 of those pairs and 849 keys hold a single pair, so the reported failures sit in a long tail of one-off conditions rather than beside the positives under a shared protocol.
- That same output confirms the `yield_kind` claim a second way, which I was not looking for: on every level of control, demanding that the four cells share `yield_kind` removes no design that demanding they share `time` had not already removed — 833 and 833, 522 and 522, 489 and 489. That is what a constant field does, and it is a cleaner demonstration than my grep for the literal string.
- Nothing is measured about checking the threshold or the model on these negatives. They are intended uses. The negatives are biased, so any precision I get from them is biased in a direction I cannot name.
- A mistake in my own notes: `docs/design/modeling.md` line 372 says the P:U question "goes to ADR 008". ADR 008 does not contain it. I noted this in ADR 024 and did not edit either file.

**Next.**
- Decide spy rate and tolerance, P:U 1:1 or full, the prior, and PAYN as a baseline. All four are open.
- Fix the dangling pointer: `docs/design/modeling.md` line 372 sends the P:U question to ADR 008, which never received it. Its line 372 also still gives ~1:242, which is the inventory U; the literature draw is 1:9.7 at line 635. Both want a pass, in their own commit.
- Count how many of the 1,121 would be lost if the negative rule were less strict.

---

## 2026-10-30 — The information axis is cut as a Setting-4 split, and RLScore stays a reference

**Context.** ADR 022 item 4 left the estimator of the information axis open. Yesterday I measured four things that say the axis as ADR 018 item 3 wrote it cannot be built as written: the acid-grouped folds test Setting 3, blocking reopens the frequency shortcut, and the probe I built to size the bias failed. The three scripts are in `scratch/` and not published, so the numbers are written out here and in ADR 023. Today I fixed the split, the raking, the estimator and the controls before any selection number exists. I also tried the authors' implementation, RLScore, to see whether it could be the estimator.

**Done.**
- Wrote ADR 023 and the subsection "The information axis — the Setting-4 protocol" in `docs/design/modeling.md`.
- Fixed the split at 3 × 3 before any selection number exists.
- Installed RLScore in an isolated throwaway environment outside the project, and ran `CGKronRLS` on a synthetic task.
- Corrected one sentence in `modeling.md`: `score` imports `features.blocks` and `io`, not nothing from `platepu`.

**Result.**

What the acid-grouped folds test.

    uv run python scratch/count_setting4_split.py

| | |
|---|---|
| test pairs with an amine seen in training, P and U, 5 folds, 5 CV seeds | 100 % |
| test pairs with both sides unseen | 0 |
| amine seen in a training positive, P / U | 68 % / 93 % |
| distinct amines / pairs | 4,658 / 113,596 |

What the 3 × 3 Setting-4 split leaves (20 partition seeds).

| | |
|---|---|
| pairs discarded | 44 % |
| training block | about 51,100 pairs |
| test cell | about 12,400 pairs |
| smallest test cell | 1,025 P, 9,776 U, 1,821 acids, 1,517 amines |
| 4 × 4: discarded / test cell | 37.5 % / about 7,000 pairs |
| 2 × 2: discarded | 50 % |

The frequency shortcut inside a block, and the raking.

    uv run python scratch/count_shortcut_probe.py

| | |
|---|---|
| rank-AUC of P above U by in-block frequency | 0.538–0.570 |
| by full-P frequency (stage gate: 0.5049 acid, 0.5095 amine) | 0.495–0.516 |
| marginal TVD before raking | 0.17–0.24 |
| marginal TVD after raking (200 passes) | ≤ 0.0009 |
| frequency rank-AUC after raking | 0.4999–0.5000 |
| block U pairs given weight 0 | 23–29 % |
| U effective sample size / nominal | about 0.68 |

    uv run python scratch/count_shortcut_probe_extra.py

| | |
|---|---|
| identity-only oracle, acid side, pair-level folds | 0.189 |
| shuffled labels, same folds | 0.498 |
| `d = all`, fitted in sample, reweighted arm (10,943 distinct fingerprints, 11,345 substrates) | 0.5001 |

The RLScore investigation. The install commands and the version output were not saved; I have the observations and not the logs, which is a gap.

| | |
|---|---|
| licence | MIT (Pahikkala and Airola, 2008–2016) |
| PyPI 0.8.1 on Python 3.13 | does not build, `longintrepr.h` |
| GitHub master on Python 3.13 | installs |
| `CGKronRLS` on SciPy 1.18.1 (project: SciPy 1.18.1, scikit-learn 1.9.1, numpy 2.5.3) | fails at runtime, `_xp` not set on `PairwiseKernelOperator` |
| `CGKronRLS` on SciPy 1.14.1 | runs |

A synthetic pure-interaction task: label = sign of one feature on each side multiplied, so no marginal signal. 3,000 pairs, 1,200 observed, 600 train, 600 test, seed 0. The Kronecker kernel there is the product of two linear per-side kernels.

I first ran this out of the session scratchpad, which would have left the two numbers citable but not reproducible — the mistake this whole day has been about. So the script now lives at `scratch/probe_kronecker_reference.py`, its docstring carries every step of building the reference environment, and it runs in either stack, computing whichever half that stack allows:

    $PROBE/.venv/bin/python scratch/probe_kronecker_reference.py   # the Kronecker half
    uv run python scratch/probe_kronecker_reference.py             # the additive half

| | AUC |
|---|---|
| `CGKronRLS`, Kronecker kernel | 0.9575 |
| logistic regression on the concatenated sides | 0.5658 |

The additive figure needed checking twice. In the reference environment scikit-learn 1.9.1 sits beside SciPy 1.14.1, which are not tuned to each other, and that half threw divide-by-zero and overflow warnings — so I re-read it in our own stack, where it is clean. Both give 0.5658, so the warnings did not move it. I would not have known that without looking.

**Interpretation.**
- The estimand changed, and that is the main result of the day. Under the 3 × 3 rule about 44 % of P is never scored, so "every positive is scored out of fold" in ADR 018 item 3 is false. The axis measures the Setting-4 subpopulation of P. I first wrote this down as a price. It is not a price. It changes what the number means, and the paper has to say it. Whether it is a sentence or a problem depends on a count I do not have: scored against discarded positives on literature frequency and heavy-atom count.
- The 0.5001 at `d = all` shows only that the one-sided marginal shortcut is gone. It does not show that U is clean, and the pairing is what the axis is meant to read. I also have nothing that shows the raking does not remove real pair signal while it removes marginal signal; zeroing 23–29 % of U is not neutral on the pair distribution. So the planted-interaction control is a gate and not a diagnostic.
- I rake the test cell as well as the training block. Otherwise the test cell keeps the 0.538–0.570 and the figure is partly a frequency reading. I have not measured the raking on a test cell. The 23–29 % and the 0.68 are from the training blocks.
- `λ` is fixed across the seven candidates, for the reason I gave against the Gaussian `γ`: otherwise there are seven instruments.
- Binarised Tanimoto, to keep one similarity in the project. I add the product beside the minimum in the model-free reading and do not touch the minimum, so ADR 020's 0.6843 stays.
- I corrected a claim of my own. I had said the boosted tree "already reported `morgan` below `morgan_centre` at 120 iterations". ADR 020 does not say that. Its 0.4780 against 0.4896 is the linear model, and the tree is there only as a 0.58–0.61 range. What ADR 020 does carry is that the tree underfits 4,096 sparse columns at 120 iterations, and the rejection rests on that alone. The per-candidate tree ordering has to be printed before I publish the clause.
- RLScore is not a dependency. Its licence is fine. The reason is the install and the SciPy contract. It needs GitHub master on Python 3.13 and then fails on our SciPy, and the fix would be a downgrade of a core library or a shim on a library I would then maintain. I write our own conjugate-gradient fit (about one module and one test module) and check it against RLScore in the isolated environment. The 0.9575 against 0.5658 reproduces ADR 020's claim in miniature. It is a synthetic control and not a result about our data.
- Two claims I use are not verified: that a product of PSD kernels is PSD and that Tanimoto on binary vectors is a kernel. If the Gram matrix has a negative eigenvalue the estimator clause fails.

**Next.**
- Counts first (CLAUDE.md §8), listed in ADR 023 Consequences: per-side counts in blocks and cells; the raked block and cell over 5 partition seeds, which also fix the TVD tolerance; the share of P never scored and how it differs from the scored; the tree's ordering of the seven candidates from `scripts/counts/count_block_ladder.py`; the minimum eigenvalue of the Gram matrices; one iteration's cost at n ≈ 51,100; the base rate per test cell.
- Put the synthetic probe into `scratch/` so the 0.9575 and 0.5658 can be regenerated.
- Open and not decided: where the code lives under `score/`, whether the protocol writes a table through `io`, whether the winner's stage F column is added before or after the Chemlex agreement check, whether the three scratch scripts are published (each hard-codes a path, ADR 021 item 2).
- The `Superseded by` lines of ADR 018 and ADR 019 still need a pointer to 023.

---

## 2026-10-29 — Can the interaction concordance index be the information axis? I counted: no

The day's third entry. The first two are below it and I do not edit them.

**Context.** I read Pahikkala et al. today (note: `docs/literature/notes/pahikkala-2025-interaction-concordance-index.md`). Their interaction concordance index works on a complete 2×2 design, two acids by two amines with all four outcomes observed. It ignores the per-side main effects, and Proposition 1 proves that an additively separable predictor scores exactly 0.5. Our information axis is plain ROC-AUC and has no such property. So I asked whether the measure can be computed on our data. Both counting scripts are in `scratch/` and not published, so the numbers are written out here and in ADR 022.

**Done.**
- Round one: counted complete 2×2 designs in the literature positives and negatives and in the Chemlex validation split, on flat keys. Checked the method against brute force on a hand-made example and on 70 random cells (seed 0).
- Round two: objection that a literature design mixes four papers. Counted the designs again at four levels of control.
- Wrote ADR 022 with the decision and the open items.

**Result.**

    uv run python scratch/count_complete_2x2_designs.py

| | literature | Chemlex validation |
|---|---|---|
| acids × amines | 6,138 × 5,015 | 94 × 112 |
| observed pairs | 12,365 | 2,611 |
| matrix density | 0.00040 | 0.248 |
| complete 2×2 designs | 10,495 | 505,865 |
| admissible, binary | 1,573 | 151,670 |
| fully crossed | 42 | 5,945 |
| admissible, continuous | 10,445 | 393,663 |

Literature observed pairs: 11,396 positive plus 1,121 negative, 12,365 after flat keys collapse 152 duplicates. A U row is a missing cell, so U contributes no design. Stereo keys move the counts by 4–7 % (literature) and 1–4 % (Chemlex).

    uv run python scratch/count_2x2_control_levels.py

| level of control | complete | admissible | fully crossed |
|---|---|---|---|
| none | 10,495 | 1,573 | 42 |
| same condition family (HATU/DIPEA/DMF) | 1,035 | 160 | 0 |
| same `condition_key` | 5,066 | 876 | 27 |
| same grant-week file (upper bound) | 3,395 | 597 | 21 |
| index gap ≤ 10 (proxy) | 2,977 | 543 | 17 |

The admissible share barely moves: 15.0 % baseline, 15.5 % family, 17.3 % `condition_key`, 17.6 % file.

Three findings.
- The document level cannot be measured. ORDerly has no patent id here. `source_document` is a grant-week block, 489 values, one `grant_date` each. So 597 is an upper bound and 543 an unvalidated proxy. Neighbouring reports at index gap 1 share agents and solvents 24 % of the time, a random pair in the same file 3.3 % (seed 0).
- The 597 designs come from 29 files. The largest holds 227, the top two about 65 %, 10 files hold one design each. 12 of the 21 crossed designs come from one group. 209 acid pairs hold the 597, 98 of them one design each. All of this is from `scratch/count_2x2_control_levels.py`, the command above — it is the only one of the three scripts that reads `extracted_from_file`, `grant_date`, `yield_kind` or `temperature`, so it is also the source of the 489 grant-week files, the 24 % against 3.3 % adjacency check, the 68 % and 63 % missing temperature and time, and the 5 × 43 and 23 × 11 group shapes. `scratch/count_complete_2x2_concentration.py` gave the baseline concentration instead: 937 acid pairs, 706 with one design, largest 144, over 485 acids and 286 amines.
- In HATU/DIPEA/DMF there are 160 admissible designs and 0 fully crossed.

Also: `yield_kind` reads "isolated" for every report with a yield, so yield comparability cannot be checked. 68 % of 28,633 kept reports have no temperature, 63 % no time.

**Interpretation.** The measure cannot be the P-versus-U axis. That is the first reason and the sharper one: U has no observed outcome, so there are no designs there, not too few. The second reason is inside the literature: the controlled subset is unmeasurable at the document level, concentrated, and has no unambiguous design in the condition I model. This is a negative result and I record it as one (ADR 022 items 1 and 2).

One lead, not a result. The surviving designs sit in what look like single-patent library tables: 5 acids × 43 amines (159 designs), and 23 acids × 11 amines over 93 cells at 37 % density (12 crossed designs). If that is right, patents hold small dense condition-controlled matrices, close to miniature HTE plates. I did not check the structures and these counts cannot.

Where the measure does fit is evaluation, on Chemlex validation and on the plate, where 151,670 admissible designs under one condition make it well estimated. I recommend that but have not decided it. It changes the pre-registration, so it has to be decided before the plate. ADR 022 item 3.

**Next.**
- Decide whether IC-index enters the evaluation (ADR 022 item 3), before the plate.
- Fix the estimator for the primary information axis before looking at any candidate's number (item 4). Candidates from the Viljanen note: Kronecker product pairwise kernel, Poly2D, Gaussian pairwise kernel, boosted tree with capacity.
- Decide whether `scratch/count_2x2_control_levels.py` is published. It would first need the §3 path fix from ADR 021 item 2.
- Unexplored: a real patent identifier from raw ORD or USPTO. Nobody has looked.

---

## 2026-10-29 — Where the validation overlap is removed: nowhere, and `leak_substrate` counted pairs

A second entry for the same day. It is a separate piece of work and it corrects a number in the entry below it, which I do not edit.

**Context.** data-guardian audited today's eight commits and raised one advisory it could not resolve. The stage F report lists overlaps between the positives and the Chemlex validation split (acid 17, amine 6, on stereo keys). The stage P report gives `leak_substrate = 215`. The guardian could not find the code that removes these overlaps from training. If the overlap were unhandled, every validation number measured so far would be contaminated. So I settled this before anything else.

**Done.**
- repo-scout traced every path through the code. I read `docs/design/` in parallel to compare intent with implementation.
- Read `src/platepu/positives/build.py` for what `leak_substrate` counts.
- Renamed the two keys: `leak_substrate` → `leak_pairs_one_side`, `leak_pair` → `leak_pairs_both_sides`.
- Regenerated the stage P report and checked that both tables are unchanged.
- Re-measured how many validation pairs have both substrates in P, against the rebuilt P.
- Wrote the rule into `docs/design/modeling.md`, new subsection under the stage F report, "The substrate overlap is reported and never removed". It is not restated here.

**Result.**

The answer to the guardian. There is no code path that filters P, the library, U, the feature table or the stage S pair assembly by membership in the validation split. Every row is kept and the overlap is only counted. This is intended.

Why. Dropping training rows because they appear in the validation set means reading the validation set to decide what may be trained on. The evaluation set then chooses the training set. That is leakage turned around, and harder to notice because it looks like caution. The design already said this for stage U ("Stage U must not … subtract the validation pairs from U — that last would use Chemlex labels to build training data"). It had never said it for P or F. That gap is why the guardian looked for code that should not exist.

What is enforced is validation against test. Validation is every labelled family row whose flat acid key and flat amine key are both absent from the test side. `scripts/benchmark_split.py` aborts if the two sides share one substrate (0 shared acids, 0 shared amines). Substrates shared between P and validation are a different thing and are allowed. The task is a pair, no validation pair is in P, and a real plate reuses known building blocks in new combinations.

What `leak_substrate` counts. It does not count substrates. `src/platepu/positives/build.py` builds a boolean mask over pairs and sums it. So 215 is the number of P pairs that share at least one substrate with Chemlex, and with all of Chemlex, test side included, not with the validation split. I read the name and not the code, and I reported 215 as substrates. The table "Chemlex leakage, substrates" in the entry below carries the same mistake. That entry stays as it is; this is the correction.

Stage P report with the new keys. Command:

    uv run python scripts/positives.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --sources orderly --benchmark-labels ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/labels.parquet

| key | n |
|---|---|
| `leak_pairs_one_side` (was `leak_substrate`) | 215 |
| `leak_pairs_both_sides` (was `leak_pair`) | 0 |

`positives.parquet` (12,023 rows across families) and `reported.parquet` (28,158 rows) are byte-identical to copies taken before the rename. Check: `pd.testing.assert_frame_equal` on each pair of files. The rename changed a name and nothing else.

Three implementations of "overlap" exist, and they answer three different questions. That is why 215 and 17 / 6 looked contradictory.

| where | key space | unit | measured against |
|---|---|---|---|
| stage F, `overlaps` | stereo | substrate keys | the family validation split |
| stage P, `leak_pairs_*` | flat | pairs | every labelled Chemlex row, test included |
| `scripts/unlabeled.py`, `share` / `pair_groups` | flat | pairs | the family validation split |

A worked case of the incomparability. At the class width, 21 of 94 validation acids are in P on flat keys, where stage F reports 17 on stereo keys. Flat keys merge stereoisomers, so a flat comparison can only match more. Both numbers are right.

The figure the paper needs: how many validation pairs have both substrates in P at all, read against the rebuilt P. Command:

    uv run python scripts/unlabeled.py amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

| width of P | both in P | one | neither |
|---|---|---|---|
| class, all conditions (what the model trains on) | 18 | 602 | 1,995 |
| HATU family | 0 | 28 | 2,587 |

Of 2,615 validation pairs. Flat-key substrate overlap at the class width (same command):

| | in P | of |
|---|---|---|
| validation acids | 21 | 94 |
| validation amines | 6 | 112 |

The design document had carried "50 of 2,615" since 2026-10-22. That was measured before the 20 % cut-off took P from 28,158 reported pairs to 11,396 positives. It is 18 now. The same rebuild moved the inventory U from 284,595 pairs to 284,600, and the lab pairs in P from 7 to 2, because a smaller P subtracts less (same command).

Gate after all of it:

    uv run ruff check . && uv run mypy src scripts && uv run pytest

`ruff check` clean, `mypy` 61 files clean, `pytest` 491 passed and 1 skipped.

**Interpretation.**

The overlap is a property of the data to report, not a defect to filter. The model is allowed to know a substrate. It is not allowed to know a validation pair, and it does not. No new ADR is needed. ADR 010 is Accepted and names the old keys, but it already states "The counts now mean distinct pairs", so the rename implements that record and does not reverse it. ADR 010 is left as it is. Its text now names keys that no longer exist, and this entry is the redirect, in the same spirit as ADR 011 item 3.

I renamed `leak_pair` too, although it never misled me. Fixing only the first key would move the trap, not remove it. A reader meeting `leak_pair` beside `leak_one_side` would reasonably assume one counts pairs and the other counts sides.

I considered merging the three overlap implementations into one function and decided against it. It would be tidier and would satisfy §5, but it would touch three stages and their reports and would not produce one new number. The protection is the table above, written into the design document, so that nobody puts two of these figures in one sentence.

The lesson of the day, and it has now happened four times. Today I wrote down a number that the command beside it would not reproduce: B8 at 0.7226 against 0.7221 (a transcription slip); the three descriptor tables in ADR 018 and the comparison in ADR 012 (both measured on a feature table that a rebuild replaced hours later); and now 215 read as molecules when the code counts pairs. Each time I caught it the same way, by running the published command again and not trusting the number beside it. I had treated "every number carries its command" as traceability, a way back to the source. It is a check, and it is worthless unless the command is actually re-run. A number with a command nobody re-runs is as reliable as a number with no command.

**Next.**
- The information axis for the block selection: PAUC with Owen's pigeonhole bootstrap (ADR 019 item 4), and wiring `score/axes.py` into `scripts/counts/count_adversarial.py`, which the consequences of ADR 018 record as outstanding.
- `CLAUDE.md` is 203 lines against its own limit of 200. Which rule goes is my call.
- The anchor ablation still owes its last step: a model trained on U against one trained on U2, scored on Chemlex validation. It cannot be done until stage S2 exists.

---

## 2026-10-29 — Yield cut-off at 20 %, stages P, U and F rebuilt, B8 implemented, and an anchor ablation that found nothing

Two days of work, one entry, because nothing was committed in between.

**Context.** I decided that a reaction works above 20 % yield. `min_yield_pct` had been left null for exactly this. Every table downstream of P depended on the old P, so they all had to be rebuilt. Reading Chen et al. then raised a worry: a model may read our construction of U and not the chemistry. I wanted a control for that, and I wanted the reading rule fixed before I measured.

**Done.**
- Set the cut-off in `configs/classes/amide_coupling.yml`. A yield outside (0, 100] is rejected with its own reason (`yield_out_of_range`), before the cut-off is applied. Without that, 453 pairs would enter P on a report above 100 %. The worst reading was 56,524,596, almost certainly a mass in the percent field.
- Added `reported.parquet`: every reported pair with its label. Stage U must subtract all of them, not only P.
- Rebuilt stage P, stage U and stage F.
- Implemented B8 as B2 on the centred block: one function with a block argument, not a second copy.
- Measured how much information the descriptors carry, and tried to compress `morgan` two ways.
- Wrote the anchor ablation: a second rule, `size_stratified_both_sides`, that replaces both sides of a pair from their own size deciles (`configs/models/unlabeled_both_sides_k10.yml`).
- Wrote ADR 020 from a measurement made on the way (below).

**Result.**

Stage P. Command:

    uv run python scripts/positives.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --sources orderly --benchmark-labels ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/labels.parquet

| | n |
|---|---|
| positive | 11,396 |
| negative | 1,121 |
| unlabeled | 15,641 |
| total | 28,158 |
| rejected, `yield_out_of_range` | 465 |
| rejected, `below_min_yield` | 1,141 |

| | before | after |
|---|---|---|
| Chemlex leakage, pairs | 1 | 0 |
| Chemlex leakage, substrates | 567 | 215 |
| HATU family | 1,177 | 627 |

Stage U on the smaller P. Command:

    uv run python scripts/unlabeled_literature.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --config configs/models/unlabeled_literature_k10.yml

| | value |
|---|---|
| pairs kept | 102,200 |
| draws | 110,585 |
| rejected | 3,375 |
| gate, acid | 0.5049 |
| gate, amine | 0.5095 |
| single-partner substrates | 67 % to 72.2 % |
| simulated null | 0.0718 to 0.0842 |

Stage F, rebuilt because `in_positives` still described the old P. Command (the stage scripts take `.../Datasets/derived`; the Chemlex benchmark lives under `.../Datasets/derived_samples`):

    uv run python scripts/features.py amide_coupling ~/University/Phd/Dataset/Datasets/derived \
        --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

| | before | after |
|---|---|---|
| substrates | 22,050 | 11,345 |
| adversarial problem, acids, lit : lab | 11,203 : 189 | 5,570 : 249 |

B-series after the rebuild. **These AUCs read the Chemlex validation labels. Unlike everything else in this entry, they are not label-free.** Two steps. First the score table:

    uv run python scripts/score_baselines.py amide_coupling HATU_DIPEA_DMF ~/University/Phd/Dataset/Datasets/derived \
        --validation ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet

then `uv run python scripts/counts/count_baseline_aucs.py ~/University/Phd/Dataset/Datasets ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet` (it refuses any file not named `validation.parquet`).

| baseline | AUC | moved? |
|---|---|---|
| B4 | 0.7426 | no |
| B6 | 0.8221 | no |
| B7 | 0.5982 | no |
| B8 (new, at class width) | 0.7221 | not comparable |
| B8 (new, at family width) | 0.7247 | not comparable |

B8 had been quoted in accepted records at 0.7115 as the bar S2 must clear. `grep` found it in no source file.

Correction, written later the same day before anything was committed: that grep proved nothing, and the
conclusion I drew from it was wrong. 0.7115 does have a source — ADR 015 item 1 measured it from the
production path, `nearest_pair_scores` on `morgan_centre`, together with 0.6317 for B2 — and a measured
output is of course not a literal in `src/`. The right word is **superseded**, not unsourced: 0.7115 and
0.6317 are the pre-cut-off pair, 0.7221 and 0.632 are the same two after the rebuild. I have corrected
ADR 031 and ADR 030 and `modeling.md` to say so before they were committed. The lesson I actually earned
here is narrower than the one I wrote down: searching the code for a number only tells you whether someone
hard-coded it, never whether a command produced it.

I first wrote 0.7221 down as 0.7226. I caught it by re-running the published command before committing, which is the second transcription error of the day and the argument for publishing the command next to every number.

ADR 012 has a transcription error. It gives 0.730 and 0.776 for the nine descriptors from `count_adversarial.py`. Command: `uv run python scripts/counts/count_adversarial.py ~/University/Phd/Dataset/Datasets` (the Datasets root, without `derived`; the count scripts join it themselves).

On the **pre-cut-off** feature table, the one ADR 012 itself used:

| | ADR 012 | the command |
|---|---|---|
| nine descriptors, amines | 0.776 | 0.735 |
| `descriptors + size` (the row below) | — | 0.776 |

Acids and both `morgan` values matched exactly there, so on that table the slip is one row in ADR 012 and nothing more.

Then the 20 % cut-off rebuilt stage F to 11,345 substrates, and **none** of ADR 012's provenance figures reproduces any more:

| block | ADR 012 (pre-cut-off) | the command today |
|---|---|---|
| `morgan`, acid · amine | 0.864 · 0.870 | 0.859 · 0.879 |
| nine descriptors, acid · amine | 0.730 · 0.776 | 0.721 · 0.700 |

The amine descriptor figure moves most, 0.776 down to 0.700, and that one matters: ADR 012's argument is that the hand-written descriptors carry *more* provenance on amines than the rooted fingerprint does. On the rebuilt table they carry **less** — 0.700 against `morgan_centre`'s 0.733 — so the direction of that claim reverses. I am not editing ADR 012. Per ADR 018's consequences this goes into a further ADR once the provenance axis is published properly, and `score/axes.py` is the first half of that work.

I checked that the table is the cause and not my own refactor: `git show HEAD:scripts/counts/count_adversarial.py`, the committed version, run on the same table prints the same figures as the rewired script on every row.

The descriptor question. Commands: `uv run python scripts/counts/count_bit_frequency.py ~/University/Phd/Dataset/Datasets`, `uv run python scripts/counts/count_fold_width.py ~/University/Phd/Dataset/Datasets`, `uv run python scripts/counts/count_centre_distance.py ~/University/Phd/Dataset/Datasets`.

I first ran these three before the stage F rebuild, on 22,050 substrates, and re-ran them
on the rebuilt table before committing. All three shapes survived; every number moved.

| block | distinct fingerprints among 11,345 substrates | bits per substrate |
|---|---|---|
| `morgan_centre` | 615 | 3.0 |
| `morgan` | 10,943 | 34.2 |

| compression of the full fingerprint | distinct |
|---|---|
| frequency floor, any useful threshold | 10,943 |
| re-fold, width 32 | 10,697 |
| re-fold, width 8 | 62 |

| bonds from the reacting atom, 0 to 6 | distinct |
|---|---|
| | 615 · 2,613 · 5,229 · 7,474 · 8,862 · 10,263 · 10,943 |

The measurement behind ADR 020. Balanced pool, 3,000 positives against 3,000 unlabeled pairs, 5-fold. Command: `uv run python scripts/counts/count_block_ladder.py ~/University/Phd/Dataset/Datasets`.

| feature set | anchored U | unanchored control |
|---|---|---|
| frequency, both sides | 0.5640 | 0.5547 |
| size alone | 0.5011 | 0.5329 |
| the nine descriptors | 0.5843 | 0.6080 |
| `morgan_centre`, both sides | 0.4896 | 0.4915 |
| `morgan`, both sides | 0.4780 | 0.4616 |

Pipeline control. Command: `uv run python scripts/counts/count_pipeline_control.py ~/University/Phd/Dataset/Datasets`.

| literature against lab, `morgan` | this path | published script | shuffled folds |
|---|---|---|---|
| acids | 0.859 | 0.859 | 0.858 |
| amines | 0.879 | 0.879 | 0.882 |

Binarising the counts changes nothing (0.4896 stays 0.4896).

The anchor ablation. Commands:

    uv run python scripts/unlabeled_literature.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --config configs/models/unlabeled_both_sides_k10.yml
    uv run python scripts/counts/count_anchor_ablation.py ~/University/Phd/Dataset/Datasets

The nearest-neighbour method lives in the tracked module `src/platepu/score/pair_knn.py`, and the script that produced these numbers is `scripts/counts/count_anchor_ablation.py`.

| | anchored U | control |
|---|---|---|
| size correlation, pair sides | −0.363 | −0.358 |
| size correlation in P | −0.377 | |
| size correlation, the sampler ADR 015 rejected | | +0.003 |
| nearest-neighbour AUC, K = 5, `morgan_centre` | 0.5668 | 0.5596 |
| nearest-neighbour AUC, K = 5, `morgan` | 0.6867 | 0.6843 |

K = 1 loses everywhere, so Chen's two-cluster signature is absent.

**Interpretation.**
- The cut-off made P smaller and cleaner. Leakage to Chemlex went to 0 pairs. It also made the popularity shortcut worse (single-partner 67 % to 72.2 %), and the null rose with it. That is why the gate is a multiple of a null and not an absolute number.
- B4 to B7 did not move. That is the control that says the rebuild behaved.
- Resolution cannot trade bias against information: a frequency floor does nothing and re-folding is a cliff, not a slope. Only what the descriptor looks at can. Locality does it.
- Fingerprints at 0.5 on the balanced pool are correct, not broken. The model cannot represent the interaction, and the pairing is all that separates P from U. ADR 020 records this and the rule it gives.
- The anchor contributes nothing these probes can find. The reading rule was fixed before any measurement: every readout a comparison, never a level, and a difference below 0.05 not a difference. By that rule all differences above are noise. This does not prove the sampler is clean. It says these probes did not find a leak.

The part worth keeping, and it is not a number. Five times these two days I built an instrument that measured itself and not the data:
- a boosted tree too short for 4,096 columns;
- a linear model that cannot represent the interaction the task is made of;
- before those, a probe blind by construction;
- a criterion no correct sampler could meet;
- a classifier on one monotone feature.

Three times I was saved only because the value was impossible: a richer fingerprint cannot score below the poorer one contained in it, and a classifier cannot fall below chance on honestly split data. Twice I was saved by a control written in advance. Those two are the results I trust.

Each time the correct rule already existed somewhere in the project and I reached for a function that did not carry it. It is the same defect I removed in the morning, when four copies of "fingerprint column to matrix" became one. It came back twice before evening.

I also made two wrong diagnoses in a row for the same symptom, unshuffled folds and then raw counts. Both were guesses. What ended it was a control with a known answer, not a sixth fix.

**Next.**
- Publish the seven count scripts cited above under ADR 011 (decided at the end of the day; ADR 021 records it). Every command in this entry now names a tracked path, which closes the gap I had already found once in this file.
- Commit this, split by topic.
- The single Chemlex check the ablation plan still owes.
- Stage S2, with interaction capability as an elimination criterion (ADR 020). The bar to clear is the nearest-neighbour 0.6843 on `morgan`.

---

## 2026-10-28 — Stage U built: the unlabeled space for amide coupling, after five decision records written before it existed

**Context.** The first commit landed on 2026-10-23. Since then I planned the whole of stage S2 and locked it in five decision records, written before any of it existed. Once a model has been fitted, a choice about how to judge it is no longer a choice.

- **ADR 013** fixes the shape of U: drawn from literature substrates, P subtracted on flat keys, the spy technique, both fingerprints, B8 declared.
- **ADR 014** fixes the order methods are tried in. PAYN is primary, the others are secondary. A secondary that wins is a finding, not a substitution. It names `morgan_centre` the primary feature block.
- **ADR 015** replaced the sampler after measurement (below).
- **ADR 016** makes the spy step cross-fitted. It also turns off a hidden validation split that the learner would otherwise have enabled by itself above 10,000 rows.
- **ADR 017** demotes a probe from a gate to a diagnostic.

**Done.**
- Counted the class-wide cross-product and how often each molecule is reported.
- Found that matching per-role marginals is not enough, and replaced the sampler.
- Built stage U: `src/platepu/positives/unlabeled.py`, `scripts/unlabeled_literature.py`, `configs/models/unlabeled_literature_k10.yml`, and `tests/test_unlabeled_literature.py`, written first (47 tests collected now; see Result).
- Ran it on the real amide-coupling data.

**Result.**

The size of the space. Command:

    uv run python scripts/unlabeled_literature.py amide_coupling ~/University/Phd/Dataset/Datasets/derived \
        --config configs/models/unlabeled_literature_k10.yml

The roots differ on purpose: this script takes `.../Datasets/derived`, the count scripts below take `.../Datasets` and join `derived` themselves. Copying one root into the other command gives a missing-file error.

The table is read from the `positives` block of the report this command writes (`derived/unlabeled/amide_coupling/unlabeled_literature_k10.report.json`). Three rows are arithmetic on that block, not fields in it.

| | n | from the report |
|---|---|---|
| acids | 11,543 | `acids` |
| amines | 9,959 | `amines` |
| class-wide cross-product | 114,956,737 | 11,543 × 9,959 |
| positives | 28,158 | `pairs` |
| acids in exactly one reported pair | 67.4 % | 7,785 / 11,543 (`single_partner_acids`) |
| amines in exactly one reported pair | 67.3 % | 6,704 / 9,959 (`single_partner_amines`) |

U has to be sampled. A uniform draw would let a classifier separate P from U by how often a molecule is reported, not by whether the pair fits. That is the provenance problem of ADR 008 in a different feature.

The size finding. The literature is disassortative in size: a complex fragment is coupled to a simple building block. Command (section 1 of its output; the independent-draw rows are section 2):

    uv run python scripts/counts/count_u_probes.py ~/University/Phd/Dataset/Datasets

| | value |
|---|---|
| corr(size_acid, size_amine) in P | −0.388 |
| same in U, both sides drawn independently | +0.001 |
| classifier on the two heavy-atom counts, independent draw | 0.640 |
| B2 for comparison — not from this script, see below | 0.632 |
| B7 for comparison — not from this script, and from nothing committed | 0.598 |

B2 (0.632) is printed by `scripts/counts/count_b2_vs_b3.py`, under the heading "is B3's equal AUC at both P widths one vector or two?" (the line `B6 0.822 vs B2_class 0.632`):

    uv run python scripts/counts/count_b2_vs_b3.py ~/University/Phd/Dataset/Datasets

B7 (0.598) has no committed command. ADR 010 keeps metrics against a benchmark label out of the stage report, and only the B2-versus-B3 question got a published comparison script, so this AUC exists only in a run nobody can repeat.

The decile numbers I quoted here earlier (0.510 and −0.390) came from a prototype sampler; the checks table below carries the measured ones.

Stage U on the real data. Command:

    uv run python scripts/unlabeled_literature.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --config configs/models/unlabeled_literature_k10.yml

| | value |
|---|---|
| draws | 274,534 |
| pairs kept | 253,896 |
| rejected | 2.50 % |
| time | 2.6 s |
| substrates absent from U | 0 |

Checks on that run (same command; the diagnostics come from the script's report):

| check | role | value |
|---|---|---|
| rank-AUC of one side's raw literature frequency | gate | 0.5036 |
| joint popularity probe | diagnostic | 0.539 |
| joint size probe | diagnostic | 0.489 |
| U's size correlation | diagnostic | −0.373 (P: −0.388) |
| joint popularity probe on a uniformly drawn U | reference | 0.908 |

Tests. The agent that wrote them had first checked them against a throwaway reference sampler kept outside the repository. That caught two tolerances of its own that were wrong. The count now:

    uv run pytest tests/test_unlabeled_literature.py --collect-only

| | n |
|---|---|
| tests collected | 47 |
| of those, `def test_` functions | 40 (the rest are parametrised cases) |
| of those, written for the two defects below, before the fix | 4 |

**Interpretation.** I nearly stopped once the per-role marginals matched. That closes the popularity shortcut, but a model could reach 0.640 on two heavy-atom counts having learned no chemistry. The spy threshold would then pick as "reliable negatives" the pairs whose sides are similar in size, which is a publishing habit. Replacing one side at a time from the same decile removes that.

The part worth keeping: three times in this one verification I set up the instrument wrongly, not the thing being measured.

1. I first wrote the size probe as the sum of the two sides. The sum reads 0.498 even on a U I knew was broken, because anti-correlated sizes sum to about the same as uncorrelated ones. A probe that cannot fail is not a check.
2. I set "the popularity probe must read about 0.5" as a pass criterion. No correct sampler can meet it. A U whose joint structure matched P's in every respect would be P, so something must differ, and a classifier with enough features will find it. Closing the size axis only made the degree axis visible. Stratifying that would reveal a third and remove real chemistry each time. This is why ADR 017 makes it a diagnostic. The 0.908 on a uniform U is what gives the 0.539 its meaning.
3. I measured the gate with a classifier on one monotone feature. On this draw it returned 0.3758, an AUC below 0.5 on a feature whose own rank-AUC is 0.5036. Classes are at 1:9.7 and the distributions are nearly identical, so there was nothing to fit and it fitted fold noise. The gate is the rank-AUC of the feature itself. A model belongs only where a probe has more than one feature and cannot be ranked. (Command: the one above, section 4 of its output.)

Why the numbers moved. The first figures (+0.003, 0.377, 0.892, 0.538, 0.493) came from probes in `scratch/` that I have since deleted. The script published in `6a58b84` (ADR 011) is now the only instrument for the probes, and every probe figure here is one it prints. The other tables name their own command, the one exception being B7's AUC above, which has none. From here a number without a committed command counts as not reproduced. It is the same lesson as the 0.984 in ADR 008.

Each time I was about to conclude something about the data that was a property of the instrument. I wrote that into the design, not only the number.

**Two defects in commit `3c80ac7`.** I caught both just after it went in.

1. `_pools` raises `KeyError: nan` instead of a named error when `pd.qcut` cannot bin the sizes. This happens when every substrate of a role has the same heavy-atom count, or when a substrate in the feature table has a NaN `n_heavy_atoms`. The existing `missing` check does not catch the second case.
2. `subtract_p_on` is parsed into the config and never read. The keys are hard-coded in `SIDES`, so a config saying `key_stereo` would run and silently subtract on flat keys.

The fix is written and green, tests first (the 4 tests above failed before it). I am not making the key functional. Stereo subtraction is a case the data does not have (CLAUDE.md §8).

- `src/platepu/positives/unlabeled.py` gained a `SUBTRACT_ON = "key_flat"` constant.
- `unlabeled_pairs` refuses any other `subtract_p_on`, next to the existing `match` refusal.
- `_pools` has a guard that raises `ValueError` naming the role's column and how many substrates fell in no bin.
- That one check covers both causes, a constant heavy-atom count and a NaN `n_heavy_atoms`. This is why there is no separate branch for the missing-size case.

Gate after the fix:

| gate | result |
|---|---|
| `uv run ruff check .` | passed |
| `uv run ruff format --check .` | 124 files already formatted |
| `uv run mypy src scripts` | success, 51 source files |
| `uv run pytest` | 411 passed, 1 skipped |

This is the third time this month that a check turned out to be decorative, after "the gate was a docstring".

**Next.**
- Stage S2: the spy step and the first PU model.
- Still open, on purpose: the five known positives in the inventory U under flat keys, and the positives table still deduplicated on stereo keys.

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
