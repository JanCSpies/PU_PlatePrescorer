# 031 — The stage S2 validation read is fixed before the numbers are seen

- **Status:** Accepted, 2026-11-02, **before the stage S2 validation read**
- **Supersedes:** — · **Superseded by:** —

I write this record before I look at any validation ROC-AUC of a stage S2b score table. That is its whole
purpose. It fixes what will be read, how it is measured and which configuration is the headline, so that none
of the three can be chosen afterwards on the strength of a result. It supersedes nothing. It is the reading
protocol that the pre-registration of ADR 008 item 6 always implied, written down now that there is something
to read.

## Context

1. **Six stage S2b score tables exist.** Three feature blocks (`morgan_centre`, `morgan`, `morgan_near`)
   times two arms (spy, naive). Each scores the 2,615 Chemlex validation pairs and the 284,600 inventory
   pairs. None of them has been compared with a label.
2. **The baselines were read earlier.** Their validation ROC-AUCs are already in the journal:

   | baseline | validation ROC-AUC |
   |---|---|
   | B6 (amine subclass + sterics + electronics) | 0.8221 |
   | B4 (amine subclass) | 0.7426 |
   | B8 (nearest positive pair, class width) | 0.7221 |
   | B2 (nearest pair, class width) | 0.632 |
   | B7 (size only) | 0.5982 |

   Command: `uv run python scripts/counts/count_baseline_aucs.py ~/University/Phd/Dataset/Datasets
   ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/HATU_DIPEA_DMF/validation.parquet`
   (journal entry of the baseline re-run; B2 from the S1 table in the journal, `scripts/score_baselines.py`).
   **The 0.7115 for B8 that earlier records carried is superseded, not unsourced.** It has a source: ADR 015
   item 1 measured it from the production path, `nearest_pair_scores` on `morgan_centre`, alongside 0.6317
   for B2 on the same path. Both were measured on the pre-cut-off positives, and the rebuild of 2026-10-29
   replaced them with 0.7221 and 0.632. The journal's note that `grep` found 0.7115 in no source file was a
   bad test and I withdraw the inference drawn from it: a measured output is not a literal in the code, so
   its absence from `src/` says nothing about whether a command produced it. A reader comparing the new
   numbers with an older ADR should use 0.7221 for B8 and 0.632 for B2, and should read 0.7115 against
   0.6317 only as the pre-rebuild retrieval pair (ADR 030 item 4).
3. **Some score columns are heavily tied.** Distinct scores over the 2,615 validation pairs, counted on
   2026-11-02:

   | configuration | distinct scores |
   |---|---|
   | naive, `morgan` | 2,251 |
   | naive, `morgan_centre` | 478 |
   | spy, `morgan` | 2,608 |
   | spy, `morgan_centre` | 595 |

   Command:

       uv run python -c "import pandas as pd; from pathlib import Path; root = Path.home() / 'University/Phd/Dataset/Datasets/derived/private/scores/amide_coupling/HATU_DIPEA_DMF'; [print(p.stem, pd.read_parquet(p, columns=['score_set','score']).query('score_set == \"validation\"')['score'].nunique()) for p in sorted(root.glob('pu_score_*.parquet'))]"

   I have not re-run it since the two `morgan_near` tables landed. Their counts will come with the read.

## Decision

1. **The primary configuration is `pu_score_spy_morgan_near`, and it was named before the read.** The spy arm
   is primary by ADR 014 item 1, decided 2026-10-23 and re-affirmed on 2026-11-02. `morgan_near` is primary by
   ADR 030's label-free rule, decided earlier the same day. Nothing in this record chooses it. It only states
   what two earlier records already fixed, which is the point.
2. **All six configurations are read and all six are reported.** Reporting only the primary would hide the
   sensitivity. Reporting whichever scores best would be the selective choice ADR 008 item 6 exists to
   prevent. If a secondary configuration beats the primary, that is reported as "the choice of PU method or
   feature block changed the answer by X", exactly as ADR 014 item 1 requires.
3. **The primary metric is ROC-AUC on the 2,615 validation pairs, and it is tie-safe by construction.**
   ROC-AUC is the Mann-Whitney statistic, in which a tied pair of one positive and one negative contributes
   exactly one half. A heavily tied score column therefore gives a correct AUC whatever order the rows sit in.
   This matters here because the tied columns are real: 478 and 595 distinct scores for 2,615 pairs
   (Context 3). `sklearn.metrics.roc_auc_score`
   implements it this way. No tie-breaking is applied anywhere in the read.
4. **Any top-k or enrichment figure is computed tie-aware, or it is not reported.** Unlike AUC, an enrichment
   at k depends on how a tied group is ordered. An arbitrary order is an uncontrolled parameter that changes
   the number. The tie-aware form is the expectation over all orderings of the tied group (McSherry and
   Najork, ECIR 2008). **Provenance:** I know this reference only through the independent memo of 2026-11-02,
   which read it as a search-engine summary and not in full (`docs/literature/notes/2026-11-02-independent-pu-memo.md`).
   I check it against the paper before it appears in anything I publish. If the tie-aware form is not
   implemented, no enrichment number is reported at all.
5. **The interval is Owen's pigeonhole bootstrap over substrates, not over pairs.** Validation pairs share
   acids and amines, so resampling pairs would treat correlated outcomes as independent and give an interval
   that is too narrow. This is the same resampling ADR 008 item 6 uses (acid and amine flat keys). The number
   of resamples is 2,000, seed 20261102, fixed here before the run and not changed after. ADR 018 used 500. I
   depart from it on purpose: the comparison turns on the tails of this interval, and at 500 resamples a
   2.5 % quantile rests on about 12 values, at 2,000 on about 50. The cost is seconds. It is not an
   inconsistency with ADR 018, whose interval served a different question.
6. **This is the validation read. The test side stays shut.** Not `test.parquet`, not HiTEA, not now and not
   after seeing these numbers. The script refuses any file not named `validation.parquet`, by exact file name
   and not by substring. It raises an error for this. It does not use `assert`, because `python -O` would
   remove an `assert`.
7. **What this read does not decide.** It does not choose the feature block (ADR 030 did, label-free). It does
   not choose the arm (ADR 014 did). It does not tune anything. If the result suggests a change, that change
   is a new record with its own date, taken knowing it was taken after a number.

## What I expect, written before I look

A prediction made before the measurement is worth more than an explanation offered after it.

- I expect `morgan` to beat `morgan_centre` for the fitted models. The grounds are the supervised ceiling
  (0.937 against 0.892, ADR 012), the yield-signal diagnostic (0.686 against 0.583, ADR 027) and the
  resolution table of ADR 030.
- I do not know whether the PU models beat B6's 0.8221. Three published results point the other way for
  literature-trained models on new plates (Fitzner, Liu and Isayev, Beker, as summarised in the memo of
  2026-11-02; the memo read the full text of Fitzner only and the abstracts of the other two, and I have not
  checked any of them). Our own diagnostics show that the P-against-U objective cannot separate reported
  failures from reported successes (AUC 0.46 to 0.48, ADR 027).
- **A null result is a publishable contribution here, not a failure**, given that prior art, provided the
  pre-registration is clean. Saying so now is part of keeping it clean.

## Alternatives considered

- **Report the primary only.** It hides how much the answer depends on the PU method and the block.
- **Report the best of six.** That is the selective choice ADR 008 item 6 exists to prevent.
- **Bootstrap over pairs.** Simpler, and too narrow, because pairs share substrates.
- **Break ties by row order or at random for an enrichment.** The number would then depend on an arbitrary
  parameter. Not reporting it is the honest alternative.
- **Write this after the read.** It would then record what I chose, not what I committed to.

## Consequences

- The read is performed by a new count script, `scripts/counts/count_pu_aucs.py`. It takes the Datasets root
  and the validation path as arguments and hard-codes neither (ADR 021). Whether it joins the published list
  is ADR 011's question and is open.
- The six numbers and their intervals go in the journal on the day they are read, with the command.
- Nothing in stages F, U, P, S2a or S2b changes as a result of this record.

## References

- ADR 008 (item 6, the pre-registered success rule), 011 (published count scripts), 012 (supervised ceiling),
  014 (item 1, method order), 021 (no hard-coded path), 027 (yield-signal diagnostic), 030 (feature block).
- `docs/design/modeling.md` (S2a, S2b).
- `docs/literature/notes/2026-11-02-independent-pu-memo.md`: third-party summary, not a paper (McSherry and
  Najork, ECIR 2008; Fitzner 2023; Liu and Isayev 2023; Beker 2022).
