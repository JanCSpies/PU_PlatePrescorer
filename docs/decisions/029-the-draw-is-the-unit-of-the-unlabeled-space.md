# 029 — The draw is the unit of the unlabeled space, and every stage S2 fit is weighted

- **Status:** Accepted, 2026-11-02
- **Supersedes:** — · **Superseded by:** —

This record is about a defect I introduced on 2026-10-30 and fixed today. It implements ADR 015 item 2,
which had already said a repeated pair is legitimate weight carried in `n_draws`. It decides nothing new
about U.

## Context

1. **The table.** Stage U writes 102,200 distinct pairs carrying 110,585 draws. `n_draws` is the row's
   multiplicity. 5,735 rows carry more than one and the largest carries 58. Command, run 2026-11-02:

       uv run python -c "import pandas as pd; u = pd.read_parquet('~/University/Phd/Dataset/Datasets/derived/unlabeled/amide_coupling/unlabeled_literature_k10.parquet', columns=['n_draws']); p = pd.read_parquet('~/University/Phd/Dataset/Datasets/derived/positives/amide_coupling/positives.parquet', columns=['family']); print(len(u), u['n_draws'].sum(), (u['n_draws'] > 1).sum(), u['n_draws'].max(), (p['family'] == 'all').sum())"

   It prints 102,200 rows, 110,585 draws, 5,735 rows above one, a largest of 58, and 11,396 positives at
   the class width. The 110,585 is also the `out` the gate reports (`scripts/unlabeled_literature.py:170`).
   The three figures below are arithmetic on that output, not measurements of their own.
2. **The defect.** `score/pu.py` as I wrote it on 2026-10-30 ignored `n_draws`.
3. **Five places say the draw is the unit:**
   - `positives/unlabeled.py:151`: "one row per distinct pair with `n_draws` as its weight".
   - `scripts/unlabeled_literature.py:67` and `:71`: the stage U **gate** sums `n_draws` per key and
     expands the table by `n_draws` before it measures. Its certified rank-AUCs, 0.5049 (acids) and 0.5095
     (amines), are therefore properties of the draw distribution and not of the rows. `:170` reports `out`
     as 110,585.
   - `scripts/counts/count_u_probes.py:88` expands U by `n_draws`.
   - `scratch/count_spy_threshold.py:111, 130, 140, 181` passes `sample_weight = n_draws` to every fit.
     **Every number ADR 026 and the journal of 2026-11-02 rest on came from a weighted fit:** the lift of
     0.26 to 0.36, the thresholds, the quartiles.
   - The P:U ratio quoted throughout as 1:9.7 is 110,585 / 11,396 = 9.704. The rows give 102,200 / 11,396 = 8.968, written 8.97.
4. **What the unweighted code would have done.** It would have fitted a U the gate never certified, dropped
   7.58 % of the drawn weight ((110,585 - 102,200) / 110,585 = 7.5824 %), counted a pair drawn 58 times once, and failed to reproduce the measurement
   it was built from.

   The certificate does not transfer: the gate measured 0.5049 and 0.5095 on the draw-expanded table, and
   an unweighted fit sees the row marginal, which the gate never measured. **How far the row marginal
   drifts from 0.5** is now measured. Command, run 2026-11-02:
   `uv run python scratch/count_s2b_inputs.py ~/University/Phd/Dataset/Datasets` (output in
   `scratch/count_s2b_inputs.out.txt`), using the gate's own `rank_auc` loaded from
   `scripts/unlabeled_literature.py`.

   | role | weighted by `n_draws` | gate certified | unweighted rows | drift |
   |---|---|---|---|---|
   | acid | 0.5049 | 0.5049 | 0.5211 | 0.0211 |
   | amine | 0.5095 | 0.5095 | 0.5332 | 0.0332 |

   A second, independent computation of mine agreed to within 0.0003 (0.5052 and 0.5094 weighted; 0.5211
   and 0.5332 unweighted). `gate_rank_auc_tolerance` is 0.02 (`configs/models/unlabeled_literature_k10.yml`)
   and both unweighted drifts exceed it. The U an unweighted fit would have been given is therefore U that
   stage U's own gate refuses to produce: `scripts/unlabeled_literature.py:125` raises past that tolerance
   and writes nothing. The drift is a property of the row marginal and not of the fit.

## Decision

1. **Every stage S2 fit is weighted, and the weight is a required argument with no default.** An optional
   weight is a weight somebody forgets.
2. **A positives row weighs 1.0, spies included.** A spy is a reported positive, not a draw.
3. **`n_draws` absent, zero or negative raises.** A zero-weight row sits in the table and not in the fit,
   which nothing in the design describes.
4. **The weight entered `fit_predict` by collapsing its learner settings into a frozen
   `Learner(params, seed)` record,** because the function had reached the five-parameter limit of
   CLAUDE.md §5. It is not a sixth parameter.
5. **The S2a table gains a `weight` column immediately after `in_p`,** and the report gains
   `reliable_negative_share_weighted` beside the row share. The two answer different questions, and the
   spy sweep already reports both.

## The measurement that proves the fix

On the hand-built test fixture every P-like pair shares one feature vector, so an in-sample score is the
weighted share of positives carrying that vector.

| | P-like spy score | threshold | reliable-negative share, rows | reliable-negative share, by draws |
|---|---|---|---|---|
| all weights 1 | 0.6394 | 0.6394 | 0.500 | 0.500 |
| one unreported P-like U pair at `n_draws` 50 | **0.3769** | 0.3769 | 0.500 | **0.145** |

By hand: 45/70 = 0.643 and 45/119 = 0.378. The learner lands at 0.639 and 0.377 at `max_iter: 30`.

Command: `uv run pytest tests/test_score_pu.py -q`. Eight new tests. Overall, run 2026-11-02:
`uv run pytest -q` prints 609 passed, 1 skipped; `uv run ruff check .` is clean; `uv run mypy src` is clean
on 39 source files.

## Alternatives considered

- **Leave it unweighted and write the deviation into `modeling.md`.** Rejected: it would silently
  contradict the gate.
- **Expand U to one row per draw.** Rejected: bigger tables, same result, and `n_draws` exists precisely
  to avoid it.
- **An optional `weights=None` argument.** Rejected for Decision 1.

## Consequences

- No S2a run has ever been made, so none already reported needs redoing.
- **Open, not decided here:** `score/pu_audit.selection_shift` still counts over rows while the fit is
  weighted, so its three measures are distances against a U the gate did not certify. Weighting them would
  change every value and needs its own tests. It is recorded in that module's docstring, and the report
  carries the reliable-negative share both ways so the mismatch is visible.
- The P:U ratio is quoted by rows and by draws from now on (8.97 and 9.70).

## References

- ADR 015 item 2; ADR 026; ADR 028.
- `docs/design/modeling.md`, "Stage S2" and "S2b — the final fit and the two arms".
- Journal, 2026-11-02.
