# 024 — The literature negatives are an instrument, not training labels

- **Status:** Accepted, 2026-10-30
- **Supersedes:** — · **Superseded by:** 027 in part (item 1)

This record documents an intent that is already in the code and gives its reasons. It changes no code
today. It supersedes nothing: ADR 013 item 3 (the spy technique) and ADR 014 item 1 (the spy threshold
calibrated on reported positives, revisited by ADR 019 item 3) stand as written. This record adds a
use of the negatives that those two do not name.

## Context

The 20 % yield cut-off (journal, 2026-10-29) split the 28,158 reported amide-coupling pairs three ways.
Command:

    uv run python scripts/positives.py amide_coupling ~/University/Phd/Dataset/Datasets/derived --sources orderly --benchmark-labels ~/University/Phd/Dataset/Datasets/derived_samples/private/benchmarks/chemlex/labels.parquet

| label | pairs |
|---|---|
| positive | 11,396 |
| negative | 1,121 |
| unlabeled | 15,641 |
| total | 28,158 |

1,121 of 28,158 is 4 % (arithmetic on the table above). Reading that number, I asked two things: how
does PAYN handle known failures, and would it be simpler to say "above 20 % is positive" and keep no
negatives at all.

**How PAYN handles them.** The numbers below are from `docs/literature/notes/boser-2026-payn.md`, an
agent summary I have not yet checked against the PDF line by line; page references are the note's.
PAYN works on HTE sets, where every well is measured, so it knows the true negatives. It discards them
on purpose to build the PU setting it then solves: P is the known positives, U is the rest with
positives and negatives mixed (p. 15069; SI p. S11). It then rebuilds negatives from U with the spy
technique (p. 15068–15069, Fig. 3; SI p. S11):

- a fraction of P is hidden in U as spies (final spy rate 0.2);
- a classifier separates P from U;
- the threshold is set so that (1 − tolerance) of the spies score above it (tolerance 0.05);
- everything in U at or below it is a reliable negative (RN); what is above is "undecisive" and is
  discarded (p. 15071; SI p. S14);
- every RN gets a synthetic yield of 0.0 % and joins the real positives in a yield regressor (SI p. S14);
- the class prior is not estimated and not used; the spy threshold replaces it (p. 15068–15069).

Four facts from the note bear on the question:

- RN fidelity was modest and still helped: about 50 % of RN are false negatives on Stevens, up to 97 %
  are true negatives elsewhere, and the authors' explanation is that RN is still richer in negatives
  than U (p. 15072; SI p. S24).
- The naive baseline "all of U is negative" lost to positives-only (SI p. S14). Noisy negatives can hurt.
- PAYN can check its threshold, because HTE gives it ground truth over U. That is how it reports RN
  precision and recall (p. 15071; SI p. S14).
- PAYN always runs at P:U = 1:1 and discards the excess (SI pp. S11, S16). Ours is about 1:10 (1:9.7 on
  the real draw, `docs/design/modeling.md` line 635). Its spy rate and tolerance were tuned in a regime
  I am not in.

**What the code already does.** `build_positives` writes P alone. The negatives are not training labels
anywhere. All three labels live in a separate table, `reported.parquet`, and the docstring of
`build_reported` in `src/platepu/positives/build.py` gives its two purposes: stage U subtracts **every**
reported pair, and "the spy step reads the known negatives, the only real ones the literature gives".
Read, not run: this paragraph is a code reading and not a measurement.

So the proposal "no negatives" is, for training, the design as it stands.

## Decision

### 1. The literature negatives stay out of training. They are an instrument.

No model in this project is fitted with the 1,121 as labels, as reliable negatives, or as targets.

### 2. They have three roles

1. **Exclusion from U.** Implemented. A reported reaction drawn as "unlabeled" is the contamination the
   subtraction exists to prevent, and whether the reaction worked is irrelevant to that.
2. **Validating the spy threshold.** This is the asymmetry with PAYN, and the reason for the record.
   PAYN measures RN precision against HTE truth over U. My U is constructed (ADR 013 item 1), so I have
   no ground truth for it at all. The 1,121 real failures are the only instrument I have for measuring
   what PAYN measured with HTE truth. Without them the spy threshold would be set blind and could not
   be evaluated.
3. **A held-out check on the PU model:** does it score known failures low? Most PU methods cannot run
   this check. It needs no Chemlex, so it transfers to a reaction class with no HTE set.

Roles 2 and 3 are intended uses. **Nothing is measured yet about either.** This record contains no
result for them.

### 3. Why not in training

1. **The reporting bias on a negative is not the bias on a positive.** A literature negative is a
   *reported* low yield. In practice it is a near-miss inside a paper whose other entries worked.
   Training on it writes that selection into the decision boundary. ADR 019 established that SCAR
   fails here; this is the same defect at another step.
2. **Conditions do not match, and the distribution says it better than a count does.** Among the
   **12,517 observed pairs** — the population the negatives sit in — there are **1,531** distinct
   `condition_key` values. One key holds **3,488** of those pairs, about 28 %, and **849** keys hold
   exactly **one** pair each; 381 hold four or more, and 254 pairs are reported under more than one
   key. So the reported failures are scattered across a long tail of one-off conditions rather than
   sitting beside the positives under a shared protocol. Source: `scratch/private/control_levels_out.txt`
   line 116, the saved output of the command below.
   (A note on how this number was got, because it is the point of the whole week. I first wrote
   **2,164** here. That figure is real and in the same file, but it counts distinct keys among all
   **28,633 kept reports**, not among the observed pairs — right field, right file, wrong population,
   and the population is exactly what the argument turns on. It was caught by checking, not by
   remembering.)
   `yield_kind` reads "isolated" for every report that carries a yield — 13,218 of the 28,633 carry
   one and all are "isolated", the other 15,415 are NA — so it cannot separate
   isolated from HPLC or crude yields; 68 % of the 28,633 kept reports have no temperature and 63 % no
   time. These come from `uv run python scratch/count_2x2_control_levels.py` (ADR 022, journal
   2026-10-29). The same saved output confirms the `yield_kind` claim a second way, without being asked
   to: on every level of control, requiring the four cells to share `yield_kind` removes **no** design
   that requiring them to share `time` had not already removed (833 and 833, 522 and 522, 489 and 489),
   which is what a constant field does. A pair at 15 % under a different coupling reagent therefore
   says little about HATU/DIPEA/DMF, and the table cannot tell us how little.
3. **They do not transfer.** They are 4 % of the reported pairs here. For a new reaction class the share
   could be near zero. A method that leans on them fails exactly where the project claims to work.
4. **PAYN measured that noisy negatives can hurt.** Its "U = negative" baseline lost to positives-only
   (SI p. S14).

The reading of Li, Liu and Ng (2010) in `docs/literature/2026-10-29-pu-model-choice.md` adds that the
mechanism of harm is coverage of the negative space and not misweighting. That is an argument against
training on 1,121 rows as if they sampled the negative class, and none against using them to validate.

### 4. A negative is strict, and that is the right side to err on

`build_reported` calls a pair negative only if every report has a usable yield and every one is below
the cut-off. A pair with one missing report is unlabeled. That makes the 1,121 conservative. For an
instrument that is right: a validation set that contains a pair which did not in fact fail would count
a correct model as wrong, and a smaller set of clean failures is better than a larger set with doubt in
it. The cost is power, and I have not counted how much.

## Consequences

- No code changes. `build_positives` still writes P alone; `reported.parquet` still carries the labels.
- The spy step and the model checks (roles 2 and 3) must read the negatives from `reported.parquet`
  and never from the training table. A test that no negative reaches the fit would belong to the stage
  that implements those roles; none exists today.
- **Limitation, stated in the paper.** The negatives are biased, so the RN precision estimated from
  them is biased too, in a direction I cannot name. A biased estimate beats none. The claim is that,
  not that the estimate is unbiased.
- This is a second consequence of the one SCAR defect of ADR 019 item 3, not a new defect.

### Open, not decided here

- **Spy rate and tolerance.** PAYN's 0.2 and 0.05 were tuned at P:U = 1:1 and mine is about 1:10, so
  they need re-tuning or the ratio needs matching. The P:U question is recorded as open in
  `docs/design/modeling.md` line 372, which says it goes to ADR 008. ADR 008 as committed does not
  contain it (its open items are the provenance finding and the plate arms). I cite modeling.md and
  do not reopen it.
- **Whether U is subsampled to 1:1** to mimic PAYN. The PAYN note recommends reporting both ways.
- **Whether the class prior is estimated at all.** PAYN does not. ADR 019's finding about SCAR bears on
  whether I can.
- **Whether PAYN is run as a baseline** at its published settings, which the note also recommends.

## Alternatives rejected

- **Fold the 1,121 into `unlabeled`** (the simplification I proposed myself). Rejected for one reason:
  it loses role 2. The exclusion (role 1) survives either way, because stage U subtracts every reported
  pair whatever its label. Nothing else is lost.
- **Use them as training negatives or as reliable negatives.** Rejected by item 3, all four reasons.
- **Treat all of U as negative.** Rejected on PAYN's own measurement: it lost to positives-only
  (SI p. S14). Mine would be worse, because my U is constructed and carries no outcome at all.

## References

- `docs/literature/notes/boser-2026-payn.md` (agent summary, not yet checked by me).
- `docs/literature/2026-10-29-pu-model-choice.md` (the 1,121 as instruments).
- `docs/design/modeling.md` (line 372, P:U; line 635, 1:9.7).
- ADR 013 items 1 and 3; ADR 014 item 1; ADR 016; ADR 019 item 3; ADR 022.
- `src/platepu/positives/build.py`, `build_reported`.
