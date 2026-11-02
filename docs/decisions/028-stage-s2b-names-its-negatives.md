# 028 — Stage S2b gets its own config record and names where its negatives come from

- **Status:** Accepted, 2026-11-02, **before any validation read**
- **Supersedes:** — · **Superseded by:** —

This record decides how stage S2b is configured. It changes no earlier decision. ADR 014 item 1 (the spy
arm is primary) and ADR 016 item 2 (the naive P-against-U classifier is secondary) stand as written.

## Context

1. **S2b has two arms, and both are already decided.** The spy arm is primary (ADR 014 item 1). The naive
   arm, P against all of U with no spies and no threshold, is secondary (ADR 016 item 2). ADR 026 item 3
   keeps it and gives it a better reason: Context 4 of that record notes that the naive approach beat the
   SCAR method on 6 of 8 of Bekker's datasets, as ADR 026 records from an agent summary not yet checked
   against the paper. "What did the spy step buy?" is therefore a live question
   and not a formality.
2. **The arms differ in exactly one thing: where the negative class comes from.** They share the block, the
   P width, the U table, the learner, its parameters and the seed.
3. **The naive arm needs none of `spy_fraction`, `spy_recall`, `spy_scoring` or `n_folds`.** These are
   properties of an S2a run. Reusing `PuConfig` would put four inert keys in a naive-arm file. A config
   that describes a run it is not would make a reader believe spies were drawn.

## Decision

1. **S2b gets its own record, `PuScoreConfig`,** in `score/pu_config.py` beside `PuConfig`. Its keys are
   `model`, `feature_block`, `p_width`, `unlabeled`, `negatives`, `learner`, `learner_params`, `seed`.
   No key is inert in either arm.
2. **`negatives` picks the arm.** A stem names the S2a config whose `<stem>.negatives.parquet` supplies the
   negative class. That is the spy arm. `null` means there is no such table: train against all of U. That
   is the naive arm. Null for "this run read none" is the convention stage S1 already uses for
   `feature_block` on a baseline that reads no fingerprint.
3. **The pointer is checked, not trusted.** S2b loads the named S2a config. It raises unless
   `feature_block`, `p_width`, `unlabeled`, `learner` and `learner_params` agree with its own, and unless
   that record's value hash equals the single distinct `config_sha256` in the negatives table. A negatives
   table cut under `morgan` must never be fitted under `morgan_centre`. The spy settings stay traceable
   through that hash and are not copied into the S2b file.
4. **The arms carry different `model` values,** `pair_hgb_spy` and `pair_hgb_naive`. The stage S layout
   promises that (model, feature block, scored pair) identifies a row, so stage E needs no lookup.
5. **The spies rejoin P for the final fit.** The draw existed only to place a threshold. A spy is a
   reported positive, and holding 2,279 of them out (0.20 x 11,396, arithmetic on the command in ADR 029) would throw away a fifth of P for nothing.
   `reliable_negative` is never true on a spy, so no row enters twice. The script asserts it by checking
   that the flat pair keys of P and of the reliable negatives are disjoint.
6. **Both arms run on both blocks, so S2b is four runs and four config files:**
   `pu_score_spy_morgan_centre.yml`, `pu_score_spy_morgan.yml`, `pu_score_naive_morgan_centre.yml`,
   `pu_score_naive_morgan.yml`. This is my decision of today. It is one more cheap fit against one more
   column in the paper's table, and I chose the fit. Items 1 and 6 are mine; items 2 to 5 follow from them.
7. **One fit per run, no cross-fitting of the scored sets.** A small share of the scored pairs is in
   training. No validation pair is in P at either width, on stereo or flat keys, but exactly one is in U
   (1 of 2,615, 0.038 %). On the inventory, 208 pairs are in P on stereo keys (0.073 %) and 1,587 are in U
   (0.558 %). Counted on 2026-11-02, with the command, in `docs/design/modeling.md`, S2b, "One fit, no
   cross-fitting". These pairs are counted in the report and kept, because filtering by membership in a
   scored set would use that set to decide what may be trained on. For the validation pair the bias runs
   conservative: a pair trained on as "not known to work" scores lower. Dropping the rows would filter a
   scoring set; cross-fitting only them would put two kinds of score inside one `score_set`, which the
   layout promises are comparable. The 1,587 inventory pairs in U also steer the plate away from part of the
   grid; that is reported, and whether to do anything about it is open for me (`modeling.md`, same place).
   Item 7's premise was corrected on 2026-11-02, before this record was committed: it was first written
   citing "no validation pair is in P or in U" and the inventory figures of 2026-10-22 (406 and 669), and
   the re-run the same day found one validation pair in U and 208 inventory pairs in P on stereo keys. The
   decision survives; its justification is the counts above, not "nothing scored is in training".

## Alternatives considered

- **Reuse `PuConfig` unchanged.** Rejected: four keys that do not apply to the naive arm.
- **Two unrelated records, one per arm.** Rejected: seven identical keys and two loaders for a one-key
  difference, and the first step towards the registry CLAUDE.md §5 forbids before a third case.
- **A shared base dataclass.** Rejected: inheritance for two cases.
- **Keep the spy keys in the S2b file so the row hash covers them.** Rejected: two files holding the same
  values could disagree. The S2a hash in the report is the single source.
- **One `model` value plus a new `arm` column.** Rejected: it would break "the stage S layout unchanged",
  the one thing S2b was told to keep.
- **The naive arm on the primary block only.** Rejected for item 6.

## Consequences

- Four config files and four tables, one per (arm, block).
- `config_sha256` must accept both records. It does, because it hashes values through `asdict`.
- The naive arm cannot become the headline whatever it scores (ADR 014 item 1).
- **S2b cannot run until S2a has been run once, and it has not been.** `scripts/score_pu.py` does not exist
  and `<root>/pu/` does not exist.

## References

- `docs/design/modeling.md`, "Stage S2" and "S2b — the final fit and the two arms".
- ADR 004 (internal inventory); ADR 014 items 1 and 2; ADR 016 item 2; ADR 026 item 3 and Context 4;
  ADR 029 (the fit weight).
- `docs/literature/notes/bekker-2019-beyond-scar-propensity.md` (agent summary, not yet checked by me).
