# 014 — The order methods are tried in, and which feature block is primary

- **Status:** Accepted, 2026-10-23, **before stage S2 was written**
- **Supersedes:** 013 in part (it left both of these open) · **Superseded by:** —

ADR 013 is Accepted and is not rewritten (CLAUDE.md §8). It was written hours ago and nothing has been
run against it, but the rule holds anyway: this record completes it rather than editing it, so that
git shows both predate any result.

## Context

Two gaps in ADR 013 were found by re-reading it, not by running anything, and a third comes from
Luka's decision to try PAYN's method first and the alternatives after.

Each of the three can only be closed **before** a model exists. Afterwards, any of them becomes the
kind of choice that makes a result unfalsifiable.

## Decision

### 1. The method order is fixed now, and a later method cannot become the headline

**PAYN's spy technique is the primary method.** Ranking-only and class-prior estimation are
**secondary**, run afterwards as robustness, and reported as such.

If a secondary method beats the primary, **that is a finding and not a substitution.** It gets
reported as "the choice of PU method changed the answer by X", which is a result about method
sensitivity, and the pre-registered comparison stays anchored on the spy technique. Swapping the
headline to whichever method scored best is the selective choice ADR 008 item 6 exists to prevent,
and "we tried three and report the best" is how it would look to a referee whatever the intent.

**The known weakness of the primary method is recorded here, not discovered later.** The spy threshold
is set so that about 95 % of *hidden known positives* score positive, which assumes the spies
represent the positives hiding in U. Ours are **reported** positives — popular, accessible, the ones
somebody chose to publish — while the positives hiding in U are **unreported** ones. Under SAR, which
is this project's whole premise, those are different populations by construction, so the threshold is
calibrated on the wrong one.

PAYN does not have this problem: it used **synthetic** positives carved from fully-labelled HTE sets,
where the split is random and the hidden positives therefore come from the same distribution as the
spies. That is SCAR by construction. Taking the method to real literature takes it out of the
conditions it was validated in. Frequency-matched sampling (ADR 013 item 1) narrows the gap, because
spies and hidden positives then share marginal structure, but it does not close it.

This is a limitation to state in the paper, not a reason to abandon the method: it is the group's own
prior work, it makes the two directly comparable, and the alternatives carry costs of their own
(ADR 013 item 3).

### 2. `morgan_centre` is the primary feature block

ADR 013 item 4 fits both fingerprints and reports both. It did not say which is primary, which would
have left the better of two numbers free to become the headline after the fact.

**Primary: `morgan_centre`. Secondary: `morgan`.** On ADR 012's evidence the rooted fingerprint
carries less provenance (0.707 / 0.752 against 0.862 / 0.870) and finds analogues better (0.711
against 0.632); the whole-molecule fingerprint has the higher supervised ceiling (0.937 against
0.892). The primary is chosen on the axis this project has decided matters — provenance — and in line
with ADR 008 item 2, which already preferred the rooted fingerprint over hand-written subclasses.

If `morgan` does better once fitted, that is reported, and it is a finding about what fitting
extracts, not a reason to rename it primary.

### 3. Why removing popularity from U is justified, and not merely convenient

ADR 013 item 1 matches U's marginals to P's so that substrate popularity cannot separate the two
classes. That deliberately discards a signal which is **not pure nuisance**: a substrate reported 321
times plausibly is more generally reactive, so some real information goes with the bias.

The justification is a measurement, and it belongs in the record rather than in somebody's head.
**B1 — a baseline that reads nothing but how often each substrate is reported — scores 0.524 on the
validation set.** In the population the claim is evaluated on, popularity tells us almost nothing
about whether a reaction works. So the signal given up is close to nothing, while the bias removed is
the one that would otherwise let the model separate P from U without ever comparing the two molecules.

## Consequences

- The result table names one primary configuration: spy technique, `morgan_centre`, k = 10. Everything
  else in it is secondary and labelled so.
- The spy-threshold limitation in item 1 is a sentence the paper has to carry.
- Nothing here touches ADR 008 item 6.
