# 015 — The sampler preserves the pair structure, and the substrate-level model is dropped

- **Status:** Accepted, 2026-10-23, **before stage U or S2 was written**
- **Supersedes:** 013 in part (item 1, how U is drawn) · 008 in part (item 5, the substrate-level
  model) · 014 in part (item 3, the justification) · **Superseded by:** —

ADRs 008, 013 and 014 are Accepted and are not rewritten (CLAUDE.md §8). This record supersedes the
parts of them that re-checking the recommendations against the data overturned. Nothing has been run.

## Context

Luka asked for every recommendation to be re-checked with numbers before any of it was built. Three
survived, two did not, and one collision between two accepted decisions surfaced.

**What survived.**

- **B8 reproduces from the production code**, not only from the scratch reimplementation it was
  declared from: `nearest_pair_scores` on `morgan_centre` gives **0.7115**, and B2 on `morgan` gives
  **0.6317** on the same path. The pre-registered value stands.
- **Frequency matching does close the popularity shortcut.** A probe that sees nothing but each
  side's literature frequency scores **0.511** against a matched U and **0.892** against a uniformly
  drawn U of the same size. The control is the point: 0.5 alone would also be what a broken probe
  returns.
- **A top-up loop is not needed.** The chance that a drawn pair is already a known positive is
  **1.11 %**, so rejected draws can simply be dropped and the deficit reported. One branch not
  written, for a case the data does not have (CLAUDE.md §8).

## Decision

### 1. U is drawn **one side at a time, stratified by size**

ADR 013 item 1 drew both sides independently from P's marginals. That closes the popularity shortcut
and opens another one, which the marginals cannot see.

**The literature is strongly disassortative in size: in P, `corr(size_acid, size_amine) = −0.388`.**
A big acid is reported with a small amine and the other way round, which is what making a complex
molecule from an elaborate fragment and a simple building block looks like. Independent draws destroy
that structure — the correlation in U comes out at **+0.003** — so a classifier given **two numbers**,
the heavy-atom count of each side, separates P from U at **AUC 0.640**. That is above B2 (0.632) and
B7 (0.598): a model could reach it having learned no chemistry at all, and the spy threshold would
then select as "reliable negatives" the pairs whose two sides are similar in size, which is a
publishing habit and not a fact about reactivity.

So each counter-pair replaces **one side**, drawing the replacement from the **same heavy-atom decile**
as the substrate it replaces, weighted by P frequency. Half the draws perturb the acid and half the
amine. Measured on a throwaway draw: the joint size probe falls to **0.510** and U's correlation comes
out at **−0.390** against P's −0.388.

This does not make U bias-free and nothing can. P's whole joint structure is shaped by what people
chose to publish. What it does is remove the one shortcut we could **name and measure**; the ones we
cannot are reported, not assumed away.

### 2. U keeps multiplicity; it is **not** deduplicated against itself

ADR 013 said U is deduplicated on the flat pair key. At k = 10 about **19 %** of draws repeat a pair,
and repeats concentrate on popular substrates — they are drawn more often. Deduplicating would
therefore pull exactly those substrates **below** their P frequency and reopen the popularity shortcut
in reverse, undoing item 1 of ADR 013 by way of its own housekeeping.

A repeated draw is legitimate weight. U keeps every draw, with an `n_draws` column so the match stays
auditable.

**Subtracting P from U stays on the flat pair key** (ADR 013 item 2), unchanged: that is about
identity between U and P, not about U's internal duplicates, and the asymmetry argument there still
holds.

### 3. The substrate-level model is dropped; ADR 008 item 5 is answered

Under matched marginals, a classifier that sees **one side only** has identical class-conditional
inputs — U's acid marginal *is* P's acid marginal, by construction — so it scores **0.5 by
construction, not by measurement**. That is the same argument ADR 008 item 6 makes for B4 to B6 within
one amine, and it means item 5's substrate-level competitor cannot be built as a P-against-U model at
all.

It could be rebuilt on P plus the reliable negatives, where the threshold has broken the marginal
match. That was rejected: it would answer a different question (is the learned signal decomposable by
side?) while looking like an answer to the original one (does the pair matter?), and a reader would
take the second meaning.

**The original question already has a measurement.** B2 sees the pair, B3 sees the two substrates
separately, and Owen's bootstrap could not separate them: **+0.010, 95 % interval [−0.070, +0.084]**,
B2 ahead in 57.6 % of resamples. The honest answer is that on this data the pair cannot be shown to
matter, and that is reported as the result it is rather than asked again of a model that cannot
answer it either.

### 4. Why popularity is removed — the corrected justification

ADR 014 item 3 justified matching the marginals with **B1 = 0.524**, reading it as evidence that
popularity carries no real reactivity signal. **That inference does not hold.** B1 scores 0.524 because
**91 % of validation pairs have an amine that appears in no literature positive at all**, so B1 is
very nearly a constant there. The number shows popularity is **unmeasurable** on this validation set,
not that it is uninformative.

The real justification is different and stronger. Popularity is a **proxy** for chemistry — the
most-reported amines are small, cheap, aliphatic primary ones (top-50 median 7 heavy atoms, 60 %
aliphatic primary, against 17 and 38.5 % in the single-partner tail) — and the model sees that same
chemistry **directly and better from the structure**, through the fingerprint. Matching the marginals
removes the reporting artefact while leaving the chemistry in the features where it belongs.

## Consequences

- The U config gains `match: size_stratified_one_side` and a bin count. The independent-draw variant
  is **not** kept as a second configuration: it is known to carry a 0.640 shortcut, and running it
  would invite the comparison to be read as a result.
- Stage U's verification gains the **joint** size probe — two features, not their sum. The sum alone
  is blind to this: it reads 0.498 on an independently drawn U, because anti-correlated sizes sum to
  about the same as uncorrelated ones. A probe that cannot fail is not a check.
- Stage S2 fits one model shape, the pair model, in two feature blocks. One fewer run.
- `docs/design/modeling.md` records B2-against-B3 as the answer to item 5 rather than listing a
  substrate-level model as pending.

## Alternatives rejected

- **Independent draws, with the shortcut reported.** Rejected: B7 is already reported beside every
  model, but a 0.640 floor reachable without chemistry would sit under every number in the paper.
- **Both samplers, stratified primary.** Rejected under ADR 014's own rule. Declaring a primary and
  then publishing a comparison with a variant we know is broken invites the broken one to be read as
  a sensitivity result.
- **The substrate model on P plus reliable negatives**, and **a second unmatched U built for that
  comparison alone**: see item 3. The second would run the comparison on data carrying a 0.892
  popularity shortcut.
