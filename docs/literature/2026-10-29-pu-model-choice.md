# Which PU model fits our data — a reading of 27 papers

Written 2026-10-29, after reading the papers listed at the end. Agent summaries were used for the
first pass; everything marked **checked** below I read in the source myself. Nothing here is a
decision — decisions go in ADRs. This is the evidence they would rest on.

## The question

Which model should stage S2 fit? PAYN used a spy step and a CatBoost regressor on synthetic P and U
carved out of fully labelled HTE data. Our data is not that, and the differences turned out to
matter more than the similarities.

## Our setting, stated precisely

| | |
|---|---|
| **P** | 11,396 pairs, positive because somebody **published** them at ≥ 20 % yield |
| **U** | 102,200 pairs **we constructed**: one side replaced from the same heavy-atom decile, weighted by literature frequency. Marginals matched to P **by design** |
| **known negatives** | **1,121** pairs whose every reported yield is below 20 % |
| | They are **doubly selected**: somebody had to try them *and* publish the poor result |
| **unlabelled reported** | 15,641 pairs with no usable yield |
| **labelling** | depends on substrate popularity, on how easy the chemistry is, and on unobservables |
| **π** | unknown |
| **output** | a **ranking** of a substrate matrix, not a calibrated probability |
| **transfer** | must also work on a class with no HTE validation set at all |

Two features of this list have no counterpart in the PU literature: the unlabeled set is
**constructed**, and the bias sits on **both** sides at once.

## The first finding: no published method covers us

Bekker & Davis's 2020 survey organises the field by assumptions on how the **positives** are
labelled. A collected set of biased negatives has no place in that taxonomy, and it is not among the
five open problems in its closing section. PUbN is the one paper that models biased negatives, and
it does so while assuming the positives are a representative sample — the opposite end of our
problem. **checked**: its constraint `p(s = +1 | x, y = +1) = 1` says every positive is labelled.

Worse, our regime is probably not even SAR. Publication depends on the **realised yield**, and the
yield is the class variable. Bekker's taxonomy names that case — SNAR — and then leaves it: nothing
in the literature handles it.

## The second finding: two criteria eliminate most of the field

These are worth more than the survey, because they decide by inspection rather than by reputation.

**1. Any objective containing the subtraction term is out.** The unbiased PU risk contains
`−π·E_P[ℓ(−g)]` — the positives counted as negatives. **checked** in Xu et al. (aaPU §3.3): that
term is an unbiased estimate only when P represents the whole positive distribution. Ours does not.
So uPU, nnPU, PUbN and PNU fail in the first line of the algebra, not in their hyperparameters. What
survives are objectives that use P **only** in the positive-loss term.

**2. Anything resting on `p(x) = π p₊(x) + (1−π) p₋(x)` is out,** because our U is constructed and
the identity is simply false for it. This kills the same family a second time. The propensity-weighted
estimator of Bekker et al. is the one that bends rather than breaks: it is unbiased *conditional on
the sample handed to it*, which for a plate may be what we want.

**The second criterion also closes the ranking objectives, which I had hoped it would not.** Sakai,
Niu & Sugiyama derive an unbiased PU-**AUC** risk, and it looked like the one route to a ranking
objective with theory behind it. It is not: their Eq. (3) *defines* the unlabeled sample as the
mixture, and the derivation of the risk is one substitution of that identity, moved from points to
pairs. π is required explicitly — it appears as a divisor in the risk and in both bounds — and their
own sensitivity study is worst in the small-π corner, which is ours. So the obstruction is the
rewriting trick itself, not classification versus ranking.

## What survives, and why it is defensible rather than merely left over

The same paper supplies the positive result, in its Related Work. It criticises an earlier pairwise
ranking SVM for "merely regarding unlabeled data as negative" — a naive P-vs-U ranker — and names the
term that method drops: the **within-P pairwise** term, carrying the factor `θ_P/θ_N`. Their own
words are that ignoring it is not negligible when the class prior is *not* sufficiently small.

Read from our side, that is a justification rather than a criticism. Our π is small. The dropped term
is the only one carrying π, so the naive P-vs-U ranker is exactly the π-free, mixture-free objective,
and the size of what we give up is bounded by a quantity the paper itself calls small in our regime.
This is the citable reason for the baseline we would otherwise have had to defend by assertion.

## The third finding: for a ranking, π never matters

This recurs in three independent places, and it is the most useful structural fact we have.

- Elkan & Noto: under SCAR the P-vs-U classifier preserves the order of `Pr(y=1|x)`, needing neither
  `c` nor `π`.
- Wei et al. (2022), Theorem 1, restated by Wang et al. (ICLR 2026) as the proxy AUC:
  `AUC_PN = AUC_PU/(1−π) − π/(2(1−π))`. Affine and increasing, so the ordering is preserved exactly.
- DETM (below): the target posterior is increasing in `Λ₁/Λ₂`, so **checked** the ranking score is
  `xᵀ(β₁ − β₂)` and π moves only the threshold.

We do not need π. We should stop trying to estimate it — see the next section.

## The fourth finding: π is not merely unknown here, it is unidentifiable

Every identification result needs either an x-independent labelling mechanism or **irreducibility** —
a region of feature space occupied by negatives and by no positives. We have neither, and the second
fails for a reason worth putting in the paper: **there is no region of acid × amine space that is
purely negative.** Almost any acid couples with almost any amine under some conditions; failure is a
matter of degree, not of a disjoint region. What little overlap-free space might have existed, our own
U construction removed, since U is drawn from P's own substrates.

The honest position is to say π is unidentifiable, cite the condition, and rank instead.

## The best structural match: DETM

Liu, Yeh, Zhang, Tian & Li (2025), *Positive and unlabeled data: model, estimation, inference, and
classification*, JASA 120(552), 2547–2558; arXiv 2407.09735. **checked** in the source.

It models two density ratios against the same reference, the labelled positives:

    Λ₁(x) = p_t(x | y=1) / p_s(x | y=1)      Λ₂(x) = p_t(x | y=0) / p_s(x | y=1)

both as `log Λ = α + xᵀβ`. Four things make it the closest fit we have found:

- **Λ₁ is our reporting bias, isolated in its own term.** `β₁ = 0` is SCAR, so `β₁ ≠ 0` is a
  *testable* statement of the bias, and the paper supplies a goodness-of-fit test for it — the test we
  wanted and could not run with Teisseyre et al., which needs π and a single-sample design.
- **π is not required.**
- **The framework is two-sample** (source and target domains, transfer learning), which is our shape.
  Every other paper assumes a single sample.
- **High-dimensional features are anticipated**: `x` may be replaced by a known `T(x)`, and they
  suggest a pretrained embedding for images.

It also disposes of the alternatives in one paragraph: Bekker's shadow variables are "generally
unknown, let alone their existence"; Gong et al. require propensity below 0.5 for every unlabeled
point; Furmańczyk et al. require separability.

**The open question it raises is ours to answer, not theirs.** DETM needs a target domain. Our
constructed U is a legitimate one formally, but it is built from P's own substrates and is not the
space we want to score. The **inventory** — 284,595 real pairs — is that space. Using it as the
unlabeled sample would fit the framework far better and would sidestep the constructed-U problem
entirely, but ADR 008 item 4 makes the inventory a scoring set and not training data. That is a
decision, and it belongs in an ADR.

## The convergence that reframes the question

Two state-of-the-art methods, found independently and solving different halves of our problem, turn
out to fail on the **same** assumption — and to be satisfied by the same change.

**Kumagai et al. (NTT, arXiv 2609.10928, September 2026)** maximise AUC from **biased** PU data. Its
Figure 1 caption states that for AUC maximisation under SAR — biased labelled positives — "there are
no studies for this setting", which confirms independently what our own reading concluded. It
dismisses the SAR methods we rejected for the reasons we found: Gerych et al. need separated class
distributions, He et al. and Teisseyre et al. need restricted functional forms for the selection
probability, and performance "drastically deteriorates" when those are violated.

Its mechanism is **positive-confidence**: a probability attached to each labelled positive. The
requirement is unusually weak — **checked**: the rewritten risk induces a Bayes-optimal ranking when
the confidence is *any strictly increasing transformation* of the true posterior, so it need not be
calibrated. We have a candidate for it in every positive: the **measured yield**.

But **checked**: it requires the unlabeled data to be drawn from the **true marginal density**. Our
constructed U is not.

That is the fourth method in a row to fall on that single assumption, and DETM needs a target domain
for the same reason. So the binding constraint is not which model we choose. It is **what our
unlabeled set is**. And both of these methods would be satisfied by the same substitution: the
**inventory grid**, 284,595 real pairs from the space we actually intend to score, is a sample from a
real marginal in a way our constructed U can never be.

That makes ADR 008 item 4 — the inventory is a scoring set, not training data — the single most
consequential open decision in the project, and it is now blocking two different methods rather than
one.

## The general result, which is worth more than any single paper

Nine derivations have now been checked against our data and all nine fail at the same step. That is
not nine footnotes; it is one result, and it should be written as one.

**Every assumption-light guarantee in PU learning holds because the contamination is a scalar.**
Menon et al. (ICML 2015) are the ancestor and the most explicit: their framework is mutually
contaminated distributions with mixing rates **α, β ∈ [0,1]** — numbers, not functions — and both
pools sharing the same class-conditionals. Their own explanation of why BER and AUC are immune is
that `(1,1)` is the only noise-agnostic eigenvector of a **constant 2 × 2 matrix**. Replace the
constants with functions of `x` and the matrix becomes an operator; the eigenvector argument has no
content, and there is no affine relation left, not even an order-preserving one.

Our contamination is a function of `x` **in both directions at once**: which pairs got published, and
which pairs we chose to enumerate. So our (P, U) is not a violated instance of the framework — it is
**not a point in its parameter space at all**. That is a cleaner thing to write than "the assumption
is violated".

The one weaker sufficient condition that would have rescued the ranking, derivable from their
Proposition 5: the contamination may depend on `x`, but **only through `η(x)`**, the feasibility
itself. Then the corrupted score stays a fixed increasing transform of the true one and the ranking
survives with no other structure. Ours does not qualify — a popular easy amine and a rare easy amine
share an `η` and have wildly different publication propensities. Worth stating in the paper as what
would have had to hold.

Menon §2.2 is also older and stronger authority than anything ADR 019 currently cites for the π
argument: the corrupted distributions contain no information about π in general, and the corruption
rates are themselves non-identifiable without further assumptions.

**One correction to our own reading.** The AUL criterion, which looked like a second, independent
route to prior-free evaluation, is not. **checked by me**: with the pool fraction written as
`p = π·TPR + (1−π)·FPR`, `AUL = ∫TPR dp = π/2 + (1−π)·AUC` — the same identity as Wei et al.'s
Theorem 1. It is our PAUC in lift coordinates, and it adds an explicit SCAR assumption. Keep PAUC.
That two literatures arrived independently at the same statistic is worth a sentence; it is not a
second instrument.

## The one constructive path out

Um-AUC (arXiv 2305.15776) is the only one of the nine whose assumption is about a **relation between
two pools** rather than a match to an unobservable reality. It needs both unlabeled pools to be
mixtures of the *same* class-conditionals with *different* priors, and — its stated contribution —
only the **order** of those priors, never their values. No π anywhere.

Our two obvious pools do not qualify, and our own measurement says so: inventory pairs average
**23.73** heavy atoms against **38.66** for the validation pairs (`docs/design/modeling.md:366`). A
marginal gap that size cannot come from a change in mixing proportion alone. Applying the method to
that pairing would optimise the separation of shelf chemistry from paper chemistry — the domain
shift, not feasibility. A confidently wrong objective is worse than a biased one.

**But the assumption is constructible rather than hoped for**, which is the difference. Build both
pools by one enumeration from one substrate universe, and induce the prior gap **deliberately** — for
instance by filtering the second with a coarse, label-free feasibility rule. Then the pools share
their class-conditionals by construction instead of by assumption, and the only thing left to check
is that the filter enriches rather than changes composition — checkable by comparing the two pools'
marginals, which is exactly the measurement that just disqualified the obvious pairing.

Whether such a filter exists for amide coupling, and is credibly orthogonal to substrate class, is a
counting question and not a reading question. It should be answered before any of this is built.

Also worth keeping from the same reading: the weighted-IPM score, `mean_P k(z,·) / mean_U k(z,·)`,
touches π only in its cutoff and never in the score, so its **ranking is prior-free by construction**.
Against a matched background it measures property-matched enrichment — which is either exactly what
we want or exactly the trap, depending on how clean the matching is. It is the only output object
among the nine that keeps a meaning when the marginal assumption fails. It also supplies the
principled reason Wei et al. preferred an IPM evaluator to an AUC one, which until now rested on a
single empirical sentence: an IPM is an unpaired two-sample mean difference, while AUC is a
U-statistic over dependent pairs.

## What we must not claim

**The AUC robustness theorem does not cover us.** **checked**: Charoenphakdee et al. work in the
mutually-contaminated-distributions framework, where the mixing proportions are **scalars** and both
pools share the same class-conditionals. Our contamination — unpublished positives hiding in U — is
x-dependent by its nature, so the algebra does not start. We have a motivating analogy, not a
guarantee, and the paper has to say so.

## The risk that outranks the model choice

Each U pair shares one substrate with a real positive, so the shared side cancels and everything
learnable lives in the replaced side. What is learnable there may be co-occurrence and fashion rather
than reactivity.

**checked** against Chen et al.'s DUD-E study: their decisive control replaced the decoys with the
*actives of other targets* — real actives, same provenance, chosen to be plausible rather than random.
Our sampler already has that property, since replacements are real published substrates. So the
critique is narrower than it first appears: what distinguishes us from their control is the **anchor**,
one side held fixed from a specific positive pair.

The test is therefore narrow and cheap: build a second U with **both** sides replaced, preserving the
joint size structure so the size shortcut we already measured at 0.640 does not return, and compare.
Designing it carelessly would measure the old mistake and look like the new one.

Sieg et al. supply the mechanism that makes this worth doing: matching the mean and standard deviation
of a property defeats a linear model but not a tree ensemble, which splits on intervals. We match a
**decile**, which is coarser still, and read it with a 2,048-bit fingerprint. Matching creates a new
bias whenever it is coarser than the model's resolution.

## Open, and not ours to settle alone

- Target domain for DETM: our constructed U, or the inventory, against ADR 008 item 4.
- The 1,121 negatives: train on them, or hold them out as instruments. **checked**: the warning that
  biased negatives are harmful rests on Li, Liu & Ng (2010), and both papers that cite it state it in
  one sentence without its condition. The paper proves nothing — it is a problem formulation plus an
  experiment on text corpora — and its mechanism is **coverage** of the negative space, not
  misweighting. Where train and test negatives agree, negatives are merely unnecessary, by 1–2 % and
  not significant, and the authors themselves allow that negatives may still be useful. Harm was
  *smaller* for negatives resembling the training ones, so it cannot be cited against our
  near-boundary rows at all. What it does describe is our shape: 1,121 rows covering a vanishing,
  doubly-selected corner while the plate is mostly uncovered negative space. That argues against
  training on them as if they sampled the negative class; it argues nothing against using them to
  validate or to diagnose.

  Their best use is as a falsification instrument. The probabilistic-gap assumption — propensity
  monotone in the class probability, the weakest named assumption in this literature — says poor
  results do not get published. These were. **checked** in He et al.: their framework has no slot for
  a labelled negative at all, so in their own terms the sharp counterexample is the published
  low-yield *positive*, and the cost of the assumption failing is unquantified; their Theorem 2 holds
  exactly under the assumption and gives nothing otherwise. Their Theorem 1 is still worth citing: it
  is the formal statement that under such a propensity the observed ordering preserves the true one,
  which is the proposition our ranking-not-calibration argument rests on. Better to name it as a
  falsifiable assumption we can argue against than to leave it implicit.
- The random arm of the prospective plate is the only unbiased sample we will ever have. Sizing it to
  calibrate a propensity model, and not only to compare hit rates, is a design decision that must be
  taken before the model is frozen.

## Papers read

Thirty, in `~/Downloads/papers/`. The ones that changed a conclusion: DETM (Liu et al., JASA 2025);
PUbN (Hsieh, Niu & Sugiyama, ICML 2019); Beyond SCAR (Bekker, Robberechts & Davis); the Bekker &
Davis survey (Mach. Learn. 2020); aaPU §3.3 (Xu et al.); Sakai, Niu & Sugiyama on PU-AUC; Wei et al.
(Sci. China Inf. Sci. 2022) and Wang et al. (ICLR 2026) on proxy AUC; Charoenphakdee, Lee & Sugiyama
(ICML 2019); Li, Liu & Ng (EMNLP 2010); He et al. on the probabilistic gap; Chen et al. (PLoS ONE
2019) and Sieg, Flachsenberg & Rarey (JCIM 2019) on dataset bias; Janes & Pepe (Biometrika 2009) and
Owen (Ann. Appl. Stat. 2007) on the two axes' statistics.

One is missing and nothing replaces it: nobody has written the method we need. That is the finding,
not a gap in the reading.

## What I would write in an ADR, if we agree

1. **No published method fits**, and the reason is structural: the bias is on both sides and the
   unlabeled set is constructed. Say so rather than adopt something that assumes one half away.
2. **π is unidentifiable here**; we rank and do not calibrate, and we give the chemical reason.
3. **The model family is the π-free, mixture-free one**: a weighted P-vs-U ranker, with Sakai's own
   bound on what the dropped term costs at small π; and DETM as the principled version, once its
   target domain is settled.
4. **The 1,121 negatives are instruments, not training rows**, and the reason is coverage, not the
   one-line warning the literature repeats.
5. **The AUC robustness theorem is an analogy for us, not a guarantee** — the contamination it covers
   is x-independent and ours is not.
6. **The sampler check comes before the model choice.** If the anchored U is what the score reads,
   the rest is academic.
