> **Provenance.** Agent summary, **not yet checked by Luka**. Read in full on 2026-10-30 from the
> **freely readable** arXiv copy (`arxiv.org/pdf/1809.03207`, v4, 28 Jun 2019), extracted with
> `pdftotext`; Luka has not read it himself. Not "open access" in the licence sense: that copy carries
> arXiv's non-exclusive distribution licence 1.0, which permits distribution by arXiv and **not**
> reuse, so the verbatim quotations below stand on ordinary scholarly quotation with attribution and
> not on a reuse grant. Page locators come from that extraction and follow the printed page numbers of
> the LNCS preprint; where no locator was taken they are `[n]`. Check a detail against the paper before
> citing it.
>
> **Before any of this is published.** data-guardian measured the quotations on 2026-10-30: 32 verbatim
> passages, about 889 words, roughly 20 % of this note, against a paper of about 20 pages — ordinary
> quotation and not a licence problem for an internal note. But `docs/` may ship in the fresh
> repository of ADR 005, and if it does, the two long passages — the 103-word identifiability
> condition (§5.1, p. 7) and the 76-word SAR-to-SCAR reduction — should be paraphrased first. No
> finding in this note depends on their exact wording.
>
> **Reference.** Jessa Bekker, Pieter Robberechts, Jesse Davis. *Beyond the Selected Completely At
> Random Assumption for Learning from Positive and Unlabeled Data*. ECML PKDD 2019, LNCS 11907;
> arXiv:1809.03207v4. No DOI is printed on the arXiv copy. Licence: arXiv non-exclusive
> distribution licence 1.0 (not a CC licence); the conference version is Springer LNCS. Date read:
> 2026-10-30.

# Bekker, Robberechts and Davis 2019: labelling bias and the propensity score

## What they did

They drop the assumption that labelled positives are a uniform random subset of all positives, and
build the machinery for the case where the labelling probability is a function of the instance.

**1. The two assumptions, verbatim.**

SCAR, stated twice. In the preliminaries (p. 2): *"it is common to make the Selected Completely at
Random (SCAR) assumption, which assumes that the observed positive examples are a random subset of
the complete set of positive examples. Selecting a positive example is therefore independent of the
example's attributes Pr(s = 1|y = 1, x) = Pr(s = 1|y = 1). The probability for selecting a positive
example to be labeled is known as the label frequency c = Pr(s = 1|y = 1)."* As a taxonomy entry
(p. 3): *"SCAR Selected Completely At Random: The labeling mechanism does not depend on the
attributes of the example, nor on the probability of the example being positive: each positive
example has the same probability to be labeled."*

SAR, p. 3: *"SAR Selected At Random: The labeling mechanism depends on the values of the attributes
of the example, but given the attribute values it does not depend on the probability of the example
being positive."*

And the class they exclude, p. 4: *"SNAR Selected Not At Random: All other cases: The labeling
mechanism depends on the real probability of this example being positive, even given the attribute
values."* They add (p. 4) that the class probability *"refers to the probability of an identical
instance to this one being positive"*.

So the whole change is one object: SCAR has a scalar `c = Pr(s = 1|y = 1)`; SAR has a function
`e(x) = Pr(s = 1|y = 1, x)`. Under SCAR the classifier falls straight out of the labelled-vs-rest
model, `Pr(y = 1|x) = Pr(s = 1|x)/c` (p. 2); under SAR that division by a constant is wrong
instance by instance.

**2. The propensity score.** Definition 1, p. 4: *"The propensity score for x, denoted e(x), is the
label assignment probability for positive instances with attributes x, e(x) = Pr(s = 1|y = 1, x)."*
They borrow the term from causal inference and flag the difference (p. 4): their score is
conditioned on the class being positive, so unlike in causal inference the score of a negative
instance is zero and plain inverse-propensity weighting is impossible (p. 4, repeated p. 13).

**The identifiability restriction** is in §5.1, p. 7, and it is **assumed, not proved**. In full:
*"Learning the propensity scores from positive and unlabeled data requires making additional
assumptions: if any arbitrary instance can have any propensity score, then it is impossible to know
if an instance did not get labeled because of a low propensity score or a low class probability.
Therefore, the propensity score needs to depend on fewer attributes than the final classifier [18].
A simple way to accomplish this is to assume that the propensity function only depends on a subset
of the attributes x_e called the propensity attributes: Pr(s = 1|y = 1, x) = Pr(s = 1|y = 1, x_e),
e(x) = e(x_e)."* There is **no numbered theorem or proposition for identifiability**. The argument
is a construction, p. 7: if `x_e` is discrete with finitely many configurations, partition the data
into one stratum per configuration; *"Within a stratum, the propensity score is a constant (i.e.,
all positive examples have the same probability to be labeled) and can thus be determined using
standard SCAR PU learning techniques."* They justify the restriction informally (p. 7): it is
*"trivially true if the labeling mechanism does not have access to all attributes"*, or if the
labeller cannot interpret some attributes, or uses only attributes known to correlate with the
class. The four numbered results in the paper (Propositions 3, 4, 5) are all about the estimator,
not about identifiability. **Numbering caveat:** the body calls the two bounds Proposition 3 and
Proposition 4, while the appendix proves them as "Proposition 1" and "Proposition 2" (pp. 6 vs
18–19). Cite them by name, not by number.

**3. Is the class prior required?** Two routes, and they differ.

- The **SAR-to-SCAR reduction** (§5.1) *does* inherit the prior problem. They say so plainly,
  p. 8: *"Note that, as previously discussed, the SCAR assumption alone is not enough to enable
  learning from PU data, and hence one of the additional assumptions [3, 34, 29, 31, 1] must be
  made."* Those are the identifiability assumptions for the prior listed on p. 3, and the prior
  itself is declared not identifiable without them (p. 3): *"Estimating the label frequency is an
  ill-defined problem because it is not identifiable: the absence of a label can be explained by
  either a small prior probability for the positive class or a low label frequency."* Under SCAR
  the prior and the label frequency are interchangeable, `c = Pr(s = 1)/α` (p. 2).
- Their **recommended method, SAR-EM** (§5.2, pp. 8–9), does **not** take π as an input. It jointly
  fits a classifier `f` and a propensity model `e` by maximising the expected log likelihood; the
  class prior appears nowhere in the E-step or the M-step formulas on p. 8, and the initialisation
  is prior-free, p. 9: *"The classification model is initialized by fitting a balanced model which
  considers the unlabeled examples as negative. The propensity score model is initialized by using
  the classification model to estimate the probabilities of each unlabeled example being
  positive."* The class prior enters their experiments only through the two **baselines** they beat,
  KM2 and TIcE, which are class-prior estimators (p. 11).

So: π is needed for the theoretical reduction, not for the method they actually ship. It is replaced
by the restriction that `e` depends on fewer attributes than `f`.

**4. The estimator.** Empirical risk minimisation with a signed propensity weighting, not inverse
propensity weighting. The insight, p. 4: *"for each labeled example (x_i, s = 1) that has a
propensity score e_i, there are expected to be 1/e_i positive examples, of which 1/e_i − 1 did not
get selected to be labeled"*, formulated (pp. 4–5) as *"learning with negative weights: every
labeled example gets a weight 1/e_i and for every labeled example a negative example is added to
the dataset that gets a negative weight 1 − 1/e_i."* Definition 2, p. 5, writes this as the
propensity-weighted estimator of MAE, MSE or log loss, and the derivation immediately below it
shows it is unbiased (the expectation is taken over `s_i` with the `x_i`, `y_i` held fixed, so the
unbiasedness is with respect to the empirical risk on the n instances held).

Guarantees, p. 6:

- *Propensity-Weighted Estimator Bound* (Prop. 3): with probability 1 − η,
  `|R̂ − R| ≤ sqrt(δ²_max · ln(2/η) / 2n)`, by Hoeffding (appendix, pp. 18–19), *"with δ_max the
  maximum absolute value of cost function δ_y"*.
- *Propensity-Weighted ERM Generalization Error Bound* (Prop. 4): for a finite hypothesis space,
  `R(ŷ_R̂|y) ≤ R̂(ŷ_R̂|e, s) + sqrt(δ²_max · ln(|H|/η) / 2n)`.
- *Propensity-Weighted Estimator Bias* (Prop. 5), when only an estimate `ê` is available:
  `bias = (1/n) Σ y_i (1 − e_i/ê_i)(δ_1(ŷ_i) − δ_0(ŷ_i))`. Two consequences they draw out, p. 6:
  *"From the bias, it follows that the propensity scores only need to be accurate for positive
  examples"*, and *"Underestimated propensity scores are expected to result in a model with a higher
  bias. Lower propensity scores result in learning models that estimate the positive class to be
  more prevalent than it is."*

They also warn off an alternative, p. 7: minimising the expected risk `R̂_exp` is **not** unbiased
and degenerates — *"the hypothesis of always predicting the positive class ∀i : ŷ_i = 1 always has
an expected risk R̂_exp(ŷ|e, s) = 0."*

Cost: logistic regression for both `f` and `e`, with Scikit-Learn defaults (p. 11), so each EM
iteration is two cheap fits. No runtime, iteration count or wall-clock figure is given — **not in
the paper**. Convergence is non-standard, p. 9: *"the likelihood could stop improving before the
propensity score model has converged. Convergence is therefore formulated as convergence of both
the log likelihood and the propensity model"*, measured by the average slope of the least-squares
line through the last n propensity predictions. What it needs that one might not have: a declared
set of propensity attributes `x_e`, strictly smaller than the classifier's features. In their
experiments these are **given** to the method — the Q1 result is phrased *"When the propensity
attributes are known"* (p. 11).

Limitation they state themselves, p. 9: *"Because this approach minimizes log loss, it will work
best if the classes are separable. If the classes are not separable, then the trained classification
model is not expected to be the optimal one for the trained propensity model."* The remedy offered
is to refit the classifier with the obtained propensity scores using the weighted risk estimator.

**5. What goes wrong if you use SCAR methods under SAR labelling.** They give no closed-form
expression for the damage; they give the measured gap, and a mechanism. The mechanism, p. 11:
*"the naive approach sometimes outperforms the SCAR approaches. This can be explained by the SCAR
methods' goal of predicting the correct ratio of the instance space as positive. When it picks the
wrong subspace to get to this ratio, it results in both false positives and false negatives, where
the more conservative naive approach would only give the false negatives."* The size, from Fig. 1
(p. 12, four propensity attributes): the worst case is **diabetes**, where the SCAR method KM2
collapses to **ROC-AUC 0.500** against **0.659** for SAR-EM and 0.669 fully supervised, with MSE
0.888 against 0.138; **20 Newsgroups** is next, KM2 **0.603** against SAR-EM **0.891** and 0.951
supervised. Statistically (p. 11): the Friedman test on average ROC-AUC ranks over the eight
datasets rejects equality at p < 0.001 *"regardless of the number of propensity attributes used"*,
and by Nemenyi post-hoc SAR-EM is better than naive, KM2 and TIcE at p < 0.01.

The direction is a mis-shaped decision region, not a uniform mis-calibration: a SCAR method forced
to label the right *fraction* of the space picks the wrong *part* of it. For a spy-threshold method
that is the relevant failure, because a spy threshold is exactly a device for choosing how much of
the space to call reliably negative.

**6. Their experiments.** Eight UCI/LIBSVM-style datasets (Table 1, p. 10) spanning text, images
and tabular data: 20 Newsgroups (3,979 instances, 200 attributes, Pr(y=1) = 0.55), Adult (48,842 /
14 / 0.24), Breast Cancer (683 / 9 / 0.35), Cover Type (581,012 / 54 / 0.49), Diabetes (99,492 /
127 / 0.11), Image Segmentation (2,310 / 18 / 0.43), Mushroom (8,124 / 111 / 0.48), Splice (3,175 /
60 / 0.52). Everything was reduced to continuous attributes scaled to [−1, 1] so logistic
regression could be used (p. 10).

The bias is **simulated, not measured** (§6.2, pp. 10–11). They add four artificial binary
propensity attributes: each dataset is clustered into five groups on its attribute values, and each
group gets a random distribution over the propensity attribute values, *"a scenario where examples
that are in the same cluster have the same prior probability of belonging to the positive class.
However, which examples are labeled depend on the propensity attributes."* The propensity score is
the geometric mean of per-attribute factors `p_low` / `p_high`, with *"all artificial propensity
attributes x_e attributing equally to it"*, and *"In our experiments the propensity scores were
between 0.2 and 0.8"* (p. 11). Protocol: five 80/20 train–test splits, five labelings each, 25 runs
averaged (p. 11).

Baselines (p. 11): SCAR-KM2 (Ramaswamy et al.), SCAR-TIcE (Bekker and Davis), Naive (all unlabeled
treated as negative), and fully Supervised as an upper bound.

What accounting for the bias bought, ROC-AUC, SAR-EM against the better SCAR baseline KM2, and
against supervised, from Fig. 1b (p. 12): diabetes 0.659 / 0.500 / 0.669; 20ng 0.891 / 0.603 /
0.951; mushroom 1.000 / 0.924 / 1.000; cover type 0.826 / 0.760 / 0.827; image segmentation 0.970 /
0.930 / 0.985; splice 0.886 / 0.855 / 0.924; adult 0.903 / 0.887 / 0.907; breast cancer 0.983 /
0.971 / 0.994. TIcE is far worse throughout (0.504–0.950).

**The honest counterweight.** Ignoring the bias entirely was harmless, or better than modelling it,
in several cases (Fig. 1b, p. 12). Naive beats SAR-EM on **breast cancer** (0.992 vs 0.983) and
ties it on **mushroom** (1.000 vs 1.000); naive also beats the SCAR baseline KM2 on six of the eight
datasets — breast cancer 0.992 vs 0.971, mushroom 1.000 vs 0.924, 20ng 0.859 vs 0.603, cover type
0.771 vs 0.760, diabetes 0.632 vs 0.500, image segmentation 0.958 vs 0.930 — losing only on adult
(0.875 vs 0.887) and splice (0.862 vs 0.855). And where SAR-EM wins over KM2, the win is often
small: adult +0.016, breast cancer +0.012, splice +0.031. So the large wins are concentrated in two
datasets; on four of eight the whole question is worth under 0.05 AUC, and the simplest thing —
treat U as negative — is competitive.

**Q2, propensity recovery** (p. 13): *"Except for the mushroom dataset, the EM method always obtains
very accurate propensity score estimates with MSEs below 0.1"*, most near zero; mushroom is 0.123
(Fig. 2, p. 13). **Q3** (p. 13): *"we observe no correlation between the number of propensity
attributes and the MSE and ROC-AUC of the classification model, nor the MSE of the propensity score
estimates."*

**7. Code and licence.** Implementation at `https://dtai.cs.kuleuven.be/software/sar` (footnote 7,
p. 11). Baselines: KM2 at `web.eecs.umich.edu/~cscott/code/kernel_MPE.zip`, TIcE at
`dtai.cs.kuleuven.be/software/tice/` (footnotes 8–9, p. 11). **No licence is stated for the code** —
not in the paper. The arXiv copy carries arXiv's non-exclusive distribution licence 1.0, which is
not a reuse licence.

## What the data is

Eight public benchmark datasets, sizes and class priors as in Table 1 above (p. 10). No chemistry,
no pairs, no reaction data. The PU labelling is entirely synthetic: the true labels are known, and
`s` is drawn from a propensity function the authors designed, with `e` in [0.2, 0.8] (p. 11). There
is no dataset in the paper where the real labelling mechanism was observed or measured.

## What carries over to this project

- **The diagnosis, in their words.** Our labelling variable `s` is "somebody published this pair".
  `Pr(s = 1|y = 1, x)` plainly depends on `x` — novel substrates, scope tables, fashionable motifs
  — so our mechanism is at best SAR and definitely not SCAR (p. 3). This note is the citation ADR
  019 was missing when it named `e(x)`: the paper that defines `e(x)` for PU data is this one,
  Definition 1, p. 4.
- **One caution on which class we are in.** SAR requires that *"given the attribute values it does
  not depend on the probability of the example being positive"* (p. 3). Chemists publish what
  worked, and a higher-yielding pair is more likely to reach a scope table than a marginal one with
  the same fingerprint. If publication depends on the true success probability beyond what our
  features capture, we are in **SNAR**, p. 4, which this paper explicitly sets aside and offers
  nothing for. That is a real possibility for us and should not be glossed over. The related-work
  paragraph on p. 13 describes a near-miss: He et al. assume the propensity falls with the
  probabilistic gap `ΔPr(x) = Pr(y=1|x) − Pr(y=0|x)`, i.e. harder-to-label positives are less
  likely to be labelled, *"a specific case of SAR"* — a shape close to our "easy, clean, publishable
  couplings get reported" story, and a pointer worth following.
- **Only positives' propensities have to be right** (Prop. 5, p. 6). All 11,396 of our labelled rows
  are positives, so the object we would have to model is exactly the one the bias term depends on.
  That is a genuinely favourable structural fact.
- **No class prior needed by SAR-EM** (§5.2, pp. 8–9). This is the answer to the question that
  decides whether the paper is useful to us: their method does not want π, and ADR 013 avoided
  needing π. On that axis the paper is compatible with our choice rather than a reversal of it.
- **The warning against our planned spy threshold.** A spy threshold is a SCAR device: it assumes
  the hidden spies are exchangeable with the unlabelled positives, which is true only if labelling
  was attribute-independent. Under SAR the spies are drawn from our published positives, so they
  carry the publication bias with them and the threshold they induce describes "where published
  chemistry scores", not "where positives score". The failure they measure (p. 11) is precisely a
  wrongly chosen subspace, so the spy threshold would cut the wrong part of the space while looking
  calibrated.
- **Could their identifiability restriction hold for us?** Possibly, and in a specific form. Their
  condition is that `e` depends on a strict subset `x_e` of the classifier's features, justified
  (p. 7) by *"the labeling mechanism does not have access to all attributes"*. Our classifier reads
  Morgan fingerprints around the reacting centre. The natural `x_e` is **corpus metadata that is not
  in the fingerprint**: how often each substrate already appears in the literature, publication
  year, how crowded the substrate's neighbourhood is, heavy-atom decile. Those are variables we hold
  outside the feature block, which is the shape §5.1 wants. The price is an assumption we would have
  to write down and defend: that the fingerprint bits around the reacting centre do **not** drive
  publication once novelty and corpus frequency are controlled for. That is uncomfortable, because
  motif fashion *is* fingerprint chemistry. The restriction is therefore plausible only in a
  weakened form — the fingerprint drives publication mostly through quantities we can measure
  separately — and the honest version is to keep the fingerprint out of `e` deliberately and say so.

## What does not carry over

- **The generative story is the wrong way round for us, and this is the main point.** Their setting
  is the single-training-set scenario, p. 2: one sample of triples `(x, y, s)`, where `s` is drawn
  per instance with probability `y_i e_i`. One sample exists first; labelling happens on it. We have
  the reverse. Publication happened in the world, and *then* we built 102,200 unlabelled pairs
  around the published ones by swapping one side within a heavy-atom decile. Our unlabelled rows
  were therefore constructed conditional on `s`, and had probability zero of being labelled by
  construction. The Bernoulli draw that Definition 2's unbiasedness rests on (p. 5) is not the
  process that produced our `s`, so the unbiasedness derivation does not transfer as an inference
  about nature. Note the narrow sense in which it survives: that derivation fixes `x_i` and `y_i`
  and averages only over `s_i`, so the weighted estimator is unbiased for the empirical risk **on
  whatever finite set you hold** — but for us that is the risk under our own sampler's marginal,
  which is not the plate's marginal and not nature's. In short: the arithmetic of the weighting is
  portable; the claim that the result is an unbiased estimate of real-world risk is not.
- **Our π is a parameter of our sampler, not of nature.** The paper treats `α = Pr(y = 1)` as a
  property of the domain (p. 2) and spends a page on why it is hard to identify (p. 3). We fixed the
  mixture when we chose to draw 102,200 unlabelled pairs with per-side marginals matched to P on
  purpose. So the quantity their reduction argues about is not a quantity we are trying to recover.
  By the same token, any `e` their EM fitted on our data would absorb our sampler's design — the
  decile matching, the 1 : 9 ratio — and report it as publication propensity. That is a concrete way
  the method could produce a confident, meaningless answer.
- **Nothing in the paper assumes U is unbiased in the features, and nothing handles U being biased
  on purpose.** To be exact: the paper states no unbiasedness assumption on U anywhere — U is simply
  the unlabelled remainder of one sample, so the question never arises. That silence is the problem.
  Ours is drawn by us with the per-side marginals matched to P, which is a deliberate feature-level
  bias in U, and the paper offers no theory for it.
- **No pair structure.** Throughout, `x` is a single attribute vector for a single entity (p. 2).
  Our instance is an acid **with** an amine, and our construction matched marginals per side, so a
  propensity attribute would have to be defined on the pair and would interact with which side was
  swapped. There is no analogue in the paper, and the stratification argument of §5.1 (p. 7) does
  not obviously survive being applied to one side of a pair.
- **Their propensities are moderate; ours would be tiny.** `e` in [0.2, 0.8] (p. 11). Publication
  propensity over a virtual substrate space is orders of magnitude smaller, and the weights are
  `1/e_i`. The paper offers no experiment, and no remark, at small `e`; the bound in Prop. 3 (p. 6)
  is stated purely in terms of `δ_max` and `n`, and how the range of the *weighted* costs grows as
  `e → 0` is **not discussed in the paper**. Before trusting any number from this method on our
  data, that gap has to be closed ourselves.
- **Separability.** Their EM *"will work best if the classes are separable"* (p. 9). Our U is matched
  to P on heavy-atom decile precisely so that the classes are not separable on that axis (and the
  anchor ablation in the Chen note exists to check for leaks of exactly that kind). We sit in the
  case they flag as the one where the EM's classifier is not the optimal one for its own propensity
  model.
- **Our 1,121 measured sub-threshold pairs have no place in their framework.** Their PU setting has
  no known negatives at all — they lean on that fact to distinguish themselves from the
  recommender-systems work (p. 13). Using real negatives as an instrument, per ADR 024, is outside
  anything in this paper.

**Verdict.** The framework does not apply to a constructed unlabelled set as a tool for recovering
the real labelling mechanism. The propensity-weighting identity is per-instance arithmetic and would
run on our table, but every inferential claim in the paper — unbiasedness with respect to the domain
risk, identifiability of `e` from P and U, the stratification argument — presupposes that one sample
was drawn from the domain and labelling then acted on it. We built U after and around the labelling,
so their EM would estimate our sampler, not publication bias. What the paper gives us is the
vocabulary (`e(x)`, SAR, SNAR), the proof that the bias only depends on the propensities of
positives, the fact that a prior-free SAR method exists, and a measured warning about SCAR methods
picking the wrong subspace. What it does not give us is a usable estimator for our data.

## Numbers we may cite

- SCAR as a constant `c = Pr(s = 1|y = 1)`; `Pr(y = 1|x) = Pr(s = 1|x)/c`; `c = Pr(s = 1)/α`. p. 2.
- SCAR / SAR / SNAR definitions. pp. 3–4.
- Propensity score, Definition 1, `e(x) = Pr(s = 1|y = 1, x)`. p. 4.
- Weighting scheme: labelled example weight `1/e_i`, plus an added negative at weight `1 − 1/e_i`;
  Definition 2 and its unbiasedness. pp. 4–5.
- Propensity-Weighted Estimator Bound (Prop. 3): `sqrt(δ²_max ln(2/η) / 2n)`. p. 6; proof pp. 18–19
  (numbered Proposition 1 there).
- Propensity-Weighted ERM Generalization Error Bound (Prop. 4): `sqrt(δ²_max ln(|H|/η) / 2n)`. p. 6;
  proof p. 18 ff. (numbered Proposition 2 there).
- Propensity-Weighted Estimator Bias (Prop. 5), and "propensity scores only need to be accurate for
  positive examples"; underestimated `e` biases towards over-predicting the positive class. p. 6.
- Expected-risk alternative is biased and degenerate: `∀i : ŷ_i = 1` gives `R̂_exp = 0`. p. 7.
- Identifiability restriction `e(x) = e(x_e)` with `x_e` a subset of the attributes; assumed, not
  proved; stratification argument. p. 7. The SAR-to-SCAR reduction still needs a class-prior
  assumption. p. 8.
- SAR-EM E-step and M-step; prior-free initialisation; convergence of both likelihood and propensity
  model; works best if classes are separable. pp. 8–9.
- Datasets: 20 Newsgroups 3,979 / 200 / 0.55; Adult 48,842 / 14 / 0.24; Breast Cancer 683 / 9 / 0.35;
  Cover Type 581,012 / 54 / 0.49; Diabetes 99,492 / 127 / 0.11; Image Segm. 2,310 / 18 / 0.43;
  Mushroom 8,124 / 111 / 0.48; Splice 3,175 / 60 / 0.52. Table 1, p. 10.
- Simulated bias: 4 artificial binary propensity attributes, 5 clusters per dataset, `e` in
  [0.2, 0.8], 5 splits × 5 labelings = 25 runs, 80/20. pp. 10–11.
- ROC-AUC, SAR-EM / SCAR-KM2 / Supervised, 4 propensity attributes, Fig. 1b, p. 12:
  diabetes 0.659 / 0.500 / 0.669; 20ng 0.891 / 0.603 / 0.951; mushroom 1.000 / 0.924 / 1.000;
  cover type 0.826 / 0.760 / 0.827; image segm. 0.970 / 0.930 / 0.985; splice 0.886 / 0.855 / 0.924;
  adult 0.903 / 0.887 / 0.907; breast cancer 0.983 / 0.971 / 0.994. SCAR-TIcE 0.504–0.950.
- MSE, diabetes: SAR-EM 0.138 against SCAR-KM2 0.888, supervised 0.096. Fig. 1a, p. 12.
- Naive (U treated as negative) beats SCAR-KM2 on 6 of 8 datasets, and beats SAR-EM on breast cancer
  (0.992 vs 0.983). Fig. 1b, p. 12; mechanism explained p. 11.
- Friedman test p < 0.001; Nemenyi, SAR-EM better than naive, KM2 and TIcE at p < 0.01. p. 11.
- Propensity recovery: MSE below 0.1 on seven of eight datasets, 0.123 on mushroom. Fig. 2, p. 13.
- No correlation between the number of propensity attributes and any reported metric. p. 13.
- Code: `dtai.cs.kuleuven.be/software/sar`; no code licence stated. Footnote 7, p. 11.
