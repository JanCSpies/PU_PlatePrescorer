> **Provenance.** Agent summary, not yet checked by Luka. Written 2026-10-28 from the PDF, which was read in that session. Locators are pages given by Luka's session (the paper has 13 pages); where none was given they are `[n]`. Read the paper before citing a detail.
>
> **Reference.** Tong Wei, Hai Wang, Weiwei Tu, Yufeng Li. *Robust model selection for positive and unlabeled learning with constraints*. Science China Information Sciences 2022, 65(11), 212101. DOI: 10.1007/s11432-020-3167-1. Received 17 July 2020. Licence of data: `[n]`. Date read: 2026-10-28.

# Wei et al. 2022: robust model selection for PU learning

## What they did

They ask how to choose between PU models when no negative labels exist. They build evaluators computed on P and U only and prove they track the supervised ones.

- **Theorem 1, "Equivalent PU model evaluator"** (p. 5):

      AUC_PN = AUC_PU / (1 - pi)  -  pi / (2 (1 - pi))

  Both coefficients are constants, so `AUC_PU` preserves the ordering of candidate models with respect to `AUC_PN`. The class prior pi is needed to convert the number, not to rank.
- **Second evaluator, IPM with the identity loss:** mean score on P minus mean score on U. Its transform is pure scale: `IPM_PN = IPM_PU / (1 - pi)`.
- **Variance reduction.** They add a control-variate step. A classifier that discriminates P from U is trained, and the density ratio is the control variate. Variance falls by a factor of (1 - rho^2). Section 3.5, "Variance reduction by control variate", p. 5.
- **Finding on the two evaluators:** "using the IPM as the model evaluator typically led to more robust performance than the AUC, therefore we report results of the IPM in following experiments". p. 8, with Table 2.

## What the data is

Benchmark PU datasets, model selection over hyperparameters of one algorithm, class frequency 0.01 in the experiment I took the claim from. Names and sizes: `[n]`.

## What carries over to this project

- `AUC_PU` is exactly the quantity ADR 018 item 3 describes in words. ADR 019 adopts it as PAUC with this paper as the primary source.
- The control-variate step is the reason ADR 019 puts a bootstrap interval on the information axis: the authors thought the variance large enough to need treatment.
- Reporting IPM beside AUC is taken from here (ADR 019 item 5).

## What does not carry over

1. **The unlabeled set must follow the true marginal.** The proof rewrites the unlabeled density as the pi-weighted mixture of the class-conditionals. Our U is built with substrate marginals matched to P's (ADR 013 item 1, ADR 015).
2. **P must be a completely-at-random sample of the positives (SCAR).** The proof says it where Term I is estimated through P "since every x in P is obtained from the underlying distribution of positive samples completely at random". Reporting bias violates this.
3. They select hyperparameters of one algorithm. We compare different descriptors, each with its own fitted model. IPM depends on the score scale, AUC does not, which is why PAUC stays primary for us.

So the ordering guarantee is not inherited. The statistic is kept, the guarantee is not.

## Numbers we may cite

- Theorem 1 as written above, p. 5.
- The IPM transform `IPM_PN = IPM_PU / (1 - pi)`, page `[n]`.
- Variance reduction factor (1 - rho^2), Section 3.5, p. 5.
- The quotation on IPM over AUC, p. 8.
- Do not cite any experimental accuracy figure from this note. I have none with a locator.
