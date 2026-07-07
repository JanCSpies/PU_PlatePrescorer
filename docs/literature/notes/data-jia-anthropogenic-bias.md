# Anthropogenic biases in chemical reaction data
- **Citation:** Jia, X.; Lynch, A.; Huang, Y.; Danielson, M.; Lang'at, I.; Milder, A.; Ruby, A. E.; Wang, H.; Friedler, S. A.; Norquist, A. J.; Schrier, J. "Anthropogenic biases in chemical reaction data hinder exploratory inorganic synthesis." *Nature* **573**, 251–255 (2019). DOI: 10.1038/s41586-019-1540-5
- **Type:** analysis (bias quantification + small ML demonstration)
- **File:** literature/s41586-019-1540-5.pdf
- **TL;DR:** Reported reaction data are a biased, popularity-driven positive sample whose reagent/condition choices are *uncorrelated with reaction success* — the empirical justification for treating literature reactions as "positives" and everything else as "unlabeled" (PU), not as negatives.

## Problem & contribution
Chemical reaction datasets are planned by humans and therefore inherit cognitive/social biases. Using amine-templated metal-oxide crystal structures in the CSD (5,010 structures, 415 unique amines) plus 557 unpublished notebook reactions and **548 newly run *randomly generated* experiments**, the authors show the popularity of a reagent is not driven by its intrinsic reactivity, and that models trained on random (unbiased) data beat models trained on larger human-selected data.

## Key findings (bias mechanisms / data availability)
- **Power-law / preferential attachment:** the top **17 % of amines appear in 79 %** of reported structures; **Gini = 0.654** (comparable to global wealth inequality). log-proportion vs log-rank is a single power law → "rich get richer" (Matthew effect), not a reactivity signal.
- **Popularity ⟂ success:** across 548 random experiments, popular and "not-popular" amines succeed at statistically indistinguishable rates (at-least-one-success: 63 % popular vs 75 % not-popular, p = 0.26). *"Any single reaction … is equally likely to be successful regardless of the popularity of the amine."*
- **Condition bias:** human reactions cluster at a single literature-precedent pH 8 and small amine amounts (one-variable-at-a-time), whereas outcomes are actually good over a wide pH range.
- **Confirmation + loss-aversion bias:** publication favours positive results; "the missing 'failures' are essential knowledge." When the human-trained and random-trained models disagree (23/110 cases), the human model *always* predicts failure — measurable pessimism.
- **ML consequence:** random-data model reached **AUC 0.80 / 79 % acc / 85 % positive recall** vs human-data model **AUC 0.64 / 69 % / 46 %**, despite 20 % fewer training points. Missing/obscured features (e.g. polar surface area, ring counts) are implicitly hidden by the biased sampling.
- **Recommendation:** enumerate all options, sample from physically-motivated distributions (structured randomness) — i.e. cover the combinatorial space, exactly what our UNLABELED pool does.

## Relevance to our pipeline
- This is the **core citation motivating PU learning**. It empirically proves the two assumptions we need: (1) reported ⇒ reactive (positives are trustworthy); (2) *unreported ≠ unreactive* — absence is a popularity artefact, so unreported combinations must be UNLABELED, never negative.
- Justifies our design of a **full combinatorial BH substrate space as the unlabeled pool**: unbiased coverage is what makes models generalize (their random set is the analogue of our combinatorial enumeration).
- Predicts a specific failure mode of any literature-trained classifier: **over-conservatism** (false negatives on feasible-but-unpopular substrates). Our JnJ test set is deliberately OOD/unpopular substrates → we should expect and correct for this pessimism (a PU class-prior / non-negative risk correction directly targets it).
- The Gini/Pareto diagnostic is a **ready-made bias audit** we can run on each candidate literature source (amine and aryl-halide occurrence curves) to report *how biased* each source is — a quantitative axis for the source benchmark.

## Leakage / licensing caveats
- Domain is inorganic hydrothermal crystallization, not BH C–N — the *mechanism generalizes*, the numbers do not; cite for the bias argument, not for BH-specific rates.
- Their held-out test used amines absent from both training sets (substrate-disjoint split) — same principle we must enforce vs JnJ.
- Data/code in SI; no reuse restriction relevant to us.

## Open questions for us
- What is the Gini/Pareto coefficient of amine and aryl-halide usage in *our* USPTO/ORD C–N positives? (Run the same occurrence analysis to quantify each source's bias.)
- Can we get a "randomized" analogue for BH? The HTE sets (Cernak, Merck, Dreher) are the closest thing to Jia's unbiased random reactions — use them to calibrate/validate the PU class prior.

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Raccuglia, P. et al. "Machine-learning-assisted materials discovery using failed experiments." *Nature* **533**, 73–76 (2016). DOI: 10.1038/nature17439 — the canonical "failed experiments help" precedent.
- Keserű, Soós, Kappe. "Anthropogenic reaction parameters — the missing link…" *Chem. Soc. Rev.* **43**, 5387 (2014). DOI: 10.1039/C4CS00081A 🔒
- Brown & Boström. "Analysis of past and present synthetic methodologies in medicinal chemistry." *J. Med. Chem.* **59**, 4443 (2016). DOI: 10.1021/acs.jmedchem.5b01409 🔒 — reaction-choice bias in med-chem (BH is one of the over-used reactions).
- Sheridan. "Time-split cross-validation…" *J. Chem. Inf. Model.* **53**, 783 (2013). DOI: 10.1021/ci400084k — split methodology.
