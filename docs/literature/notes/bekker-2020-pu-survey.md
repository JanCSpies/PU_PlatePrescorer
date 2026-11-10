> **Provenance.** Summary written by a Claude agent on 2026-10-09 from the full PDF (arXiv 1811.04820v3), focused on this project's questions. Page references point to that PDF. Checked only where `docs/design/modeling.md` or ADR 003 relies on it; read the paper before citing a detail.
>
> **Reference.** Bekker & Davis, "Learning from positive and unlabeled data: a survey", *Mach. Learn.* 2020, 109, 719–760; arXiv 1811.04820. Sections marked "[not in this PDF]" are from memory and unverified.

# SAR PU learning and the unknown JACS SI: reading notes

## Important correction on paper 1

`1811.04820v3.pdf` is **not** "Beyond the Selected Completely At Random Assumption..." (Bekker, Robberechts, Davis). It is:

> **Jessa Bekker and Jesse Davis, "Learning From Positive and Unlabeled Data: A Survey"**, KU Leuven. Received 12 Nov 2018, accepted 1 Mar 2020; arXiv v3, 18 May 2020 (p. 1). 47 pages, 40 pages of text plus references.

The SAR paper is the survey's reference [4]: Bekker, Robberechts and Davis, ECML PKDD 2019 (p. 41). The survey covers SAR only briefly, on pp. 4–5, 8–10, 21, 23–25, 27–28, 32 and 39. Everything below comes from the survey and has page numbers. Where I add something from the SAR paper itself (which I have not read here), I mark it **[not in this PDF]**. Someone should check those parts against the real ECML paper before we rely on them.

---

## Paper 1: Bekker and Davis, PU survey (read in full)

### 1. Setup and notation (pp. 3–6)

- Each example is a triplet (x, y, s): y is the true class and s is "was it labeled" (p. 4). Labeled implies positive: Pr(y=1 | s=1) = 1.
- **Propensity score** (p. 4): e(x) = Pr(s=1 | y=1, x), the probability that a positive example with attributes x gets labeled.
- **Label frequency**: c = Pr(s=1 | y=1) = E_x[e(x)] (p. 5).
- **Labeled distribution = biased positive distribution** (Eq. 2, pp. 4–5): f_l(x) = (e(x)/c) · f_+(x). The *relative* propensity e(x)/c is the density ratio between "positives we observe" and "all positives". This is the key identity for our setting.
- **Two scenarios** (pp. 5–6):
  - *Single-training-set*: one i.i.d. dataset, and a fraction of its positives is labeled.
  - *Case-control*: two independent datasets. One is positive-only. The other, the unlabeled set, is an i.i.d. sample of the true population, x | s=0 ~ f (Eq. 4).
  - **Our setup (literature positives + inventory pairs as U) is case-control.** The survey's derivations mostly assume single-training-set (p. 6) and give case-control variants where they differ. Using the wrong scenario changes the formulas.
- **Class prior vs label frequency** (p. 6, Eqs. 5–7). In case-control, α is defined on the unlabeled data, α = Pr(y=1 | s=0), and
  c = Pr(s=1) / (α(1−Pr(s=1)) + Pr(s=1)) and α = ((1−c)/c) · Pr(s=1)/(1−Pr(s=1)).
  If you know one, you know the other.

### 2. The SAR assumption (pp. 9–10)

- **Definition 2 (SAR)**: labeled examples are a biased sample of the positives. The bias "completely depends on the attributes" and is given by e(x) = Pr(s=1 | x, y=1) (p. 9).
- Motivation is labeling bias, for example ad-click position or socioeconomic status affecting doctor visits (p. 9). This matches literature bias: popular building blocks and patent-type chemistry get "labeled" (reported) more often.
- SCAR is the special case e(x) = c for all x (Def. 1, p. 7). Under SCAR, f_l = f_+ (p. 8) and Pr(s=1|x) = c·Pr(y=1|x) (p. 9). Ranking by a "non-traditional" P-vs-U classifier then preserves the true ranking (p. 9). **Under SAR this ranking property is lost.** A naive P-vs-U model partly learns "is this a popular, often-reported substrate" rather than "does this coupling work".
- Key statement (p. 10): "When the labeling mechanism is understood, incorporating it during the learning phase enables learning an unbiased classifier from SAR PU data. However, when it is not known, additional assumptions are needed."
- Limitation (p. 39): SAR "is so general that it essentially always holds. However, it is so general that effective learning in this setting requires making additional assumptions."
- Missing-data analogy (p. 37): SCAR and SAR mirror MCAR and MAR. Formally both are MNAR, because negatives are never labeled.
- **Probabilistic gap (PGPU)** is a special case of SAR (Def. 3, p. 10). Here e(x) is a monotone increasing function of Pr(y=1|x) − Pr(y=0|x), so "easier" positives are labeled more often. It allows reliable positive and negative extraction (pp. 10–11, 17). The only PGPU method is a two-step technique that also assumes separability (pp. 27–28). The authors give a counter-example where PGPU does not hold (p. 39).

### 3. How to learn under SAR

**(a) ERM reweighting with propensity scores** (pp. 21, 23–25). The survey says this is the only class-prior method explored so far under SAR (p. 21). Rewrite the risk so expectations over f_+ become expectations over f_l, weighted by c/e(x):

R(g) = α·E_{f_l}[(c/e(x))·(L+(g(x)) − L−(g(x)))] + E_f[L−(g(x))]   (p. 24)

*Case-control* empirical risk (p. 24), which applies to us:
- every unlabeled example is a **negative** with weight 1/|s=0|
- every labeled example is a **positive** with weight (1/|s=1|)·α·c/e(x) **and** a **negative** with weight −(1/|s=1|)·α·c/e(x).

*Single-training-set* version (p. 24): unlabeled examples are negative with weight 1. Labeled examples are positive with weight 1/e(x) and negative with weight (1 − 1/e(x)).

This was "the first SAR PU learning method [4]" (p. 25). Under SCAR it reduces to known estimators.

Practical points:
- The method gives unbiased estimates of Pr(y=1|x) under SCAR or SAR (p. 28).
- Downside: **negative weights**, which not all learners or implementations accept (p. 28).
- In the case-control form you need α and the ratio c/e(x), not e(x) itself. A propensity known only up to a constant can be fine if α is known or tuned. This is my reading of the p. 24 formula; the paper does not say it.

**(b) EM with propensity attributes** (p. 27, §5.5), quoted in full because the survey says nothing more:
> "Expectation Maximization (EM) can be used for SAR PU data with the additional assumption that the propensity scores only depend on a known subset of the attributes. An EM approach is then used to simultaneously train the classifier and a model for estimating the propensity scores [4]."

So the **propensity attributes** are a *known subset* of the features, and e(x) = e(x_e) depends only on that subset. The survey does not say which attributes they are, and it does not give the E- and M-step equations.

**[not in this PDF, from my knowledge of Bekker et al. 2019; please verify]** SAR-EM alternates two steps. The E-step computes, for each unlabeled example, the expected Pr(y=1 | x, s=0) from the current classifier and propensity model. The M-step refits the classifier on soft labels and the propensity model on the (expected) positives. The propensity model is logistic over x_e. The "labeling mechanism" is s=1 for labeled and s=0 for the expected positives in U. Their experiments use simulated propensities on UCI-type data. Identifiability needs restrictions such as the propensity attributes being a strict, known subset and the classifier and propensity models not being able to absorb each other. Otherwise the product Pr(y=1|x)·e(x) cannot be split into its two factors. They also discuss that the propensity function cannot be learned from labeled data alone, because it also needs the unlabeled positives.

**(c) Smoothness and ensembles as partial relaxations of SCAR** (p. 28): "Rebalancing and class prior incorporation methods are sensitive to the SCAR assumption. Ensemble methods provide more robustness [19, 72]. Alternatively, the smoothness assumption can be leveraged to relax the SCAR assumption [44, 46, 67, 97]." When classes are clearly separated, two-step methods are "more robust against deviations from the SCAR assumption" (p. 27).

### 4. Requirements and identifiability

- **Class prior is not identifiable from PU data alone** (p. 13): a missing label can mean either a low α or a low c. The assumptions for identifiability, from strongest to weakest, are (p. 13): separable classes, a positive subdomain (anchor set), a positive function, and irreducibility. All of them are stated **under SCAR**.
- For SAR, a formal identifiability condition is **not stated** in the survey. It only says that additional assumptions are needed (pp. 10, 39), and the EM route needs "a known subset of the attributes" (p. 27).
- Prior estimators are evaluated almost only on SCAR data. One sentence on SAR: "recently it was shown that KM2 results in more accurate classifier performance than TIcE on SAR PU data [4]" (p. 32).
- Prior from outside the data (p. 31): domain knowledge, a small fully labeled dataset from the same domain, or treating α as a hyperparameter tuned with a PU metric. **For us, plate or HTE results on a sample of inventory pairs would give α directly.**

### 5. Class prior estimators and PU metrics (all SCAR-based)

- Elkan–Noto c-estimate: the average predicted Pr(s=1|x) on held-out labeled data, which needs calibration (p. 28). Also rank pruning (p. 28), max g(x) under a positive subdomain (p. 29), partial matching with penalised divergences (p. 29), TIcE (pp. 29–30), ROC-based (pp. 30–31), and kernel embeddings KM1/KM2 (p. 31). TIcE and KM2 were the most accurate on SCAR, and TIcE is much faster (p. 32).
- PU metrics (p. 14): the Lee–Liu criterion r²/Pr(ŷ=1) and contingency-table reconstruction from α (pp. 14–15). **Both rely on SCAR** (p. 38). The authors list "metrics that work for other labeling mechanisms" as open work (p. 40). So under SAR we cannot trust PU-F1 or Lee–Liu on literature positives. We need a small labeled (plate) test set.

### 6. Limitations stated by the authors

- There are few SAR methods: ERM reweighting is the only class-prior-based one (p. 21), plus SAR-EM (p. 27).
- SAR needs extra assumptions to be learnable (p. 39).
- Evaluation is mostly on semi-synthetic PU data. The authors call for real-world PU benchmarks and simulated violations of the assumptions (p. 40).
- PU evaluation metrics assume SCAR (pp. 38, 40).
- Negative weights in ERM (p. 28).

### 7. Practical recipe for our setting (derived from the survey)

1. **Frame it as case-control** (pp. 5–6). U = all acid × amine pairs in the inventory, which is our target population. P = literature HATU/DIPEA/DMF couplings.
2. **A subtle problem.** Case-control SAR still assumes the labeled positives come from f_+ of the *same* population that U samples (Eqs. 2 and 4). Our literature positives mostly use substrates **outside** our inventory. So part of the P–U difference is covariate shift between substrate spaces, not only labeling bias within one population. Eq. 2 only makes sense where f_+ > 0 inside the inventory space. We should either restrict or reweight P to the inventory's region of chemical space, or model the shift explicitly, for example with a domain classifier. This is my inference; the survey does not discuss it.
3. **Pick propensity attributes x_e** (p. 27). These are features that drive "being reported" but ideally do **not** drive "the coupling works". This separation matters for identifiability (see the [not in this PDF] note above).
4. **Estimate e(x_e)**, either by SAR-EM (p. 27) or with an external model of reporting frequency. Then train with **case-control ERM weights** (p. 24) using a learner that accepts negative weights. Alternatively, clip the weights, or use the soft-label duplication trick (pp. 22–23), which needs only non-negative weights but is SCAR-only as stated.
5. **Get α from outside** (p. 31), ideally from a small random plate of inventory pairs. Otherwise treat it as a hyperparameter.
6. **Evaluate on a labeled random sample** of inventory pairs (plate results), not on held-out literature positives (pp. 38, 40).

### 8. Concrete propensity-attribute candidates for us

Good candidates are features that plausibly affect *whether a coupling gets reported* more than *whether it works*:

- **Substrate literature frequency**: the number of reactions or documents per acid and per amine in ORDerly, USPTO or ChemLex, for example log-count per building block or for the pair.
- **Patent count / medicinal chemistry prevalence**: the fraction of a substrate's occurrences coming from patents vs journals.
- **Commercial availability and price**: in the catalogue or not, vendor count, $/g, lead time.
- **First-appearance year** of the substrate, as a proxy for age and exposure. The **year** of the reporting document helps if we model per-document propensity.
- **Source/assignee type**, such as pharma patent vs academic paper, and **dataset of origin** (USPTO vs ORDerly vs ChemLex), since each has its own reporting filter.
- **Building-block "drug-likeness" class** (e.g. aryl/heteroaryl acids, piperazines) as a coarse popularity bucket.

Cautions:
- Some of these correlate with reactivity. Steric bulk and exotic functional groups are both rarer *and* lower-yielding. If x_e overlaps strongly with the classifier's features, SAR-EM cannot tell propensity from class probability. Keep x_e to substrate-identity or usage statistics, not structural descriptors.
- Success bias ("only successful reactions get published") is **not** a SAR feature bias. It is the PU mechanism itself, since negatives are never labeled. SAR handles *which positives* get reported, not the fact that failures are missing.
- Yield-dependent reporting (low-yield successes are under-reported) would be PGPU-like (p. 10). That fits Def. 3: easier, higher-yield couplings are labeled more often.

---

