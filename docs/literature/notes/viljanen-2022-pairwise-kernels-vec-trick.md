> **Provenance.** Agent summary, **not yet checked by Luka**. The PDF was read in full in the working session of 2026-10-29 from the open-access arXiv copy; Luka supplied the file but has not read it himself. Page locators come from that reading (the printed page numbers of the preprint), and where none was taken they are `[n]`. Two things in the brief needed correcting and are flagged in place: the author list, and the identification of "per-bit product" with the Kronecker kernel. Check a detail against the paper before citing it.
>
> **Reference.** Markus Viljanen, Antti Airola, Tapio Pahikkala. *Generalized vec trick for fast learning of pairwise kernel models*. arXiv:2009.01054v2 [cs.LG], 4 February 2022. **The PDF is a preprint, not an accepted version:** the header reads "Noname manuscript No. (will be inserted by the editor)" and "Received: date / Accepted: date" (p. 1), so it gives **no journal, no year of publication, no volume, no pages and no DOI**, and it states **no licence**. The IEEE TNNLS paper is a *different, earlier* paper by two of these authors, cited here in the reference list: Airola A, Pahikkala T (2018) *Fast Kronecker product kernel methods via generalized vec trick*, IEEE Transactions on Neural Networks and Learning Systems 29(8):3374–3387 (p. 33) — that one introduced the GVT for the Kronecker kernel alone; this preprint extends it to all the reviewed kernels. Date read: 2026-10-29.
>
> **Published version, from a second source.** This preprint appeared as *Machine Learning*
> **111(2):543–573, 2022**. That detail is **not** in the PDF read here: it comes from the
> bibliography of arXiv:2510.14419 (p. 28), the IC-index paper, which cites this work as
> "Viljanen et al. 2022" — and cites it for the result that additively separable models reach a
> competitive C-index **without** capturing the interaction effect, which is this paper's own
> "surprising" finding about the linear kernel (pp. 31–32). Verify the volume and pages against the
> journal before citing them.

# Viljanen, Airola and Pahikkala: pairwise kernels and the generalized vec trick

## What they did

A review of pairwise kernels plus one algorithmic result. They set out the pairwise learning problem — each input is a pair `x = (d, t)` of a "drug" `d` and a "target" `t`, and the label is a property of the pair (p. 2, p. 6) — then:

- **Define four prediction settings** by whether each member of the test pair was observed in training, and prescribe a different data split for each (p. 6, Table 1 p. 7).
- **Review nine pairwise kernels** and show all of them can be written as sums of (permuted) Kronecker products of a per-side kernel (Table 3 p. 10, Table 4 p. 11, Corollary 1 p. 17).
- **Extend the generalized vec trick (GVT)** from the Kronecker kernel to all of them, giving matrix-vector products in `O(nm + nq)` instead of `O(n²)` (Theorem 1 p. 10, p. 19).
- **Compare the kernels experimentally** on four biological interaction data sets, in all four settings, with kernel ridge regression solved by the minimum-residual method with early stopping, 9-fold cross-validation per setting and a nested 75/25 inner split (pp. 23–24, Figs 4–7).

### Question 1 — the four settings, verbatim

The definitions (p. 6, around Figure 2), quoted:

> 1. `d ∈ Dobs` and `t ∈ Tobs`: prediction for known drugs and targets
> 2. `d ∈ Dobs` and `t ∉ Tobs`: prediction for novel targets
> 3. `d ∉ Dobs` and `t ∈ Tobs`: prediction for novel drugs
> 4. `d ∉ Dobs` and `t ∉ Tobs`: prediction for novel drugs and targets

The prescribed validation method, Table 1 "Training and test set split in different settings" (p. 7):

| Setting | Validation method, as the table gives it |
|---|---|
| 1 | "Split samples `Zobs = Ztrain ∪ Ztest`" |
| 2 | "Split targets `Tobs = Ttrain ∪ Ttest`", then a pair goes to `Ztrain` if `ti ∈ Ttrain` and to `Ztest` if `ti ∈ Ttest` |
| 3 | "Split drugs `Dobs = Dtrain ∪ Dtest`", then a pair goes to `Ztrain` if `di ∈ Dtrain` and to `Ztest` if `di ∈ Dtest` |
| 4 | "Split both targets `Tobs = Ttrain ∪ Ttest` and drugs `Dobs = Dtrain ∪ Dtest`", then `(xi, yi) ∈ Ztrain` if `ti ∈ Ttrain` **and** `di ∈ Dtrain`; `∈ Ztest` if `ti ∈ Ttest` **and** `di ∈ Dtest`; `∈ Zignored` otherwise |

**Setting 4 precisely:** both sides are split independently; only pairs whose *both* members are on the train side train the model, only pairs whose *both* members are on the test side are tested, and **every mixed pair — train acid with test amine, or the reverse — is discarded into `Zignored`, used neither for training nor for testing** (Table 1, p. 7). They "always generate separate test sets for each of the four settings" and implement it "either with a single split to training and test sets, or by using cross-validation with repeated splits" (p. 7); in the experiments this is 9-fold cross-validation performed according to the setting, with the training fold split again 75/25 into inner-training and validation by the same setting rule for the early-stopping hyperparameter (p. 24).

They also record the standing experimental finding: "Setting 1 is usually the easiest to predict accurately, followed by Settings 2 and 3, whereas making accurate predictions in Setting 4 tends to be very challenging" (p. 7).

### Question 2 — the Kronecker product pairwise kernel

**Definition** (p. 8): given a drug kernel `kD : D × D → R` and a target kernel `kT : T × T → R`,

> "The Kronecker product kernel is then defined as the product of the drug and target kernels `kD,T((d, t), (d̄, t̄)) = kD(d, d̄) kT(t, t̄)`."

In operator form the whole-domain kernel operator is `D ⊗ T`, and the kernel matrix for two samples is `R(d, t)(D ⊗ T)R(d̄, t̄)ᵀ`, where `R` is the sampling/indexing operator picking the observed pairs (pp. 8–9, Corollary 1 p. 17).

**Why it models the interaction, in their terms.** The implicit feature map of the pair is the tensor product of the two sides' feature maps, `φ(d,t) = φD(d) ⊗ φT(t)` (Table 4 p. 11), and §4.4 (pp. 12–13):

> "The feature vector is given as a tensor product of the drug and target feature vectors `x^{d,t} = x^d ⊗ x^t`. The resulting feature vector consists of simply all the pairwise interactions `(x_i^d x_j^t)_{i=1...r, j=1...s}`. These are same as the pairwise interactions in the polynomial kernel with self-interactions excluded. The Kronecker product kernel can be motivated as the simplest kernel that models actual pairwise interactions in drug and target features."

Set against the linear kernel, whose feature map is the plain concatenation `(φD(d), φT(t))` and whose predictor is `f(d, t) = ⟨w_d, x^d⟩ + ⟨w_t, x^t⟩` — "In the resulting model, some drugs and targets simply have more interactions than others, but there are no interactions between drug and target features" (§4.1, p. 12). They also note the Kronecker kernel "is an universal kernel, if the drug and target kernels are universal (e.g. Gaussian)" (p. 13).

**Dimension.** With `φD(d) ∈ R^r` and `φT(t) ∈ R^s` (p. 11), the Kronecker feature vector has one coordinate per `(i, j)` pair, so its dimension is `r · s`. For two 2,048-bit fingerprints: 2,048 × 2,048 = **4,194,304 ≈ 4.19 million** features. The brief's "about 4.2 million" (from the working session, not from Luka) is **confirmed** — it is the exact product `r · s`, not an approximation of something else. (The 2nd-degree polynomial kernel on the concatenation is larger: `(r+s)² = 4,096² ≈ 16.8 million`, since it adds the acid-acid and amine-amine self-interaction blocks, §4.2 p. 12.)

### Question 3 — the statement we would cite

The paper makes the claim about additive/linear combinations **explicitly and twice**, and it is theirs to cite. It does **not**, however, claim the Kronecker form is *necessary* — see the caveat below.

The claim, as the "non-linearity assumption" (pp. 3–4):

> "The second, what we refer to as **non-linearity assumption**, states another property inherent in pairwise learning problems, which is that the functions to be learned are usually not linear combinations of functions depending only of `d` or `t`. The opposite case, where the function can be expressed as `f(d, t) = fd(d) + ft(t)` for some `fd` and `ft`, would indicate that `f(d1, t1) > f(d2, t1) ⟹ f(d1, t2) > f(d2, t2)` for all drugs and targets, that is, if drug `d1` is more effective than drug `d2` against target `t1`, then drug `d1` is also more effective than drug `d2` against target `t2`. In other words, there would always be a single drug that would be the best choice for all targets (and vice versa)."

And the impossibility result, §4.1 (p. 12), on the pairwise *linear* kernel — the exact estimator class of our 0.478 probe:

> "This implies a global ordering of drugs, where drugs and targets are completely decoupled. ... In the resulting model, some drugs and targets simply have more interactions than others, but there are no interactions between drug and target features. The artificial chessboard problem illustrated in Figure 1 is an example of a data set, that is impossible to model using the pairwise linear kernel."

The chessboard is the XOR of the two sides' parities, "unlearnable with linear methods according to the classical result by Minsky and Papert (1969)" (p. 4, Fig. 1 p. 4).

**The honest caveat, which belongs next to any citation of the above.** The paper's own conclusion is that the Kronecker form is *sufficient and natural*, not necessary, and that in practice the linear kernel often does not fall over (Discussion, pp. 31–32):

> "An interesting observation that can be made from the experimental results is that in many cases the linear pairwise kernel produces results that are competitive with those obtained using the non-linear kernels. This is surprising in the sense that the kernel allows only expressing functions of the form `f(d, t) = fd(d) + ft(t)` that score the drugs and targets separately without truly modeling interactions between them."

and, directly relevant to 4.19 million features (p. 32):

> "The nonlinear component also may easily 'drown' with high dimensional data, as the number of interaction terms increases quickly as its function."

So: a cite for *"a linear model on concatenated features cannot represent the interaction, only a per-side sum"* is available and strong. A cite for *"therefore the Kronecker/bilinear form is necessary"* is **not** in this paper — the Poly2D kernel also carries the interaction block, and their own experiments repeatedly find linear "surprisingly good" (pp. 25, 26, 27). Our 0.478 is better read through their Figure 1 argument than through their experiments, which do not reproduce a collapse that severe.

### Question 4 — the other pairwise kernels

From Table 3 (p. 10), Table 4 (p. 11) and §§4.1–4.8 (pp. 11–14). `D ≠ T` means the kernel admits **heterogeneous** domains, i.e. the two sides need not be the same kind of object — our case, acid vs amine.

| Kernel | What it assumes / when preferred | `D ≠ T`? | Usable in Setting 4? |
|---|---|---|---|
| **Linear** | `kD + kT`; concatenated features. Assumes no interaction at all, `f = fd(d) + ft(t)`. Preferred when data are few — they find it best at `N ≤ 1000` (p. 29). | yes | yes, but cannot express interaction (p. 12) |
| **Poly2D** | `(kD + kT)²`; tensor square of the concatenation. Carries acid-acid, acid-amine and amine-amine blocks; the cross block "model[s] actual interactions of drug and target features" (p. 12). Preferred when self-interaction terms also help. | yes | yes; empirically tied with Kronecker throughout (pp. 26, 27, 29) |
| **Gaussian** | Gaussian on the concatenation; factorises into a product of per-side Gaussians, hence "a special case of the Kronecker product kernel" (p. 12). | yes | yes |
| **Kronecker** | `kD · kT`; "the simplest kernel that models actual pairwise interactions" (p. 13). Universal if the per-side kernels are. The default when the two sides are different kinds of object. | yes | **yes — and it is the one they recommend by weight of results** |
| **Symmetric Kronecker** | `½(x^d ⊗ x^{d'} + x^{d'} ⊗ x^d)`; assumes the relation is symmetric by definition. Preferred when it genuinely is, e.g. protein-protein (p. 13). | no, needs `D = T` | yes where the domain is homogeneous; best or tied best on their homogeneous sets (pp. 25, 29) |
| **Anti-symmetric Kronecker** | The antisymmetric half; assumes `f(d, d') = −f(d', d)`. | no | yes, same caveat |
| **Ranking** | Feature map `x^d − x^{d'}`; "can model ranking representable relations, i.e. relations constructed from some utility function `h` such that `f(d, d') = h(d) − h(d')`" (p. 13) — an antisymmetric *linear* kernel. Preferred for preference/ordering tasks. | no | yes structurally, but expresses only a global ranking |
| **MLPK** (metric learning) | The ranking kernel squared; `f(d, d') = (x^d − x^{d'})ᵀ M (x^d − x^{d'})` — "models interaction of a pair in the terms of how similar the drug and the target in the pair are" (p. 14). Preferred when the label really is a similarity/distance. | no | yes on homogeneous data; "by far the best" on one feature set (p. 25) but **worst** in Settings 2/3/4 on another (p. 29) |
| **Cartesian** | `kD(d,d̄)δ(t=t̄) + δ(d=d̄)kT(t,t̄)`; learns separate per-drug and per-target parameter blocks. Only for Setting 1. | yes | **no** |

The Setting-4 exclusions, quoted. On Cartesian (p. 11):

> "Further, the Cartesian kernel is designed to be used in Setting 1 only, as it does not allow generalization to such drugs and targets that are not included in the training data."

and (p. 14):

> "In this sense the learned model includes pairwise interactions, but it does not utilize information between similar interactions in different pairs and cannot generalize to drugs or targets that have not been seen in the training set."

Borne out: "The cartesian kernel is not much better than random on the task" (p. 26).

On matrix factorisation, the quote asked for (p. 6):

> "For example, Setting 1 can be solved even without drug or target features using matrix factorization methods (Basilico and Hofmann, 2004). However, the latent representations learned by matrix factorization methods do not generalize to drugs and targets outside the training set (Settings 2-4). The pairwise kernel learning approach considered in this work is applicable in all of the four settings."

And on heterogeneity (p. 11): "Table 4 highlights an important limitation that applies to some of the kernels. These require homogeneous domains, i.e. they assume both objects in the pair belong to the same domain `D = T`." For us, acid and amine are different domains, which **rules out symmetric, anti-symmetric, ranking and MLPK** on their own terms, independently of the setting. What is left for us in Setting 4 is: linear, Poly2D, Gaussian, Kronecker.

### Question 5 — cost

The baseline: "Solving this system with a method that computes `K` requires at least `O(n²)` time and memory, which is not practical in many pairwise learning problems, where `n` can be in the range of 10⁵ or more" (p. 8). The GVT result, Theorem 1 (p. 10):

> "the operation `p ← R(d, t)(D ⊗ T)R(d̄, t̄)ᵀ a` can be carried out in `O(min(q̄n + m̄n, mn̄ + qn̄))` time using a sparse Kronecker product multiplication algorithm known as the generalized vec-trick (GVT)."

For the training kernel matrix this simplifies to **`O(qn + mn)` against `O(n²)`** (p. 19), with `n` the number of pairs, `m` the number of unique objects on one side and `q` on the other, and the gain comes from the pairwise data assumption `n >> m + q` (p. 3). Memory likewise: "The naive method requires significantly more memory because it stores the full `O(n²)` pairwise kernel matrix whereas GVT only stores the `O(m²)` drug and `O(q²)` kernel matrices" (p. 29). After Corollary 1 (p. 17) the same `O(min(q̄n + m̄n, mn̄ + qn̄))` holds for *all* the kernels in the table, each extra Kronecker summand costing only a constant factor — "The Kronecker kernel is fastest of these because it has only one term and the MLPK slowest because it has 10 such terms" (p. 29).

**Our scale, worked from those expressions.** `n = 11,396 + 102,200 = 113,596` pairs; `m + q ≤ 11,345` distinct substrates (the paper needs the per-side counts, acids `m` and amines `q`, which our brief does not split — fill in before quoting a figure).

- Naive: `n² ≈ 1.29 × 10¹⁰` kernel entries; at 8 bytes that is **≈ 103 GB** for one kernel matrix. Out of the question, and their own naive runs were stopped at 16 GiB (p. 29).
- GVT: `(m + q)·n ≈ 11,345 × 113,596 ≈ 1.29 × 10⁹` operations per matrix-vector product, a **factor ≈ 10⁴ saving**, with per-side kernel matrices costing `O(m²) + O(q²)` — at most `11,345² × 8 bytes ≈ 1.0 GB` if one side held all substrates, and about 260 MB total for a roughly even 5,700/5,700 split. Times the modest iteration counts they report (Fig. 7, p. 28: well under 300 minimum-residual iterations, and *fewest* in Setting 4).
- **Verdict: feasible, comfortably, and well inside what they demonstrate.** Their largest task is kernel filling with 8,803,089 pairs over 2,967 + 2,967 objects (Table 5, p. 20), run as a scaling curve from 10³ to about 10⁶ samples (Fig. 7, p. 28), with a 1,024,000-sample Falkon comparison at 2,048 basis vectors (p. 30). Our `n ≈ 1.1 × 10⁵` sits an order of magnitude below their largest, but our `m + q ≈ 11,345` is about twice theirs, so the `(m+q)·n` product is the number to check, not `n` alone. Their other sets: Heterodimer 5,497 pairs / 1,526 + 1,526 objects, 0.2% dense; Metz 93,356 pairs / 156 + 1,421; Merget 167,995 pairs / 2,967 + 226 (Table 5, p. 20) — Metz and Merget bracket our `n` directly.
- The cost that is *not* cheap is the feature dimension: 4.19 million interaction terms never get materialised (that is the point of the kernel), but see their drowning warning (p. 32).

### Question 6 — implementation

Yes. "**RLScore**", and specifically its `CGKronRLS` learner: "The method `CGKronRLS` in the RLScore software package includes an user friendly implementation of GVT (Pahikkala and Airola, 2016)" (p. 23). Code availability (p. 33):

> "The GVT algorithm for fast training of pairwise kernel methods is made publicly available as part of RLScore package under open access license at https://github.com/aatapa/RLScore."

**Licence: not named.** The PDF says only "under open access license"; it does not give MIT, BSD or anything else, so the licence has to be read off the repository before we could depend on it. The reference is Pahikkala T, Airola A (2016) *RLScore: Regularized least-squares learners*, Journal of Machine Learning Research 17:1–5 (p. 33). They also wrote a pairwise-kernel class against **Falkon** (Rudi et al. 2017; Meanti et al. 2020) as a Cython C extension for the comparison (p. 30). Data links given: `http://staff.cs.utu.fi/~aatapa/data/DrugTarget/` for Metz, `https://github.com/aalto-ics-kepaco/pairwiseMKL` for Merget and kernel filling (pp. 32–33). Nothing installed or run.

## What the data is

Four data sets, all biological interaction prediction, Table 5 (p. 20):

| Set | Pairs | "Drugs" | "Targets" | Homogeneous | Density |
|---|---|---|---|---|---|
| Heterodimer | 5,497 | 1,526 | 1,526 | yes | 0.2% |
| Metz | 93,356 | 156 | 1,421 | no | 42% |
| Merget | 167,995 | 2,967 | 226 | no | 25% |
| Kernel filling | 8,803,089 | 2,967 | 2,967 | yes | 100% |

- **Heterodimer** (pp. 20–21): yeast protein pairs, 152 positive and 5,345 negative examples, labels from CYC2008 and WI-PHI. Features are three binary protein vectors — domain (2,554 bits), phylogenetic profile (768 bits), subcellular localisation (83 bits) — and **the per-side kernel is the Tanimoto (MinMax) kernel on those binary vectors**, written out on p. 21 as `kd(v, v̄) = Σ min(vᵢ, v̄ᵢ) / Σ max(vᵢ, v̄ᵢ)`. This is the closest thing in the paper to our setup.
- **Metz** (p. 21): 156 kinase inhibitors × 1,421 targets, real-valued `Ki` binarised at `Ki < 28.18 nM` into 2,798 interacting and 90,558 non-interacting pairs. Drug side is 2D Tanimoto fingerprint similarity, target side normalised Smith-Waterman; similarity-matrix rows are used as feature vectors under a linear or Gaussian kernel.
- **Merget** (p. 22): 2,967 drugs × 226 kinases, 167,995 binding values, 10 fingerprint Tanimoto drug kernels and 9 protein kernels.
- **Kernel filling** (pp. 22–23): a synthetic scalability task — predict the entries of one drug kernel matrix from another. Used for the timing and memory curves.

All labels are **observed** quantities: curated complexes with explicit negatives, or measured affinities thresholded. Metric throughout is AUC (Figs 4–7).

## What carries over to this project

**The four settings and the Setting-4 split are the frame for our whole validation story.** Their Setting 4 is exactly our hardest case — an acid and an amine neither of which appeared in training — and Table 1 (p. 7) gives the split rule to implement: split acids and amines independently, train only on pairs with both members train-side, test only on pairs with both members test-side, **discard the mixed pairs**. That last part is the piece easy to get wrong, and it is the paper's own prescription, with 9-fold CV and a nested 75/25 inner validation split made by the same rule (p. 24). Their standing observation that Setting 4 is "very challenging" (p. 7) and their curves showing Setting 4 lowest throughout (Figs 4–7) also set an honest expectation: our prospective plate lives in their hardest regime, and the literature does not promise much there. Reporting our numbers split by these four settings would make our results directly comparable to a decade of this field's work.

**On the candidate fix — "per-bit product of the two sides".** Partly right, and the gap matters.

- If "per-bit product" means the **elementwise (Hadamard) product** of the two 2,048-bit fingerprints — 2,048 features, `x_i^{acid} · x_i^{amine}` for the *same* bit `i` — then this is **not** the Kronecker kernel. It is only the **diagonal** of it. The Kronecker feature map is the **outer** product, `x^d ⊗ x^t`, "all the pairwise interactions `(x_i^d x_j^t)_{i=1...r, j=1...s}`" (p. 13), i.e. every bit of the acid against every bit of the amine, 4,194,304 terms. The Hadamard version keeps 2,048 of those 4.19 million and throws away all cross-bit terms `i ≠ j`. Worse, for Morgan-style fingerprints a shared bit index means "both molecules contain the same substructure", so the Hadamard product reads co-occurrence of identical substructures, not "this acid substructure meets that amine substructure" — which is the chemistry we want. Note also that for us the two sides are *different* folded fingerprints over different chemistry; bit `i` on the acid side and bit `i` on the amine side are only aligned by the hash, not by meaning.
- If "per-bit product" means **all products of one side's bits with the other side's bits**, then yes: that is **exactly** the Kronecker product pairwise kernel's feature map (Table 4 p. 11, §4.4 p. 13), the field has had it since Ben-Hur and Noble 2005 / Basilico and Hofmann 2004 / Oyama and Manning 2004 (p. 2), and this paper is the fast-algorithm and kernel-taxonomy survey for it.

So: the instinct is the right one and the field's canonical answer to our 0.478 is the Kronecker pairwise kernel — but write it as the **outer** product, not the elementwise one.

**What the kernel form gives that naive materialisation would not.**

1. **The 4.19-million-dimensional vector is never built.** `kD,T((d,t),(d̄,t̄)) = kD(d,d̄)·kT(t,t̄)` (p. 8) — one acid-side kernel value times one amine-side kernel value. Two 2,048-bit Tanimotos, and we have the pair similarity. Naive materialisation would mean 113,596 × 4,194,304 explicit products.
2. **Non-linear per-side kernels come for free, and universality with them.** We can put a **Tanimoto kernel** on each side, which is what they do on binary fingerprints (p. 21), rather than being stuck with bit products; with universal per-side kernels (e.g. Gaussian) the Kronecker kernel is itself universal (p. 13), so the hypothesis class is not limited to bilinear-in-the-bits functions.
3. **`O(nm + nq)` instead of `O(n²)`** per iteration, and `O(m²) + O(q²)` instead of `O(n²)` memory (Theorem 1 p. 10, p. 19, p. 29) — the difference between about 1 GB and about 103 GB at our scale.
4. **A ready implementation**: `CGKronRLS` in RLScore (p. 23, p. 33), with the Setting 1–4 protocol already the way that group evaluates.
5. **Named alternatives to try alongside** rather than inventing: Poly2D, which adds the per-side self-interaction blocks and matched Kronecker everywhere in their experiments (pp. 26, 27, 29), and the plain linear kernel as the honest baseline — which is the one we already measured.

**One warning to carry over with it.** Their Discussion (p. 32): "The nonlinear component also may easily 'drown' with high dimensional data, as the number of interaction terms increases quickly as its function", and from Figure 7 the non-linear kernels only beat the linear one once the sample is large enough (`N > 1000` in their task, p. 29). With 2,048-bit sides and 113,596 pairs we are exactly in the regime where this needs checking, not assuming. Folding the fingerprints to fewer bits per side, or using a Tanimoto base kernel instead of a linear one, are the levers the paper hands us.

## What does not carry over

- **Their labels are observed; ours are not.** Heterodimer has curated positives *and* explicitly constructed negatives from the same databases (p. 20); Metz and Merget have measured `Ki`/binding values, binarised at a threshold (pp. 21–22). Ours is positive-unlabeled with a *constructed* unlabeled set. Nothing in this paper addresses PU learning, class-prior estimation, or what a kernel ridge regression fitted with unlabeled-as-negative actually estimates. Their AUCs are therefore not comparable to our PU discrimination numbers, and their kernel ranking was established under a loss we are not minimising.
- **No analogue of reporting bias.** The paper never discusses why a pair is present in the data. Their missingness is assay coverage; ours is what chemists chose to publish. Their Setting-4 split protects against substrate leakage, not against a model learning our construction of U — that worry stays ours, and the control for it is the anchor ablation (ADR 013, ADR 015), not anything here.
- **Regression / binary classification, kernel ridge regression, AUC.** They "focus on kernel ridge regression, as it is a widely used method that admits a closed form solution and simplifies the following considerations" (p. 7), with squared loss. They note the shortcuts also apply to kernel logistic regression or SVMs (p. 31), but they do not demonstrate it, and they do not touch PAUC or any top-of-ranking metric, which is what a plate hit rate actually cares about.
- **Half their kernel zoo is unavailable to us.** Symmetric, anti-symmetric, ranking and MLPK all require `D = T` (p. 11). Acid and amine are different domains, so these are out — including MLPK, which was their single best kernel on one feature set (p. 25). Cartesian is out for Setting 4 by construction (p. 11, p. 14). We are left with linear, Poly2D, Gaussian and Kronecker.
- **A kernel needs a valid similarity *between pairs*, and whether ours is one is an open question.** A pairwise kernel in this framework must be a positive semidefinite function on `(D × T) × (D × T)` (p. 7), and the Kronecker form specifically requires the **product** `kD(d,d̄)·kT(t,t̄)` (p. 8). Our nearest-neighbour probe uses the **minimum** of the two sides' Tanimoto similarities. The minimum is not a product, so it is not the Kronecker kernel, and it cannot be written as a sum of Kronecker products in any form Corollary 1 (p. 17) covers — which means **the GVT shortcut would not apply to it even if it were a valid kernel**. Whether `min(kD, kT)` is positive semidefinite at all is **not addressed anywhere in this paper** and I am not resolving it here; it needs either a proof, a counterexample, or an eigenvalue check on our own Gram matrix before the min-Tanimoto probe could be called a kernel. Flagged as open. (The Tanimoto kernel *per side* is something they use directly, p. 21 — that part is safe.)
- **Their own results undercut the strong version of our motivation.** They expected the non-linear kernels to dominate and repeatedly did not find it: "The linear kernel offers surprisingly good results, not very far from optimal" (p. 26), similarly on Merget (p. 27), and the Discussion calls the result "surprising" (pp. 31–32). Our 0.478 is a far more severe failure than anything they report for a linear pairwise kernel, which is itself a signal worth explaining — a linear pairwise kernel that cannot model interaction should still land near the per-side marginal signal, not below chance. The gap between their "surprisingly good" linear kernel and our 0.478 is ours to account for; this paper does not account for it.

## Numbers we may cite

- Kronecker feature-space dimension: `r · s`; for two 2,048-bit fingerprints **4,194,304 ≈ 4.19 million** (Table 4 p. 11, §4.4 p. 13 — the product is my arithmetic from their definition).
- Poly2D feature-space dimension for the same inputs: `(r+s)² = 4,096² ≈ 16.8 million` (§4.2 p. 12, my arithmetic).
- GVT matrix-vector product: `O(min(q̄n + m̄n, mn̄ + qn̄))`; for the training matrix `O(qn + mn)` against `O(n²)` naive (Theorem 1 p. 10, p. 19).
- Memory: GVT stores `O(m²) + O(q²)`, naive stores `O(n²)`; in their kernel-filling task with `n ≈ 0.5q²` this is `O_naive(q⁴)` vs `O_GVT(q²)` (p. 29).
- Observed speed gap: "At a time when the standard method ran out of memory, the training was taking over an hour whereas GVT completed in a second" (p. 29); naive runs stopped at 16 GiB.
- Data-set sizes: Heterodimer 5,497 pairs (152 positive, 5,345 negative) / 1,526 + 1,526 / 0.2% dense; Metz 93,356 pairs (2,798 interacting, 90,558 not) / 156 + 1,421 / 42%; Merget 167,995 / 2,967 + 226 / 25%; kernel filling 8,803,089 / 2,967 + 2,967 / 100% (Table 5 p. 20, pp. 20–22).
- Scaling curves run from 10³ to about 10⁶ samples (Fig. 7 p. 28); Falkon comparison at 1,024,000 samples and 2,048 basis vectors, which "already consumes 16GiB memory" (p. 30).
- Iteration counts to the optimal early-stopped model: under about 300, **fewest in Setting 4**, most in Setting 1 (Fig. 7 p. 28, text p. 28).
- Pairwise data assumption: `n >> m + q` (p. 3). Pairwise learning problems where "`n` can be in the range of 10⁵ or more" (p. 8).
- Tanimoto (MinMax) kernel on binary features, as they define it: `kd(v, v̄) = Σᵢ min(vᵢ, v̄ᵢ) / Σᵢ max(vᵢ, v̄ᵢ)` (p. 21).
- Our scale against their expressions: `n = 113,596`, `m + q ≤ 11,345`; naive `n² ≈ 1.29 × 10¹⁰` entries ≈ 103 GB, GVT `(m+q)·n ≈ 1.29 × 10⁹` operations, per-side kernels ≤ 1.0 GB (my arithmetic from p. 10 and p. 19; the per-side `m`/`q` split still needs filling in).
- RLScore: https://github.com/aatapa/RLScore, learner `CGKronRLS`, "under open access license" — **licence not named in the PDF** (p. 23, p. 33).
