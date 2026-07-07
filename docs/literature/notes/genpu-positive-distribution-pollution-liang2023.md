# Positive Distribution Pollution — a Unified View of PU Learning (CoVPU)
- **Citation:** Qianqiao Liang, Mengying Zhu, Yan Wang, Xiuyuan Wang, Wanjia Zhao, Mengyuan Yang, Hua Wei, Bing Han, Xiaolin Zheng. "Positive Distribution Pollution: Rethinking Positive Unlabeled Learning from a Unified Perspective." AAAI 2023, pp. 8737–8745.
- **Type:** method (+ conceptual framework)
- **File:** literature/Diffusion_FlowMatching__PU/Positive Distribution Pollution Rethinking Positive Unlabeled Learning from.pdf
- **TL;DR:** Argues the three big PU pathologies — data imbalance, selection bias, unknown class prior — are all one problem ("positive distribution pollution": you can't estimate p_P from tiny biased labels) and fixes them jointly with CoVPU, a variational model with a **prior-free** loss that mines positives out of the unlabeled pool.

## Problem & contribution
**The unifying insight (this is the key conceptual takeaway of the whole set):** in realistic PU,
- **P1 Data imbalance** — labeled positives ≪ unlabeled, so p_PL (labeled-positive density) is a noisy, unstable estimate of the true p_P.
- **P2 Selection bias** — labeled positives are chosen by biased experts, so p_PL ≠ p_PU (the positives hiding in the unlabeled pool) ≠ p_P.
- **P3 Prior agnostic** — the class prior π_P is unknown, and *without it a mixture has infinitely many valid P/N separations* so p_P is unidentifiable.
All three are the *same* failure: **the estimated positive distribution is "polluted"** (deviates from the true p_P) because you fit it from very little, biased data. Prior work fixes one at a time and usually *assumes π known*; CoVPU is the first to treat all three as one and to be **prior-free**.

## Method (key technical detail)
CoVPU = three collaborating modules:
1. **VPU-Calibrator** (unsupervised): a VAE that separates the unlabeled pool into positive vs negative *components* using **discrete normalizing flows** (planar flows + argmax transform, Gumbel sampling) over a 2-component Gaussian-mixture latent. This *mines the diverse positive-unlabeled samples* to de-pollute the positive-distribution estimate (attacks P1, P2).
2. **VPU-Classifier:** derives an **unbiased + rebalanced risk estimator** (combines the selection-unbiased estimator of Na 2020 with pseudo-oversampling rebalancing of Su 2021), then — because that still needs π — replaces it with a **prior-free variational loss** L_var. Theorem 1: KL(p_P‖p_Φ) = L_var(Φ) − L_var(Φ*), so *minimizing L_var directly minimizes the distance between the polluted and true positive distributions* on the distribution itself rather than on a prior-weighted risk mean. Theorem 2: L_var(Φ)=L_var(Φ*) iff Φ=Φ* → converges to the optimal Bayesian classifier.
3. **VPU-Collaborator:** couples the two — takes the *consensus* (samples flagged positive by both the flow component-assignment and the classifier class-assignment) as trusted pseudo-positive-unlabeled data, re-estimates π from the Bayesian classifier's own expectation (E[Φ(x)]), and iterates (Algorithm 1, MixUp-regularized).

## Modality/data it targets (images? tabular? sequences?)
Both images **and tabular/time-series**: MNIST, CIFAR10, **and BankLoan — a real industrial tabular time-series dataset** (238k loan events, LSTM encoder). This is the only paper of the four validated on genuinely tabular, extremely-imbalanced, selection-biased data — i.e. the setting closest to ours.

## Datasets / benchmarks
MNIST (π_P=0.10, π_PL=0.002), CIFAR10 (π_P≈0.10), **BankLoan (π_P=0.034, π_PL=0.001 — 0.1% labeled positive rate, heavy selection bias)**. Baselines span all four PU families incl. **GenPU** and nnPU/uPU/ImbalancednnPU/PUSB/VAE-PU/APU/VPU/PAN. Metrics: Accuracy, F1.

## Results & claims
CoVPU is best on all three datasets (e.g. BankLoan F1 0.447 vs best baseline 0.422; MNIST F1 0.985 vs 0.955) **without being given π**, while prior-based baselines are shown to be *highly sensitive* to a wrong π (F1 decays fast under ±20% prior deviation, Fig 4). Ablations: removing the Calibrator hurts most (−29% F1) → mining unlabeled positives is the biggest lever. On BankLoan (where feasible/infeasible-analog overlap violates irreducibility) the variational loss only ties the prior-based version — an honest, directly-relevant caveat for us. Crucially it **benchmarks GenPU and finds it weak**: "the quality of the generated data cannot be guaranteed, which introduces uncertainty in classifiers."

## Relevance to our BH C–N PU pipeline (be honest about fit)
**Highest strategic relevance of the four.** (1) The "positive distribution pollution" frame *is our problem*: literature positives (feasible BH reactions) are few, **selection-biased** (chemists publish what works, on popular substrate classes), imbalanced, and our feasibility prior over the combinatorial grid is unknown — exactly P1+P2+P3. This reframes how we should think about the whole benchmark: our real enemy is a biased/polluted positive-distribution estimate, not just "missing negatives." (2) It is validated on **tabular imbalanced selection-biased data (BankLoan)**, our modality. (3) Its **prior-free variational loss** and **self-estimated π** (π = E[Φ(x)]) let us avoid guessing the feasibility base-rate. (4) Its **honest verdict against GenPU** is a strong signal to treat generative negative-synthesis as a stretch goal, not the backbone. Weaknesses for us: CoVPU is a fairly heavy 3-module VAE+flow+collaborator system (nontrivial to reproduce in a hackathon week); the discrete normalizing flow Calibrator is the most fragile part; and it only *ties* prior-based methods when irreducibility fails (which is likely for reactivity, where feasible/infeasible overlap).

## Caveats / limitations / leakage risks
- No public code found in the paper; several derivations deferred to a "longer version."
- Discrete-flow Calibrator is finicky (argmax flows, Gumbel) — reproduction risk.
- Only ties, not beats, prior-based loss when the irreducibility (separability) assumption breaks — a scenario we should *expect* for reactions.
- Selection-bias definition (p_PL≠p_PU) is exactly literature publication bias; good news it's modeled, but their fix assumes we can *mine* the true positives from U — if our unlabeled grid genuinely lacks enough hidden feasible reactions, the Calibrator has nothing to recover.
- Leakage: the Collaborator's pseudo-labeling loop can drift and pull test reactions into "consensus positives" if the split isn't frozen.

## Open questions for us
- Adopt just the **prior-free variational loss** + **self-estimated π (E[Φ])** on top of a simple classifier (drop the heavy Calibrator/Collaborator) — how much of the gain survives? This is the cheapest high-value thing to prototype.
- Model publication bias explicitly as p_PL≠p_PU — does distinguishing "labeled" vs "hidden" feasible reactions help on the JnJ test?
- Since reactivity likely *violates* irreducibility, do we need Zhu&Scott-2023 "MPE beyond irreducibility" rather than this?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Chen, Liu, Wang, Zhao, Wu (2020a), "A Variational Approach for Learning from Positive and Unlabeled Data" (VPU), NeurIPS — the prior-free variational loss CoVPU builds on; likely the *lighter* method we'd actually implement. (open, NeurIPS proceedings)
- Na et al. (2020), "Deep generative PU learning under selection bias" (VAE-PU), CIKM — selection-unbiased risk estimator. doi:10.1145/3340531.3411998 🔒
- Su, Chen, Xu (2021), "PU Learning from Imbalanced Data" (ImbalancednnPU), IJCAI — rebalancing. (open, IJCAI proceedings)
- Hu et al. (2021), "Predictive Adversarial Learning from PU Data" (PAN), AAAI — generative PU without π. (open)
- Hoogeboom et al. (2021), "Argmax Flows and Multinomial Diffusion," NeurIPS — the discrete-flow machinery. (open)
- Bekker & Davis (2018), "Estimating the Class Prior in PU Data via decision trees," AAAI. (open)
