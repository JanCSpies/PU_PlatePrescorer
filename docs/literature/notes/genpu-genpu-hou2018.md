# GenPU: Generative Adversarial Positive-Unlabeled Learning
- **Citation:** Ming Hou, Brahim Chaib-draa, Chao Li, Qibin Zhao. "Generative Adversarial Positive-Unlabeled Learning." IJCAI 2018, pp. 2255–2261. arXiv:1711.08054
- **Type:** method
- **File:** literature/GAN_PU/Generative Adversarial Positive-Unlabeled Learning.pdf
- **TL;DR:** A multi-generator/multi-discriminator GAN that, from positive + unlabeled data only, learns *both* the positive and the negative data distributions and then trains a standard PN classifier on the synthetic (incl. synthetic-negative) samples.

## Problem & contribution
Discriminative PU methods (uPU, nnPU) reweight unlabeled data to build a decision boundary but overfit badly when the positive set is tiny and the classifier is a flexible deep net. GenPU is the first *generative* attack on PU: instead of reweighting, it learns the underlying P and N densities and samples "infinite" realistic P and N examples, so a fully-supervised (PN) deep classifier can then be trained without the small-P overfitting problem.

## Method (key technical detail)
Two generators {Gp, Gn} and three discriminators {Dp, Du, Dn} play a minimax game (objective is a prior-weighted sum, πp·Φ_Gp + πn·Φ_Gn):
- **Gp** (positive generator) is co-supervised by Dp (real P vs fake) and Du (unlabeled vs fake) → pushes p_gp toward pp while staying inside the overall data support p(x).
- **Gn** (negative generator) is supervised by Du and Dn. The crucial trick: for the Gn–Dn game, the sign of the generator loss is **flipped** (Gn *minimizes* Dn's output on its fake samples instead of maximizing it), so Dn sends uniformly negative feedback that *repels* p_gn away from the positive region pp, while Du keeps p_gn inside p(x). Net effect: p_gn is driven toward the negative region pn = the part of the unlabeled support not covered by pp.
- **Theory (Thm 2):** at Nash equilibrium (assuming pp and pn are *well-separated* and the class prior πp is known), the game jointly minimizes JSD(pp‖p_gp) and JSD(p‖πp·p_gp+πn·p_gn) while maximizing JSD(pp‖p_gn), giving p_gp=pp and **p_gn=pn** — i.e. it recovers the true negative distribution. Requires the prior πp as a fixed, predetermined input; λp, λn tuned by grid search on a validation set.
- Extends trivially to semi-supervised (PNU) learning by swapping Gn's loss to a standard GAN loss when a few labeled negatives exist.

## Modality/data it targets (images? tabular? sequences?)
Continuous vectors and **images**. Toy 2-D synthetic data (concentric circles, Gaussian mixtures, half-moons) + MNIST/USPS digit pairs + CelebA faces. Authors explicitly flag that discrete/sequential data (e.g. text tokens) is *hard* for the underlying GAN — a red flag for discrete molecular representations.

## Datasets / benchmarks
2-D synthetic mixtures; MNIST ('3'vs'5', '8'vs'3'); USPS; CelebA (male vs female, 64×64). Baselines: Oracle PN, uPU (du Plessis 2015), nnPU (Kiryo 2017).

## Results & claims
Dramatically more label-efficient than uPU/nnPU with very few positives: on MNIST '3'vs'5' with only Nl=1 labeled positive, GenPU stays at 0.976 accuracy while nnPU collapses to 0.563 and uPU to 0.557. On CelebA, PN classifier trained on GenPU synthetic images: 87.9% vs nnPU 86.8% vs uPU 62.5%. Avoids the unbounded-negative-risk overfitting of uPU.

## Relevance to our BH C–N PU pipeline (be honest about fit)
Conceptually the **most on-point** method for our "synthesize reliable negatives" idea: Gn is literally a generator of *negative* (would-be-infeasible) reactions, learned from positives+unlabeled, that we could sample to augment a downstream feasibility classifier. BUT the fit to our data is **Low–Medium in practice**: (1) GANs on tabular/fingerprint reaction descriptors are notoriously unstable and mode-collapsing; (2) it *requires a known class prior* πp (our feasibility base-rate over the combinatorial BH space is exactly what we don't know); (3) the equilibrium guarantee assumes pp and pn are **well-separated**, which is false for reactivity — feasible and infeasible couplings overlap heavily in descriptor space; (4) a generated "negative" reaction vector need not correspond to any *valid or chemically-meaningful* reaction. Note: the AAAI-2023 "Positive Distribution Pollution" paper (CoVPU) benchmarks GenPU and finds it underperforms because "the quality of the generated data cannot be guaranteed, which introduces uncertainty in classifiers" — a direct warning against betting the pipeline on GAN-synthesized negatives.

## Caveats / limitations / leakage risks
- Needs known πp; sensitive to λ weights.
- Mode collapse / mode oscillation (authors' own Discussion); mitigations = WGAN/IPM instead of JSD, or multiple generators per class.
- Well-separation assumption unrealistic for reactivity.
- No tabular/molecular experiments at all — untested on our modality.
- Leakage risk if positives (literature reactions) leak into the "unlabeled" pool used to train Du/Gn; and generated negatives could accidentally reproduce held-out test reactions.

## Open questions for us
- Could Gn be repurposed as a *sampler over the combinatorial substrate grid* rather than an unconstrained density generator — i.e. generate over a discrete, enumerable, always-valid reaction space so "invalid chemistry" is impossible by construction? That would sidestep GenPU's worst failure mode.
- Can we estimate πp well enough (from TACIT/PAYN priors) to even run GenPU?
- Is the well-separation assumption salvageable via a learned reaction embedding?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Kiryo, Niu, du Plessis, Sugiyama (2017), nnPU, NeurIPS — the risk estimator our whole pipeline leans on. (open, NeurIPS proceedings)
- du Plessis, Niu, Sugiyama (2015), "Convex formulation for learning from PU data," ICML. (open)
- Gulrajani et al. (2017), Improved WGAN training, NeurIPS — if we ever try GenPU, use this backbone. (open, arXiv:1704.00028)
- Jain, White, Radivojac (2016), "Estimating the class prior... from noisy positives and unlabeled data," NeurIPS — for πp estimation. (open)
