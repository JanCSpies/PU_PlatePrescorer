# Joint Diffusion Sampling via Positive-Unlabeled Guidance (JDS)
- **Citation:** Matt Raymond, Yilun Zhu, Jianxin Zhang, Angela Violi, Clayton Scott. "Joint Diffusion Sampling via Positive-Unlabeled Guidance for Multi-Modal Data." ICML 2025 Workshop on Multi-modal Foundation Models and LLMs for Life Sciences.
- **Type:** method
- **File:** literature/Diffusion_FlowMatching__PU/44_Joint_Diffusion_Sampling_vi.pdf
- **TL;DR:** Sample from a hard-to-get *joint* distribution by combining pretrained per-modality diffusion models and steering them with a **prior-free** PU classifier ("balanced PU posterior"), using tiny amounts of joint data and never needing the class prior π.

## Problem & contribution
In life sciences, multi-modal *joint* datasets are scarce but single-modality foundation models are plentiful. Goal: generate a pair (A,B) from a target joint density f, given pretrained diffusion models for the marginals f1, f2 and only a *small* joint sample. Contributions: (1) reduce joint sampling to PU-guided diffusion; (2) introduce **prior-free PU guidance** that, unlike prior PU-diffusion work (Takahashi 2025; Na 2024), does **not require estimating π** — competitive with oracle PN guidance even at 0.0012% label rate. From Clayton Scott's group (a core PU-learning theory lab) + Violi (chem eng).

## Method (key technical detail)
- **Copula/mixture framing:** treat the *product* of the marginals as the unlabeled distribution and the *joint* as the positive: fu(x)=f1(x1)f2(x2), fp(x)=f(x). Then f1·f2 = π·f + (1−π)·fn for some π and "reject" density fn. Appendix B gives necessary+sufficient conditions (Radon–Nikodym derivative bounded) for this mixture to be well-posed.
- **Prior-free guidance (the reusable insight):** classifier-guided diffusion needs ∇log fp(x;t). Using Ivanov's (DEDPUL) identity, the ratio fp/fu = g/(1−g) where g is the **Balanced PU (BPU) posterior** ("non-traditional classifier" of Elkan–Noto) learnable by *balanced* ERM (weight P and U classes equally, Appendix A). The guidance term ∇log[g/(1−g)] is **invariant to π** — π cancels out of the score correction. So the correction to the independent marginal scores is an additive log-copula-like term learned by PU-CPE, with the small joint set as positives and independently-paired marginal samples as unlabeled.
- Uses DDPM for images/Booleans, **Geo-LDM (E(3)-equivariant latent diffusion) for molecules (QM9)**; g trained via balanced cross-entropy with bootstrap resampling of positives; guidance via backprop through the classifier (with "recurrence steps" for quality).

## Modality/data it targets (images? tabular? sequences?)
Multi-modal continuous/structured: MNIST images, CelebA-HQ faces, **QM9 small molecules (3-D, via Geo-LDM)**, and 1-D Booleans (discrete relaxed to reals). Semi-synthetic "joint" tasks (same-parity digits, same-parity-dipole molecule pairs, digit↔Boolean).

## Datasets / benchmarks
MNIST, CelebA-HQ(256), QM9 (dipole moment parity), MNIST×Boolean. Metrics: MMD / CLIP-MMD / ChemNet-MMD + accuracy of generated pairs. Baselines: fully-supervised PN classifier guidance, unconditional diffusion, and PUDM (Takahashi 2025) at several π values.

## Results & claims
Prior-free PU guidance matches PN (oracle) guidance on accuracy across modalities at extreme label scarcity (0.0012–0.0026% positive rate); slightly worse MMD on many-mode MNIST (25 modes vs 10k samples). For molecules it can even steer *toward the negative* class ("PN(−1)"). Main limitation: test-time cost of backprop-through-classifier guidance; results are on semi-synthetic joints only.

## Relevance to our BH C–N PU pipeline (be honest about fit)
Two things are genuinely useful, but the headline task (multi-modal joint sampling) is **not** our problem:
1. **Prior-free PU guidance / BPU posterior** is the single most transferable idea: our BH feasibility base-rate π over the combinatorial grid is unknown, and a method whose *classifier/guidance is invariant to π* is exactly what we want. The BPU "non-traditional classifier" (train P-vs-U with balanced weights, then recover posterior) is trivially implementable on our fingerprint features and directly avoids the π-estimation error that GenPU/PUDM inherit.
2. **It uses a real molecular generator (Geo-LDM/QM9)** and even demonstrates *guiding toward the negative class* — a proof-of-concept that PU-guided molecular diffusion can bias generation toward "reject"/negative regions, i.e. toward infeasible-like reactions.
Honest fit for *our* pipeline: **Medium.** JDS itself (compose two pretrained marginal models) doesn't map onto reaction feasibility. But the prior-free BPU-guidance component is High-value and cheap to adopt as our *classifier* recipe (and, if we ever build a reaction generator, as its steering mechanism). It's a workshop paper (semi-synthetic only), so treat as a technique source, not a validated system.

## Caveats / limitations / leakage risks
- Workshop paper; semi-synthetic datasets; no real-world compositional validation.
- Guidance is expensive (backprop through classifier at each step).
- Mixture well-posedness needs bounded density ratio (Appendix B) — may fail where feasible/infeasible reactions overlap heavily.
- Whether prior-free guidance transfers to flow-matching / rectified flow is stated as open.
- Leakage: unlabeled = independently paired marginals here; in our setting we must make sure the P-vs-U split doesn't put test reactions in P.

## Open questions for us
- Adopt the **BPU balanced-ERM posterior** as our baseline π-free feasibility classifier — does it beat nnPU (which needs π) on the JnJ test set?
- Could "guide toward the negative class" over an *enumerable* BH grid generate reliable-negative *labels* (rank grid cells by low BPU posterior) instead of synthesizing raw vectors — sidestepping invalid chemistry entirely?
- Does prior-free guidance survive on sparse fingerprint features vs the continuous embeddings tested here?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Ivanov (2020), "DEDPUL: difference-of-estimated-densities PU learning," ICMLA — source of the g/(1−g) posterior identity. doi:10.1109/ICMLA51294.2020.00128 🔒 (arXiv open)
- Elkan & Noto (2008), "Learning classifiers from only positive and unlabeled data," KDD — the non-traditional classifier. doi:10.1145/1401890.1401920 🔒
- Bekker & Davis (2020), "Learning from positive and unlabeled data: a survey," Machine Learning. doi:10.1007/s10994-020-05877-5 (open)
- Garg et al. (2021), "Mixture proportion estimation and PU learning: a modern approach," NeurIPS. (open)
- Zhu, Scott et al. (2023), "Mixture proportion estimation beyond irreducibility," ICML — relevant when feasible/infeasible supports overlap. (open, PMLR v202)
- Xu et al. (2023), Geo-LDM 3-D molecule diffusion, ICML. (open, PMLR v202)
- Hoogeboom et al. (2022), E(3)-equivariant diffusion for molecules, ICML. (open, PMLR v162)
