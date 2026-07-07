# Positive-Unlabeled Diffusion Models (PUDM)
- **Citation:** Hiroshi Takahashi, Tomoharu Iwata, Atsutoshi Kumagai, Yuuki Yamanaka, Tomoya Yamashita. "Positive-Unlabeled Diffusion Models for Preventing Sensitive Data Generation." ICLR 2025. OpenReview id=jKcZ4hF4s5. Code: github.com/takahashihiroshi/pudm
- **Type:** method
- **File:** literature/Diffusion_FlowMatching__PU/ICLR-2025-positive-unlabeled-diffusion-models-for-preventing-sensitive-data-generation-Paper-Conference.pdf
- **TL;DR:** A diffusion model trained on unlabeled + a little "positive/sensitive" data that learns to generate *only the negative (normal) distribution* — by rewriting the normal-data ELBO as (unlabeled − β·positive) via the PU mixture identity, plus an nnPU-style non-negativity clamp.

## Problem & contribution
Diffusion models trained on web-scraped unlabeled data reproduce the "sensitive" data hidden inside it. You can label a *little* sensitive (positive) data cheaply but cannot curate a clean "normal" (negative) set. So you must train a generator in the **PU setting**. Contribution: first diffusion model trained under PU; it maximizes the ELBO for normal (negative) data and minimizes it for sensitive (positive) data *using only unlabeled + positive samples*, without ever seeing labeled negatives, and without degrading sample quality (FID).

## Method (key technical detail)
1. **Diffusion-as-classifier bridge (Yamanaka 2019):** define p(y|x) from the per-sample negative-ELBO ℓ(x;θ): p(y=0|x)=exp(−ℓ), p(y=1|x)=1−exp(−ℓ). Train with binary cross-entropy ℓ_BCE. This makes "maximize ELBO for normal / minimize for sensitive" bounded on both sides (avoids the ELBO-diverging-to-−∞ failure of naive concept-forgetting methods).
2. **The PU trick (core idea):** model unlabeled density as a mixture pU = β·pS + (1−β)·pN. Therefore the *normal* expectation you cannot compute directly is rewritten as
   (1−β)·E_pN[ℓ_BCE(x,0)] = E_pU[ℓ_BCE(x,0)] − β·E_pS[ℓ_BCE(x,0)].
   So the full supervised PN objective L_PN = β·E_pS[ℓ_BCE(x,1)] + E_pU[ℓ_BCE(x,0)] − β·E_pS[ℓ_BCE(x,0)] is computable from **positive + unlabeled only**.
3. **Non-negative clamp (Kiryo 2017):** the (unlabeled − β·positive) term should be ≥0; when it goes negative, flip the gradient to push it back up — the nnPU correction, preventing overfitting. β (sensitive fraction in U) is a hyperparameter, estimable by standard class-prior methods.
4. Extends to conditional models (classifier-free guidance) and to PNU when a few normals are available.

## Modality/data it targets (images? tabular? sequences?)
**Images only** — continuous-space DDPM/DDIM with a U-Net noise predictor. MNIST(32×32), CIFAR10, STL10, CelebA(256×256), plus Stable Diffusion in the appendix. No tabular/molecular experiments. The *math* is modality-agnostic (any diffusion ELBO), but everything demonstrated is continuous image data.

## Datasets / benchmarks
MNIST (even/odd), CIFAR10 (animals/vehicles), STL10, CelebA (male/female). Metrics: "non-sensitive rate" (fraction of generated samples in the normal class per a pretrained classifier) + FID. Baselines: Unsupervised, Supervised (naive), and (in appendix) comparison to Selective-Amnesia concept forgetting.

## Results & claims
Non-sensitive rate improves across all datasets (e.g. CIFAR10-vehicles 0.762 unsup → 0.842 proposed; fine-tuned CelebA-male 0.868 → 0.973) with negligible FID cost; the naive "supervised" method barely beats unsupervised because sensitive data in U cancels the effect — the PU correction is what matters. β can be tuned *above* its true value (1/11≈0.091) to trade a bit of FID for a higher non-sensitive rate. Converges to the supervised objective as N,M→∞.

## Relevance to our BH C–N PU pipeline (be honest about fit)
**Directly reframes "generate reliable negatives."** In our mapping, *positives = feasible literature reactions* and the *"normal/negative" class = infeasible couplings*. PUDM learns a generative model of the **negative** (infeasible) region from positives+unlabeled — exactly a reliable-negative synthesizer. The (unlabeled − β·positive) ELBO identity is the same mixture algebra our whole PU setup rests on and is the cleanest formulation among the four papers. **BUT:** (1) it is built for *continuous* diffusion; our reaction features are fingerprints/descriptors (mixed discrete/continuous, sparse) — needs a tabular/latent diffusion or flow backbone (see TabDiff, RecTable, TabbyFlow). (2) It needs β, our unknown feasibility base-rate. (3) Its *deliverable* is a generator; to help a classifier we'd then sample synthetic negatives and hope they're valid reactions — same "invalid chemistry" risk as GenPU. **Higher-value use for us may not be negative synthesis at all but the loss itself:** the diffusion-as-classifier + PU-corrected ELBO gives a *density/feasibility score* ℓ(x) per reaction we could use directly as a feasibility ranker, no generation needed.

## Caveats / limitations / leakage risks
- Images only; tabular/molecular untested.
- β sensitivity; authors call optimal-β analysis "future work."
- Generating a "negative reaction vector" ≠ a valid enumerable reaction; needs to be constrained to the real combinatorial grid.
- Leakage: if held-out JnJ test reactions sit in the unlabeled pool, the learned negative density is contaminated by them.
- Compute: from-scratch diffusion ~100 epochs on A100/V100 — heavy for a hackathon week.

## Open questions for us
- Can we run this on a *latent* reaction embedding (learn an encoder over the combinatorial grid, diffuse in latent space) so generated negatives always decode to valid reactions?
- Better: skip generation and use the PU-corrected per-sample ELBO ℓ(x) as a **feasibility score** — does it beat a plain nnPU classifier on the JnJ test set?
- Can β be tied to TACIT/PAYN's implicit-negative rate?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Yamanaka et al. (2019), "Autoencoding binary classifiers for supervised anomaly detection," PRICAI — the diffusion-as-classifier bridge; the loss trick we'd reuse. (Springer 🔒; arXiv version open)
- Kiryo et al. (2017), nnPU, NeurIPS — the non-negativity clamp. (open)
- Takahashi et al. (2024), "Deep PU anomaly detection for contaminated unlabeled data," arXiv:2405.18929 — non-generative sibling; may be the more practical tabular analog. (open)
- Ho, Jain, Abbeel (2020), DDPM, NeurIPS. (open)
- Nakajima & Sugiyama (2023), prior-invariant PU via density-ratio, Machine Learning — for β/π estimation. 🔒
