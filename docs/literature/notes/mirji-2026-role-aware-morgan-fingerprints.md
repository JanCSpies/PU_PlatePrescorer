> **Provenance.** Agent summary, not yet checked by Luka. Written 2026-10-28 from the text of the PDF as read by the coordinating session; I did not read the PDF myself. No page or table numbers were passed to me, so locators are `[n]`. Read the paper before citing a detail.
>
> **Reference.** Chinmay Mirji (Aerospace Engineering, Embry-Riddle Aeronautical University), Prashant Shekhar (Mathematics, Embry-Riddle), Foram Madiyar (Chemistry, Bethune Cookman University), Hao Peng (Aerospace Engineering, Embry-Riddle). *Role-Aware Morgan Fingerprints for Reaction Yield Prediction*. arXiv:2609.22167v1 [cs.LG], 26 August 2026. Code: https://github.com/chinmaymirji/morgan-fp-yield-prediction.git. DOI: not stated. Licence: not stated. Date read: 2026-10-28.

# Mirji et al. 2026: role-aware Morgan fingerprints for yield prediction

## What they did

**The descriptor.** Let phi(.) be the compressed Morgan count fingerprint of one molecule. For reactant set L, reagent set M and product p they form R = sum of phi(s) over L, M = sum of phi(s) over M, and P = phi(p). The descriptor concatenates six blocks: R, M, P, (P - R), P - (R + M), and |P - (R + M)|. These stand for identity, transformation and transformation magnitude. Six scalars (component counts and presence flags) are added, so x is in R^(6D+6), and it goes into an MLP. Counts are clipped, then log(1 + x). They say the differential idea is adapted from DRFP, here for count fingerprints with explicit role separation.

**The ablation, which is what we cite.** Eight configurations over three radius settings: radius 2 only, radius 3 only, and radii 2 and 3 concatenated. nBits is {1024, 2048, 4096} for a single radius and {(2048, 2048), (4096, 4096)} for the concatenation. In their words, "This grid separates the effect of neighborhood depth (radius) from hash collision frequency (nBits)." They screen on split 0 of each dataset, then run the best radius across all ten splits. Their conclusion: "Radius-2 neighborhoods capture the ortho substitution and ligand steric environments that drive cross-coupling yield, while radius-3 adds conserved scaffold fragments that increase hash collisions without improving discrimination."

## What the data is

Suzuki-Miyaura and Buchwald-Hartwig HTE yield benchmarks, with shared preprocessing and evaluation. Sizes: `[n]`. They report R^2 = 0.878 on Suzuki-Miyaura and R^2 = 0.969 on Buchwald-Hartwig, against YieldBERT (with and without augmentation) and GNAN, and train an order of magnitude faster. Locators: `[n]`.

## What carries over to this project

- Radius 2 is a sound default and radius is not the lever. ADR 018 fixes radius 2 and 2,048 bits for every candidate `d`, so only the root set varies.
- **ADR 018 leans on this paper only as corroboration.** The primary evidence that radius is not the lever is our own measurement: `fromAtoms=[root]` yields exactly `radius + 1` environments, so radius 5 would give six bits. The ADR does not rest on this paper.

## What does not carry over

1. Their task is yield **regression** on HTE data. Nothing here speaks to reporting bias, to PU learning, or to substrate-level descriptors. Their descriptor is reaction-level.
2. **They state their own limit, and it matters more:** "This study reports performance only under random split evaluation. Harder protocols like scaffold splits, component-family holdouts, or temporal partitions would test generalization under distribution shift more rigorously." Our setting is a distribution shift: literature substrates scored against a lab inventory. Their radius conclusion is established in the easiest regime and may not carry to ours.

## Numbers and claims we may cite

- Radius 2 at nBits 2048 as the strongest default in their grid, on random splits of two cross-coupling HTE sets. Table locator: `[n]`.
- R^2 = 0.878 (Suzuki-Miyaura) and 0.969 (Buchwald-Hartwig), random splits. Locator: `[n]`.
- **Their citation of someone else's result, not their own measurement:** an analysis of 25 pretrained molecular embedding models across 25 datasets found that "nearly all learned representations performed at or below the level of baseline ECFP fingerprints". The original paper is not identified in what I was given: `[n]`. Relevant the next time anyone proposes a learned embedding here.
- They also cite Saebi et al. on dataset bias and noise, and Voinarovska et al. on unresolved challenges. Full citations: `[n]`.
