# Nippa, Atz et al. — Expediting Hit-to-Lead via Reaction Prediction + Multi-Dimensional Optimization
- **Citation:** D. F. Nippa*, K. Atz*, Y. Stenzhorn, A. T. Müller, …, R. E. Martin, U. Grether, G. Schneider.
  *Expediting hit-to-lead progression in drug discovery through reaction prediction and multi-dimensional optimization.*
  *Nat. Commun.* 2025, 16, 11646. DOI 10.1038/s41467-025-66324-4. (Roche pRED + ETH Zurich + Leiden.)
- **Type:** application (closest end-to-end analogue of our pipeline vision)
- **File:** literature/s41467-025-66324-4.pdf
- **TL;DR:** Train a GNN on 13,490 HTE **Minisci C–H alkylation** reactions to predict reaction outcome, then use it as a
  synthesizability filter inside a multi-objective "ML funnel" over a 26,375-molecule virtual library → 212 candidates →
  14 synthesized MAGL inhibitors, some subnanomolar (up to 4,500× potency gain), in a one-month cycle.

## Problem & contribution
Hit-to-lead is bottlenecked by synthesis. Contribution: an integrated **reaction-outcome prediction + potency + ADME**
"machine-learning funnel" that prioritizes only **synthetically accessible, potent, well-behaved** designs before any lab
work — demonstrated prospectively to real, potent inhibitors with co-crystal structures.

## Dataset(s) — scope, size, conditions, feasibility definition
- **Minisci-type C–H alkylation (NOT BH)**: 13,490 reactions = 80 N-heteroarene fragments × 59 sp3-rich carboxylic acids
  (4,720 substrate combos) × conditions (6 Ag catalysts, 4 additives, oxidant (NH4)2S2O8, MeCN/H2O), 24-well plates, LC-MS.
  ~115k theoretical combos, ~12% sampled.
- **Feasibility threshold = ≥5% LCMS yield = positive**; **30% of reactions positive**. Yield modeled as regression in [0,1]
  (sum of mono+di-alkylation); binary outcome also as regression. Data in **SURF** format (FAIR).

## Method & features
- **E(3)-equivariant graph transformer (GTNN)**, ~2M params; 3D conformers (RDKit/UFF, 10 confs averaged); atom one-hot +
  distance Fourier features; reaction conditions one-hot (4 reagents, 4 solvents, 11 catalysts, 4 additives, 2 atmospheres);
  mean of 3 models. Code github.com/ETHmodlab/minisci; data Figshare 10.6084/m9.figshare.28294850.

## OOD / splitting strategy & metrics
- Four splits: **0D** (random, known combos), **1DA** (novel acids), **1DN** (novel N-arenes), **2D** (both novel).
- Yield: MAE 6.7% (0D) → 11–13% (1D/2D); Pearson r 0.83 (0D) → 0.51–0.55 (2D). Binary: accuracy 85.7%, **precision 90.9%
  (0D) → 85% (2D)**. Extrapolation degrades gracefully; precision stays high (good for a "don't-waste-synthesis" filter).

## Relevance to our pipeline (esp. leakage risk vs JnJ test set)
- **This is the template for the downstream half of our thesis**: reaction-outcome model → score a big virtual
  substrate library → keep only synthesizable+valuable → make the plate. We should explicitly position our work as
  "same funnel, but the reaction filter is a **PU feasibility model trained on literature-only data with no HTE negatives**,
  applied to **BH C–N** and general conditions."
- Their **≥5% positive threshold + precision-first evaluation on 1D/2D splits** is a clean pattern to mirror (BH: novel-aryl /
  novel-amine / both-novel splits). Their precision-over-recall priority matches ours (a false "feasible" wastes a well).
- **No BH / no JnJ overlap** (Minisci chemistry) → zero leakage; purely a methodological/architectural precedent.
- Delta we can claim vs Nippa: (i) literature-only training, no proprietary HTE negatives; (ii) PU/implicit negatives instead
  of measured negatives; (iii) explicit **general-conditions** framing; (iv) benchmarked against a real OOD HTE test (JnJ).

## Open questions for us
- Adopt their multi-objective "funnel" ordering (feasibility → potency/ADME) for the eventual plate-design story, or keep
  feasibility-only for the hackathon scope?
- Their split taxonomy (0D/1D/2D) vs Neves' K-means clusters vs Zhong's stratified — standardize on one for reporting.
- Is an E(3)-equivariant GTNN worth it for BH feasibility, or is DRFP+GBM/RF enough (Neves suggests the latter for <25k)?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Nippa, …, Grether. Late-stage diversification via HTE + geometric deep learning. *Nat. Chem.* 2024, 16, 239. DOI 10.1038/s41557-023-01360-5 🔒
- Götz et al. HTE data predicts molecular properties + reaction success. *Sci. Adv.* 2023, 9, eadj2314. 🔓
- Atz, Nippa et al. 2D3D-hybrid scoring (potency). (ref 40 in paper) — code public.
- SURF reaction format (ref 16). FAIR HTE data standard.
