> **Provenance.** Summary written by a Claude agent on 2026-10-09 from the full PDF (paper and Supplementary Information), focused on this project's questions. Page references point to that PDF. Checked only where `docs/design/modeling.md` or ADR 003 relies on it; read the paper before citing a detail.
>
> **Reference.** Zhong et al., *Nat. Commun.* 2025, 16, 4522, doi:10.1038/s41467-025-59812-0; data Zenodo 10.5281/zenodo.12920294 (CC BY-NC-ND 4.0).

# Chemlex amide-coupling HTE dataset: structured summary

Source: Zhong, Liu, ... Yu, Wang, Wang. "Towards global reaction feasibility and robustness prediction with high throughput data and bayesian deep learning." *Nat. Commun.* 16, 4522 (2025), doi:10.1038/s41467-025-59812-0, plus its Supplementary Information (SI).

Citation key: **M p.X** = main article (11 pp.), **SI p.X** = Supplementary Information (51 pp., printed page numbers = PDF page numbers). "Not stated" means I found nothing on it in either document.

---

## 0. Headline facts

- 11,669 distinct acid-amine coupling reactions for 8,095 target products, 200–300 µL scale, run in 156 instrument working hours (M p.1, p.2).
- 272 acids, 231 amines, 6 coupling reagents (HATU, TCFH, BOP, PyBrOP, PyBOP, EDCI), 2 bases (DIEA, NMI), 1 additive (HOBt), 1 solvent (DMF) (M p.2; Fig. 2a, M p.3).
- About 5,600 of the reactions were deliberately designed by "chemist-designed rules" to be likely negatives (M p.4; SI p.4–6).
- The label is binary: conversion ≥ 20 % = feasible (SI p.12–13).
- Data: Zenodo doi:10.5281/zenodo.12920294 (M p.9). Code: github.com/Chemlex-AI/bayesian-reactivity-prediction, also Zenodo 10.5281/zenodo.15165244 (M p.9, ref. 49 M p.11).
- Licence: CC BY-NC-ND 4.0 (M p.11). Most authors work for ChemLex, a company that sells high-throughput synthesis; Y.H. is CEO of MegaRobo, which supplies the HTE equipment (M p.11).

---

## 1. Experimental protocol

**General procedure** (SI p.10–11):
- Coupling reagent: 0.04 mmol, 2.0 eq, dispensed as a **solid** by an automated powder-dispensing station, weighed **under air**.
- Then acid: 0.02 mmol, 1.0 eq, as a 0.2 M solution in dry DMF.
- Then amine: 0.02 mmol, 1.0 eq, 0.2 M in dry DMF.
- Then base: 0.06 mmol, 3.0 eq, 1.2 M in DMF.
- Liquids are added by microlitre pipette. No pre-activation step is described.
- Fig. 2a gives the same summary: coupling reagent 2.0 eq, base 3.0 eq, acid and amine 1.0 eq each, **DMF 0.08 M, rt, 0.5 h** (M p.3).
- The volumes add up: 100 + 100 + 50 µL = 250 µL, which gives 0.08 M, consistent with "200–300 µL" (M p.2). This calculation is mine.

**Temperature and time.** Shaker set to 30 min, "30 degrees", 200 rpm (SI p.10, p.11). The figures and reaction schemes say "rt" (M p.3, Fig. 2a; Fig. 3d M p.5). So the reaction ran at about 30 °C for 30 min, and was then analysed by LC-MS (SI p.11). 30 min is short for a literature-style coupling, which is often left overnight.

**Solvent and atmosphere.** DMF with ≤ 50 ppm water (Karl Fischer), SafeDry over molecular sieves, from Adamas (SI p.9). The feeding bin can be run under N₂ "according to the experimental environment requirements" (SI p.7). Whether N₂ was used for this dataset is not stated, and coupling reagents were weighed under air (SI p.10).

**Plate format.** Not a microtitre plate. The reactions ran in individual 8 mL glass screw-cap vials with PTFE-lined septa, held in 12-well trays (SI p.8–10). The reaction bin has 9 shakers × 12 wells, so 144 reactions run at the same time, and the platform does 240–288 reactions per day with 24 h operation (M p.9; SI p.7). The platform is CASL-V1.1, built with MegaRobo (M p.4, p.8–9; SI p.7–9). Reagent vendors were Leyan, Bide, Energy, Adamas, Sigma-Aldrich and TCI (SI p.9).

**How conditions were assigned to pairs.** Not stated. The paper never explains why some pairs were run under EDCI, TCFH or PyBOP and others only under HATU, and it gives no per-condition counts. The only clues:
- The dataset has 11,669 reactions but only 8,095 products (M p.2), so roughly 3,500 reactions repeat a product under another condition or as a replicate.
- The SI case studies show PyBOP/DIEA reactions next to HATU/DIEA reactions (SI p.5–6, Figs. S2–S3).
- BOP and PyBrOP appear in Fig. 2a (M p.3) but are not among the conditions we counted in the released table.
- HOBt is listed as the one additive (M p.3), but which reactions received it (presumably EDCI) is not stated.

**Why there are two HATU drawings.** Not stated. Fig. 2a draws HATU in the guanidinium (N-) form (M p.3). The O-form vs N-oxide difference in the released file is a data-entry artefact that the paper does not discuss.

---

## 2. Conversion measurement

**Instrument.** Agilent 1290 Infinity UHPLC with diode-array detector, coupled to an Agilent 6125B MS (SI p.9–10). The main text says G6125C (M p.8).
- Column: Waters SunFire C18, 50 × 4.6 mm, 3.5 µm, 40 °C, 2 mL/min.
- Mobile phase: water/MeCN with 0.1 % formic acid; 3 min gradient (SI p.10).
- DAD reads 214 nm and 254 nm, each with 4 nm bandwidth (SI p.10).

**Sample preparation.** 40 µL of the reaction mixture is diluted into 450 µL DMF/MeCN 1:1, filtered and injected (SI p.11). The injection volume is reported as 0.2 µL (SI p.11) and as 1 µL (SI p.10), so the SI contradicts itself.

**Wavelength used for the label: 254 nm** (SI p.11).

**Exact formula** (SI p.11, Eq. 1, citing King-Smith et al. *Nat. Chem.* 2024):

> Conversion Yield = Product area(%) / [1 − Coupling reagent area(%) − Acid or Amine area(%)] × 100 %

- All terms are area-% of the 254 nm chromatogram.
- The denominator removes the coupling-reagent peak and the peak of "acid **or** amine". Which of the two (the limiting one, or the UV-active one) is not stated.
- The main text calls this "the uncalibrated ratio of ultraviolet (UV) absorbance" (M p.4).

**Response factors.** None. The authors say that neither internal- nor external-standard methods could be used because the building blocks differ, and that isolating every product was impractical at 200–300 reactions per day (SI p.11). Product and starting-material extinction coefficients at 254 nm are therefore treated as equal.

**How product peaks were identified.** Implied by MS (the post-hoc LC-MS traces show product mass, SI p.43–46). A general peak-assignment rule is not stated.

**Replicates** (SI p.41; M p.7–8):
1. **486 reactions** were randomly chosen and run **2 or 3 times**, and these repeats were "added to our training set" to sharpen the aleatoric-uncertainty estimates (SI p.41).
2. Then the **3,507** test samples were ranked by aleatoric uncertainty. The 25 highest, 25 lowest and 25 random in between were each re-run **3 times** (SI p.41; M p.7, Fig. 5a).
3. Results: ICC 0.97 for the lowest-25, 0.62 for the random-25 and **0.07 for the top-25** (M p.7, Fig. 5b).

- How replicates are merged in the released table (mean, max, separate rows) is **not stated**. Your observation that replicates disagree as "worked once / zero once" matches the paper's own examples:
  - Case 1 (Fmoc-anthranilic acid + aminophenol, HATU): 0.0 %, 20.8 %, 35.3 % (M p.8).
  - Case 2 (Boc-protected pyrazole acid + 5-bromoquinolin-8-amine, HATU): 0.0 %, 10.2 %, 23.2 % (M p.8).
- Fig. 5a shows many top-25 reactions with replicate ranges spanning 0 % to above 50 % (M p.7).

**Why 20 %.** LC-MS yields are "susceptible to human factors... which inevitably introduce inherent noise", so yield regression was reframed as binary classification. Yield < 20 % = infeasible (class 0), ≥ 20 % = feasible (class 1) (SI p.12–13). Why 20 % in particular, rather than another cut-off, is **not stated**. The model's decision threshold is 0.5 (SI p.13; M p.9, Eq. 3).

---

## 3. Substrate selection

- **Reference space.** Acid-amine condensations in the Pistachio patent database, chosen because the target users are industrial medicinal chemists (M p.2).
- **Commercial availability.** Substrates are commercially available compounds from local vendors in Shanghai. Many Pistachio structures cannot be bought (M p.2–3).
- **Single reactive group.** "To minimize ambiguity from competing functional groups, we specifically choose substrates that contain only one carboxyl or amine group" (M p.3).
  - The dataset still contains other nucleophiles (-OH, -SH, NH-heteroaryl). The authors blame them for poor reproducibility: 15 of the top-25 high-aleatoric reactions have ≥ 2 active H, against 2 of the lowest-25 (SI p.41–42, Fig. S12).
- **Diversity method** (M p.3; SI p.13–14, Table S2, Algorithm 1):
  - Acids are sorted into 7 structural types and amines into 9, based on the carbon at the reaction centre. Types: SingleHomoAromatic, PolyHomoAromatic, SingleHeteroAromatic, PolyHeteroAromatic, SingleAliphaticRing, PolyAliphaticRing, AliphaticChain; amines add notCyclicSecondaryAmine and isCyclicSecondaryAmine.
  - The share of each type is matched to Pistachio.
  - Within each type, MaxMin-like picking on Tanimoto similarity is used.
- **Complexity check.** Böttcher complexity of the selected set is closer to Pistachio than the full buyable pool, which is simpler (M p.3, Fig. 2b–d; SI p.14–15).
- **Novelty.** "All the reactions in our HTE dataset are novel compared to the Pistachio dataset" (M p.3). How novelty was checked (exact reaction match or product match) is not stated.
- **Designed negatives** (M p.3–4; SI p.4–6):
  - Hirshfeld partial charge on the amine N: purchasable amines with charge > −0.14 (90th percentile of Pistachio) were selected.
  - Summed steric hindrance (occupied vdW volume in a 5 Å sphere) of amine N + acid C: candidate pairs with value > 0.35 (90th percentile of Pistachio) were selected.
  - About 5,600 reactions come from these rules (M p.4).
  - The SI steric text says the sphere is centred on "the carbon with the carbon-halogen bond or carbon-triflate bond" (SI p.6). This looks like a copy-paste error from another project.
- **Non-amine N-nucleophiles** (sulfonamides, amidines, thioamides, hydrazines, hydroxylamines). The paper never mentions them and gives no reason for including them: **not stated**. The likely route is the partial-charge rule, which favours weakly nucleophilic N (my inference, not the authors').

---

## 4. Splits

Defined at M p.9 and SI p.15–16, Fig. S4. Splits follow Raghavan et al. (JACS 2024) and Xu et al. (2022).

- **Random:** 70 % train, 30 % test (M p.9; SI p.16).
- **Stratified, one reactant unseen:**
  - Acids are sorted along one axis and amines along the other, and the grid is cut into four blocks.
  - The test set is the lower-right block; the other three blocks are training.
  - Definitions: "the test set includes either an acid or an amine that was not present in the training set" (M p.9; SI p.16); "the lower right part have only one reactant overlap with the rest parts" (SI p.15, Fig. S4a).
  - The **sort key is not stated**.
- **Stratified, both reactants unseen:**
  - Same grid, but the off-diagonal blocks are **discarded**, so no acid or amine appears in both train and test, "while still maintaining the 70 % training and 30 % test set distribution" (M p.9; SI p.15–16, Fig. S4b).
- **Why 4,475 rows have no both-unseen label.** They are the discarded off-diagonal block (SI p.15, Fig. S4b caption: "Discard off-diagonal data"). 4,475/11,669 ≈ 38 % dropped (my arithmetic). The paper gives no count of discarded rows.
- **Train/test sizes.** Only the 70/30 ratio is stated. The one explicit count is **3,507 test samples** in the aleatoric analysis (SI p.41), which is about 30 % of 11,669, so it is probably the random-split test set. That link is my inference, and the split is not named there.
- t-SNE shows the domain shift growing from random to one-unseen to both-unseen (M p.6, Fig. 4a).

---

## 5. Models and results

**Features.** DRFP (differential reaction fingerprint), radius 0–3 over reactants and products (SI p.20). DRFP takes the full reaction SMILES; whether the coupling reagent and base are in the SMILES is not stated.

Alternatives tested:
- MPNN/GNN embeddings: summed reactant + reagent embeddings, concatenated with product embeddings (SI p.25–27).
- rxnfp transformer fingerprints (SI p.27).

DRFP stayed the primary representation (SI p.28).

**Models.** All use a two-layer network on top of the fingerprint (SI p.20; M p.4):
- BNN + NUTS with a composite Laplace + Gaussian prior and 4 chains (M p.9; SI p.32–34).
- BNN + SVI.
- MC-dropout with 100 passes (SI p.17).
- A 100-model deep ensemble (SI p.18).
- DKL-GP with 70 inducing points (SI p.19).

Hyperparameters were grid-searched on the **random split only** and reused for the other splits and for the Suzuki data (SI p.20, p.29).

**Wetlab results, DRFP** (Acc, F1-macro, AUC-ROC; SI p.21 Table S3; M p.4):

| Model | Random | One unseen | Both unseen | Average |
|---|---|---|---|---|
| BNN+NUTS | **89.48 %, 0.86, 0.94** | **80.84 %, 0.74, 0.87** | 72.47 %, 0.65, 0.79 | **80.93 %, 0.75, 0.87** |
| BNN+SVI | 88.22 %, 0.85, 0.92 | 79.27 %, 0.74, 0.84 | **73.14 %, 0.67, 0.77** | 80.21 %, 0.75, 0.84 |
| MC-dropout | 89.08 %, 0.86, 0.93 | 77.47 %, 0.66, 0.84 | 70.81 %, 0.62, 0.78 | 79.12 %, 0.71, 0.85 |
| Ensemble | 88.31 %, 0.85, 0.93 | 75.39 %, 0.73, 0.84 | 69.87 %, 0.69, 0.77 | 77.86 %, 0.76, 0.85 |
| DKLGP | 83.09 %, 0.80, 0.88 | 76.16 %, 0.72, 0.78 | 52.78 %, 0.53, 0.68 | 70.68 %, 0.68, 0.78 |

- On both-unseen, BNN+NUTS is **not** the best on accuracy and F1; BNN+SVI is (SI p.21). The main text reports only the BNN+NUTS numbers (M p.4).
- With GNN features (Table S7, SI p.25), BNN+NUTS both-unseen = 73.72 %, 0.70, 0.79.
- Table S11 (SI p.26) gives BNN+NUTS both-unseen = 74.89 %, 0.71, 0.81. It is captioned "GNN-based features", but the text says S11–S14 are the transformer-based features (SI p.24), so the caption appears to be wrong.
- The **best both-unseen AUC anywhere is about 0.79–0.81**.

**Other results**
- Calibration, ECE for BNN+NUTS: 0.009 on random and 0.015 on one-unseen. ECE was **not** computed for both-unseen, because "calibration makes sense only when the model demonstrates a reasonable rate of accuracy" (SI p.35, Table S15).
- Active learning: about 1/5 of the data gives similar F1 ("saves ~80 %") (M p.1, p.8, Fig. 4e–g).
- Pfizer Suzuki HTE (5,760 reactions, 10 folds): BNN+NUTS 88.61 % accuracy, F1 0.87, AUC 0.95 (SI p.21–23, Tables S4–S6).
- Compared with Schleinitz et al. (NiCOlit) on one-substrate hold-out: AUC 0.78 vs their 0.74 (SI p.30).
- OOD detection: mean epistemic uncertainty is 0.075 on the wetlab test set and 0.268 on the Suzuki set (SI p.40).

**Per-condition results (HATU, etc.).** **Not stated.** There is no breakdown by coupling reagent or base anywhere in the paper or SI.

**Literature and reporting bias**
- Patent and literature data skew towards high yield; this HTE set has "richer low-yield reactions" (M p.4, Fig. 3b violin plots comparing NiCOlit, AZ ELN, this HTE, Suzuki HTE, B-H HTE, a SciFinder query and Pistachio 2.1.2).
- "Accessible vs explored" chemical space, accessible / explored:
  - This HTE: 9.5×10⁵ / 11,669.
  - AZ ELN: 7.3×10⁷ / 456.
  - NiCOlit: 7.2×10¹¹ / 1,406 (M p.5, Fig. 3b).
- "Surprising" positives: a hindered pair gave 62.12 % and a low-charge amine gave 73.97 % (M p.4–5, Fig. 3d).
- SciFinder literature set, built for the aleatoric/robustness analysis:
  - Single-step couplings with 11 common reagents, including HATU, EDC, PyBOP, HOBt and CDI (SI p.47).
  - Discovery phase: mg-scale reactions from BMCL, J. Med. Chem. and EJMC. Process phase: kg-scale reactions from OPRD (M p.8; SI p.47).
  - Counts: the main text says **17,332 discovery / 84 process** (M p.8). SI Table S16 says 719 + 9,399 + 4,478 = **14,596 discovery / 70 process** (SI p.48). The two do not agree.
  - Mean aleatoric uncertainty: 0.199 kg vs 0.221–0.226 mg (SI p.48).
  - This set was used only for uncertainty analysis. No model was trained on it, and no accuracy of a literature-trained model on the HTE data is reported.

---

## 6. Issues for using Chemlex as an external benchmark for a literature-trained HATU/DIPEA/DMF model

1. **Condition mismatch with the literature.**
   - Chemlex runs 30 min at about 30 °C, 0.08 M, 2 eq coupling reagent, 3 eq base, solid reagent added first, no pre-activation (SI p.10–11; M p.3).
   - Literature HATU couplings are typically run longer, often with pre-activation. Some literature "positives" that need hours will be Chemlex negatives.
   - The 2 eq of HATU also favours guanylation of the amine and O-acylation of phenols, which the authors themselves note (M p.8).
2. **The label is UV area-% conversion, not yield.**
   - 254 nm only, no response-factor correction, and a denominator that removes the reagent and one starting material (SI p.11).
   - Weak 254 nm chromophores (aliphatic acids and amines) can distort the ratio either way.
   - Literature yields are isolated yields, so the two scales are not directly comparable.
3. **Enrichment for designed negatives.**
   - About 5,600 of 11,669 reactions were chosen to be hard: high-charge (weakly nucleophilic) amine N or high steric hindrance (M p.4; SI p.4–6).
   - The base rate and difficulty are therefore not representative of medicinal chemistry. A literature model trained on mostly positive data will look worse here than in practice.
   - Results should be reported split by "designed-negative" origin if that flag can be recovered. The paper does not say whether the released file marks these rows: **not stated**.
4. **Label noise.**
   - The paper's own replicates show ICC 0.07 for the top-25 high-aleatoric reactions, and individual pairs flip between 0 % and above 20 % (M p.7–8).
   - The 20 % threshold sits right where these reactions scatter (M p.7, Fig. 5a).
   - Benchmark accuracy ceilings should account for this, for example by reporting metrics on pairs with consistent replicates.
5. **Duplicates and replicates.**
   - 486 reactions were repeated 2–3× and added to training (SI p.41), and another 75 were repeated 3× (M p.7).
   - The table has 11,669 rows but only 8,095 products (M p.2).
   - Aggregate by (acid, amine, condition) before scoring, or the benchmark double-counts.
6. **Leakage against the training corpus.**
   - The authors claim every reaction is novel relative to Pistachio (M p.3), but the method is not stated.
   - Building blocks were deliberately picked to resemble Pistachio substrates (M p.2–3), and they are cheap commercial compounds that likely appear in USPTO/ORDerly as reactants.
   - So molecule-level overlap is likely even where reaction-level overlap is absent. Exact-product and per-reactant overlap with our training set should be checked directly.
7. **Splits are defined for their in-domain models.** For an external model, the more relevant fact is the overlap of Chemlex acids and amines with *our* training set; their stratified labels say nothing about that. The both-unseen split also drops 38 % of rows (SI p.15; my count).
8. **Reported metrics depend on hyperparameters tuned on the random split** (SI p.20). The comparison numbers are AUC ≈ 0.79 for both-unseen and 0.87 for one-unseen (SI p.21).
9. **No per-condition analysis, and the condition assignment is undocumented** (§1). Differences between HATU, PyBOP, EDCI and TCFH in our data cannot be traced to a design choice described in the paper.
10. **Minor inconsistencies to be aware of:**
    - MSD model 6125B (SI p.9–10) vs G6125C (M p.8).
    - Injection volume 1 µL (SI p.10) vs 0.2 µL (SI p.11).
    - "rt" (M p.3) vs a 30 °C shaker setting (SI p.10).
    - SciFinder counts differ between main text and SI (M p.8 vs SI p.48).
    - Table S11 caption (SI p.26).
    - Steric-hindrance text mentions a C–halogen/triflate centre (SI p.6).
11. **Conflict of interest and licence.** The authors work for the companies that sell the HTE platform (M p.11). The dataset is under CC BY-NC-ND (M p.11). Check whether redistributing derived benchmark files counts as "adapted material".

---

## 7. Facts to cite in our paper

- 11,669 acid-amine couplings, 8,095 products, 272 acids × 231 amines, 6 coupling reagents, 2 bases, DMF, run on an automated HTE platform at 200–300 µL scale (Zhong et al. 2025, M p.2).
- Conditions: coupling reagent 2.0 eq, base 3.0 eq, acid/amine 1.0 eq each, DMF 0.08 M, 30 min, shaker at 30 °C (M p.3, Fig. 2a; SI p.10–11).
- Outcome: LC-MS conversion from uncalibrated 254 nm UV peak-area ratios, Product% / (1 − reagent% − acid-or-amine%) (SI p.11; M p.4). Binary label at ≥ 20 % (SI p.12–13).
- Substrates are commercially available, contain a single carboxyl or amine group, and were diversity-sampled to match Pistachio's structural-class distribution (M p.3; SI p.13–14). About 5,600 reactions were designed as likely negatives via partial-charge and steric rules (M p.4; SI p.4–6).
- The dataset contains more low-yield reactions than literature/patent data (M p.4, Fig. 3b).
- Reproducibility: ICC 0.97 / 0.62 / 0.07 for lowest / random / highest-aleatoric triplicate sets (M p.7, Fig. 5b). This is direct evidence of label noise near the 20 % threshold.
- Splits: random 70/30; one-reactant-unseen; both-reactants-unseen, which discards off-diagonal data (M p.9; SI p.15–16).
- Their best in-domain DRFP model: accuracy/AUC 89.48 %/0.94 (random), 80.84 %/0.87 (one unseen), 72.47 %/0.79 (both unseen) (SI p.21, Table S3; M p.4).
- Data: Zenodo doi:10.5281/zenodo.12920294 (M p.9), CC BY-NC-ND 4.0 (M p.11).
