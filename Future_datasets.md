# Public HTE datasets — survey, and why the usable ones are exhausted

> **Status: CLOSED — question answered, decision made (updated 2026-07-29).**
> Read this as *evidence*, not as a menu. Its conclusion is the motivation for the wet-lab
> experiment in `PROSPECTIVE_PLATE_PLAN.md`.
>
> **What it was for (2026-07-09):** at the time the project tested on `JnJ HTE 2024` (Neves et al.)
> and transfer was ~random, so this was a search for a *replacement public test set*.
> **What happened:** the search ran, the two viable candidates were adopted, and both were used —
> **Chemlex amide** (TEST n=2,234) and the **HiTEA BH slice** (TEST n=36). Nothing here is pending.
>
> **The conclusion that matters now.** Applying the three requirements below strictly, **no public
> dataset besides Chemlex and HiTEA qualifies**, and each of those has a hard ceiling:
>
> | | why it fails the bar |
> |---|---|
> | **Chemlex amide** | ✅ used — all 11,669 rows measured; caveat is that negatives are *enriched by expert-rule design* rather than randomly sampled, and conditions are narrowed (1 solvent, 2 bases) |
> | **HiTEA BH** | ✅ used — but the BH slice is only **29 distinct substrate pairs** (→ our n=36 test, CIs so wide the result barely constrains anything) |
> | LSF borylation | mono-molecular (40 substrates), not a substrate-pair matrix |
> | Jaman SNAr | ~192 pairs, readout is a semiquantitative MS threshold, not yield |
> | AstraZeneca ELN BH | negatives **refuted** — "no product" is likely missing-not-failed |
> | AbbVie Suzuki | 23k diverse products, but **structures redacted** — no SMILES, cannot featurise |
> | RSC 2025 amide | 632 pairs, but (b) and (c) never verified |
>
> **Therefore:** the public literature contains **no untapped dataset** with a diverse product matrix,
> genuinely measured negatives, and literature-common conditions. This is not a gap to keep searching
> for — it is a structural fact about what has been published, and it is precisely why the next test
> set has to be **generated in the lab**. For the paper, this survey is the Methods/SI justification
> that the prospective experiment was necessary rather than merely nice to have.
>
> **Note on Suzuki specifically:** its absence from the viable list is not an oversight. Perera 2018
> varies leaving group, boron source, and conditions on a fixed coupling (≈ one product); Reizman/
> Jensen is condition optimisation. Neither is a substrate matrix. AbbVie is the only diverse-product
> Suzuki corpus and it is unusable. That is *why* Suzuki is a legitimate prospective target.
>
> **Method (unchanged):** deep-research harness — 3 search angles, 14 primary sources fetched,
> 68 claims extracted, 25 adversarially verified (3-vote, need 2/3 to kill),
> 22 confirmed / 3 refuted. All claims below rest on peer-reviewed primary sources
> with live open deposits unless flagged.

---

## The problem we are solving

Current test set = **JnJ HTE 2024** (Neves et al., *Robust OOD prediction of BH reactions*).
Our literature positives **P** transfer to it only ~randomly because JnJ is **adversarially
OOD by design**: 54.6% of test rows use a base that is <1% frequent in the literature
(P2-Et phosphazene 33.6% vs ~0% in lit; HMDS/silanolate ~0% in lit). No representation
closes a data-coverage gap that was *engineered in*.

**Requirements for a replacement test set:**
- **(a) Diverse substrates** — a broad aryl × amine (or analogous) matrix, not a
  condition-optimization screen on 1–2 fixed products.
- **(b) Real negatives AND positives** — genuinely *measured* failed reactions (0% / low
  yield), so react/no-react labels are real, not positive-only.
- **(c) Natural / representative sampling** — natural selection of reactions/conditions,
  NOT an exotic robust-condition OOD grid.

**Key structural insight from the search:** HTE datasets sidestep the "missing-not-failed"
trap by construction — every well is physically assayed, so a 0% is a *measured* 0%, not an
unrecorded reaction. This is why HTE beats ELN/patent corpora on requirement (b), and why the
natural-conditions HTE sets (below) beat JnJ on (c) without losing (b).

---

## Ranked shortlist (verified candidates)

| # | Dataset | Reaction | Diversity (a) | Real negatives (b) | Natural conditions (c) | Available? |
|---|---|---|---|---|---|---|
| 1 | **Chemlex acid–amine** | Amide coupling | ✅ 272 acids × 231 amines, 8,095 products | ✅ **all 11,669 measured** (82% of negatives are a hard 0.00); ⚠️ negatives *enriched by expert-rule design*, not randomly sampled | ✅ standardized common reagents (narrowed, not exotic) | ✅ Zenodo, CC-BY-NC |
| 2 | **Pfizer HiTEA "reactome"** | **BH** + Ullmann + 2× hydrogenation | ⚠️ full corpus diverse; **BH slice 29 distinct pairs (wide-but-shallow)** | ✅✅ genuinely measured 0% yields | ✅✅ ~10 yr real pharma HTE | ✅ GitHub, CC-BY |
| 3 | **ETH/Roche LSF borylation** | C–H borylation | ⚠️ 40 substrates (mono-molecular, not a pair matrix) | ✅ LCMS success/fail, authentic negatives | ✅✅ **only candidate verified on all 3** | ✅ GitHub |
| 4 | **Jaman SNAr** | SNAr (amine + aryl halide) | ✅ ~192 pairs (16 amines × 13 halides) | ✅ measured, majority fail | ⚠️ semiquantitative MS threshold | ✅ via PMC |
| — | AstraZeneca ELN BH | BH | ✅ 340 aryl × 260 amines | ❌ **REFUTED — likely missing-not-failed** | ✅ natural ELN | ⚠️ contested |
| — | AbbVie 15-yr Suzuki | Suzuki | ✅ 23k products | ✅ isolated yields | ✅ med-chem parallel libs | ❌ **structures redacted** |
| — | RSC 2025 amide | Amide coupling | ✅ 632 pairs (70 amines × 66 acids) | ❓ unverified | ❓ unverified | ✅ open-access |

---

## Detailed candidate cards

### 1. Chemlex acid–amine (amide) coupling — *best overall public dataset*
- **Citation:** "Towards global reaction feasibility and robustness prediction with high
  throughput data and Bayesian deep learning," *Nat. Commun.* 2025.
- **Reaction:** amide coupling (acid + amine). *Not BH — permitted since reaction class doesn't matter.*
- **Size / diversity (a):** 11,669 reactions for 8,095 target products, covering **272 acids ×
  231 amines**, 6 condensation reagents, 2 bases, 1 solvent. Authors explicitly contrast their
  substrate-diversity emphasis against narrow BH/Suzuki condition screens. ✅
- **Negatives (b):** explicit react/no-react boundary at **20 % conversion** (Fig. 5).
  ✅ **RESOLVED 2026-07-29 — the negatives ARE measured.** An earlier revision of this note claimed
  ~5,600 negatives were "chemist-rule-designed synthetic (never lab-run)". **That was wrong**, and it
  conflated *how reactions were chosen for the plate* with *how their labels were assigned*.
  Verified directly against `datasets/raw/Chemlex/Chemlex_Acidamine_Wetlab_Data.xlsx`:
  - **11,669 / 11,669 rows carry a measured `Conversion`; zero nulls; 3,305 distinct non-zero values.**
    Every reaction was physically run (200–300 µL, 156 instrument-hours, LC-MS UV readout).
  - At the 20 % threshold: **3,055 feasible (26.2 %) / 8,614 infeasible (73.8 %)**. Of the negatives,
    **82.4 % are a hard measured 0.00** — a detected absence of product, not a missing record.
  - The paper itself reports "surprising" reactions — expert-predicted-negative that came back
    high-yield. Those can only be found by *running* them.
- ⚠️ **The real caveat is selection bias, not fabricated labels.** The ~5,600 expert-rule reactions
  were **designed onto the plate** to be *likely* failures (steric hindrance, weak nucleophilicity)
  so the dataset would be negative-rich — running only literature-plausible pairs yields an almost
  all-positive set, useless for learning failure. Consequence: the negative class is **enriched for
  chemically obvious failure modes** rather than being a random sample of substrate space. A model
  can therefore score well partly by learning "bulky acid + weak nucleophile → fails", which is real
  but shallow. **So the honest concern is that ROC-AUC on this set may be optimistic relative to a
  random substrate matrix**, where failures are subtler and more varied — not that the labels are
  synthetic. This is precisely the bias a prospective random-vs-model plate does not have.
- **Conditions (c):** standardized common amide-coupling reagents — common, but a *narrowed*
  condition distribution (single solvent, 2 bases). Mild condition shift, not exotic OOD.
- **Availability:** Zenodo `10.5281/zenodo.12920294` (Open Access, CC-BY-NC-4.0) —
  `Chemlex_Acidamine_Wetlab_Data.xlsx`. Code: `github.com/Chemlex-AI/bayesian-reactivity-prediction`.
- **Links:** https://www.nature.com/articles/s41467-025-59812-0 ·
  https://doi.org/10.5281/zenodo.12920294
- **Why it wins:** best diversity + availability + explicit react/no-react, AND amide coupling is
  *the* most abundant reaction in the literature → re-mining positives **P** is easier than BH,
  so the domain gap shrinks.

### 2. Pfizer HiTEA "reactome" — *best if staying in BH*
- **Citation:** King-Smith et al., "Probing the chemical reactome with high-throughput
  experimentation data," *Nat. Chem.* 2024.
- **Reaction:** 4 classes — **Buchwald–Hartwig**, Ullmann, heterogeneous + homogeneous hydrogenation.
- **Size / diversity (a):** 39,000+ reactions over 350+ target products from ~10 yr of real
  pharma HTE. **BH slice = 31 halides × 32 nucleophiles but only 29 distinct reacting pairs**
  (~3% of the grid — a condition-screen structure, "wide but shallow"). Full multi-class corpus
  is broadly diverse; BH slice alone is not a substrate matrix.
- **Negatives (b):** ✅✅ every well physically assayed → 0% yields are **genuinely measured
  failures**. Paper argues removing 0%s "leads to a far poorer understanding of the reaction class."
- **Conditions (c):** ✅✅ natural med-chem HTE practice — directly attacks the JnJ OOD gap.
- **Availability:** `github.com/emmaking-smith/HiTEA`, CC-BY-4.0. Ships
  `8_SEPT_APPROVED_full_dataset.csv` + per-class files.
- **Links:** https://www.nature.com/articles/s41557-023-01393-w · https://github.com/emmaking-smith/HiTEA
- **Note:** nucleophiles include amide/aromatic-N/alcohol — broader than strict C–N amination.
- **Trade-off:** cleaner *feasibility* test than *substrate-generalization* test (only 29 BH pairs).

### 3. ETH/Roche late-stage-functionalization C–H borylation — *only all-3 verified*
- **Citation:** "Enabling late-stage drug diversification by high-throughput experimentation with
  geometric deep learning," *Nat. Chem.* 2023.
- **Reaction:** Ir-catalyzed C–H borylation (mono-molecular, **not** an aryl × amine matrix).
- **Size / diversity (a):** 956 measured data points over **40 substrates** (23 drugs + 12
  fragments + 5 simple). Diverse substrates but single-substrate reaction.
- **Negatives (b):** ✅ binary success/unsuccessful by LCMS (conversion ≥1%); unsuccessful =
  "authentic negative observations where no detectable product formed, not missing data."
  Classifier hits 92% balanced accuracy (needs both classes).
- **Conditions (c):** ✅✅ **the clearest (c) satisfier** — 24-well plate designed from a
  meta-analysis of **1,301 literature reactions**, continuous conditions set to *median literature
  values* (T=80°C, t=16h, c=0.2M). The exact opposite of Neves/JnJ's engineered-exotic conditions.
- **Availability:** `github.com/ETHmodlab/lsfml` (also `atzkenneth/lsfml`) —
  `data/experimental_rxndata.csv` (956 rows).
- **Links:** https://www.nature.com/articles/s41557-023-01360-5
- **Trade-off:** the only candidate independently verified on all three requirements, but the
  mono-molecular reaction is the furthest from our BH/aryl×amine framing.

### 4. Jaman SNAr — *secondary*
- **Citation:** Jaman et al., "High Throughput Experimentation and Continuous Flow Evaluation of
  Nucleophilic Aromatic Substitution Reactions," *ACS Comb. Sci.* 2020.
- **Reaction:** SNAr (amine + aryl halide).
- **Size / diversity (a):** 3,072 reactions evaluated; **16 amines × 13 aryl halides ≈ 192 pairs**
  across two rounds. ✅ (Note: the "1,536 in Round 1" figure counts format × base × solvent, NOT
  substrate diversity — Round 1 was 8 amines × 12 halides = 96 pairs.)
- **Negatives (b):** ✅ every well run + MS-analyzed; failure fractions large (Round 1 bulk ~80%,
  droplet ~90%; Round 2 ~95% fail). ⚠️ "success" = MS intensity ≥150 counts (S/N~5),
  semiquantitative, not calibrated yield; some false negatives acknowledged.
- **Conditions:** 4 bases × 2 solvents.
- **Availability:** readable via PMC.
- **Links:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12043022/

---

## Rejected / trap candidates (do NOT use as-is)

### AstraZeneca ELN Buchwald–Hartwig (Chem. Sci. 2023) — *the biggest trap*
- Saebi et al., "On the use of real-world datasets for reaction yield prediction," *Chem. Sci.* 2023.
- **Looks perfect:** 781 curated real ELN reactions, **340 aryl halides × 260 amines**, 24 ligands,
  15 bases, 15 solvents — a natural sparse scatter, *actual BH*. Satisfies (a) and (c) strongly.
- **BUT (b) REFUTED (3 separate claims, votes 0-3 / 0-3 / 1-2):** the "39.9% no-product = real
  negatives" claim was killed — for legacy ELN data, "no product recorded" is likely
  **missing-not-failed** (same trap as CHORISO). Methods define the 39.9% as a *mixture* including
  non-chemical, non-measured causes.
- **Availability also contested** (GitHub `nsf-c-cas` / ORD deposit unconfirmed, vote 1-2).
- **Verdict:** the most tempting-looking BH set, and the one most likely to silently corrupt the
  test labels. Avoid unless the measured-failure subset can be positively confirmed.
- **Links:** https://pubs.rsc.org/en/content/articlehtml/2023/sc/d2sc06041h

### AbbVie 15-year Suzuki parallel-library (JACS 2024) — *ideal but unusable*
- Raghavan et al., *JACS* 2024. 23,236 products / 24,203 reactions / 629 libraries; ~68%
  registration success / ~32% non-delivery with true isolated yields; common med-chem catalysts/
  bases/solvents. Satisfies all three **in substance**.
- **BLOCKER:** public release is **one-hot encoded — no SMILES, no structures, no DFT descriptors**
  (to conceal proprietary structures). A literature-trained model cannot featurize it compatibly.
- **Verdict:** benchmark reference only, not a usable cross-dataset test set.
- **Links:** https://pubs.acs.org/doi/10.1021/jacs.4c00098 ·
  https://github.com/priyanka-rag/suzuki_yield_predict_external

### RSC 2025 amide coupling — *supplementary, needs verification*
- "Intermediate knowledge enhanced the performance of the amide coupling yield prediction model,"
  *Chem. Sci.* 2025 (DOI 10.1039/D5SC03364K).
- 632 unique substrate pairs (70 amines × 66 acids) — satisfies (a). Sampling: 71% K-means-selected
  over ~71k virtual compounds, 29% manual. **(b) real-negatives fraction and (c) condition
  naturalness NOT assessed** — must be checked before use.
- **Links:** https://pubs.rsc.org/en/content/articlelanding/2025/sc/d5sc03364k

---

## Recommendation — *decided, both arms taken* (outcome added 2026-07-29)

> **Outcome:** we did not choose between these — we ran **both**. Chemlex became the anchor result
> (TEST ROC-AUC 0.768 inductive, beating the in-domain supervised ceiling 0.733); HiTEA BH was run as
> the independent replication and reached 0.709 on n=36 with CIs so wide it neither confirms nor
> refutes. That asymmetry — one solid class, one underpowered class, and nothing else public to try —
> is what closed this line of inquiry and opened the wet-lab one.

The fork "better BH set" vs "switch reaction" had a clear winner on each side:

- **Stay in BH → Pfizer HiTEA.** Only new dataset that is *actual BH* with genuinely measured
  negatives under natural pharma conditions → directly fixes the (c) OOD gap that killed JnJ.
  Price: modest BH substrate-pair diversity (29 pairs) → cleaner *feasibility* test than
  *substrate-generalization* test.
- **Switch reaction → Chemlex amide coupling.** Strongest single dataset on diversity +
  availability + explicit react/no-react, AND amide coupling positives **P** are far denser in the
  literature than BH → the domain gap gets *easier*. One open item: verify measured-vs-designed
  negative fraction.

**Leaning:** Chemlex is the strongest single dataset if switching reaction is acceptable (it is).
The domain-gap problem shrinks rather than grows.

---

## Open questions — status as of 2026-07-29

> Q1 is **RESOLVED** (2026-07-29) — and the answer was favourable. Q2–Q3 are moot (those datasets were
> not adopted / the HiTEA slice was used as-is and its small n is reported honestly). Q4 folds into the
> general condition-scope argument.

1. ✅ **RESOLVED — Chemlex negatives are measured.** All 11,669 rows carry a measured LC-MS
   conversion (0 nulls, 3,305 distinct non-zero values); 82.4 % of the negatives are a hard 0.00.
   Nothing was labelled by rule. **The residual caveat is selection bias, not label fabrication:**
   the negative class was deliberately enriched with expert-rule "likely-fail" designs (steric /
   weak-nucleophile), so it over-represents obvious failure modes and the headline AUC may be
   optimistic relative to a randomly sampled substrate matrix. State this as a limitation; it is a
   far weaker objection than fabricated labels, and the prospective random-vs-model plate is not
   subject to it at all.
2. **AstraZeneca ELN:** is 39.9% "no product" truly measured or missing-not-failed, and is the
   781-reaction file actually downloadable? (Both left unresolved/refuted.)
3. **HiTEA:** is the 29-pair BH slice diverse enough alone, or must we pool the broader multi-class
   corpus (or another class) for sufficient substrate-pair coverage?
4. **Amide HTE conditions (Chemlex / RSC 2025):** do their fixed standardized conditions (1
   solvent / 2 bases) introduce their own milder OOD gap vs literature-common conditions, i.e. do
   they actually satisfy (c)?

## Cross-cutting caveats

- **Reaction-class mismatch:** only HiTEA's BH slice + AstraZeneca ELN are actually BH; the
  strongest public options are amide coupling, C–H borylation, SNAr, Suzuki. Testing a
  BH-literature-trained model cross-class demonstrates only *generic* feasibility-learning, not
  BH-specific transfer.
- **(b) granularity is uneven:** HTE sets (HiTEA, LSF, Chemlex-measured portion, SNAr) give real
  measured failures; ELN/parallel-library "no product" (AstraZeneca, AbbVie) risk missing-not-failed.
- **Readout quality varies:** SNAr = MS-intensity threshold; LSF = LCMS ≥1%; not calibrated yield.
- **(c) is only strongly verified for LSF** (literature-median plate) and AbbVie/AstraZeneca (real
  practice); the amide HTE sets use narrowed standardized conditions.

## Primary sources

- Chemlex amide — https://www.nature.com/articles/s41467-025-59812-0
- Pfizer HiTEA — https://www.nature.com/articles/s41557-023-01393-w · https://github.com/emmaking-smith/HiTEA
- ETH/Roche LSF borylation — https://www.nature.com/articles/s41557-023-01360-5 · https://github.com/ETHmodlab/lsfml
- Jaman SNAr — https://pmc.ncbi.nlm.nih.gov/articles/PMC12043022/
- AstraZeneca ELN BH — https://pubs.rsc.org/en/content/articlehtml/2023/sc/d2sc06041h
- AbbVie Suzuki — https://pubs.acs.org/doi/10.1021/jacs.4c00098
- RSC 2025 amide — https://pubs.rsc.org/en/content/articlelanding/2025/sc/d5sc03364k
