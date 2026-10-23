> **Provenance.** Summary written by a Claude agent on 2026-10-21 from the full PDF (15 pages: cover + article pp. 11809–11822). **Not yet checked by Luka.** The ESI was not bound into this PDF and was not read; every "Table S…" claim below is only what the main text says about it. Read the paper before citing a detail.
>
> **Reference.** C. Zhang, Q. Lin, C. Yang, Y. Kong, Z. Yu, K. Liao, "Intermediate knowledge enhanced the performance of the amide coupling yield prediction model", *Chem. Sci.* 2025, **16**, 11809–11822, doi:[10.1039/d5sc03364k](https://doi.org/10.1039/d5sc03364k). Licensed CC-BY-NC 4.0 (p. 11809).

# Zhang 2025 — intermediate knowledge for amide-coupling yield prediction

Citation key: **p.X** = printed journal page. "Not stated" = I found nothing on it in the main article.

---

## 1. What "intermediate knowledge" is

- A **reaction SMARTS that converts the carboxylic acid into its activated intermediate**, applied before featurisation. For HATU (Fig. 5a, p. 11816):
  `[#6:1](=[#8:2])[#8:3]>>[#6:1]([#8:3]N1N=NC2=CC=CN=C21)=[#8:2]` — i.e. the acid becomes the HOAt ester.
- One such SMARTS per condition; all six are "detailed in the code repository" (p. 11815). The acid itself is **replaced**, not augmented: "This template converts the acid into the activated acid" (p. 11815).
- Three descriptor patterns were compared (p. 11815): (1) no intermediate; (2) amine + acid + intermediate → amide; (3) amine + intermediate → amide.
- Pattern (3) usually wins, because "our intermediate is represented as an activated acid, it already encapsulates acid information, making the amine + acid + intermediate descriptors redundant" (p. 11817, pointing to Table S13).
- The intermediate replaces explicit condition encoding: in the selected-condition models "the reaction conditions were no longer concatenated with the reaction vector" (p. 11816). Example reaction SMILES in Fig. 5b: `CN.CC(=O)ON1N=NC2=CC=CN=C21>>CNC(C)=O`.
- It is **not** a new per-substrate descriptor. The featuriser stays the same (1024-bit ECFP r=2 for the tree/ensemble models, tokenised reaction SMILES for BERT/T5); only the molecule fed in changes.

## 2. Dataset

- Substrate selection, not reaction mining: USPTO-50k filtered by an amide SMARTS → 11,663 amide-coupling entries (p. 11811); products → 1024-bit ECFP r=2 → PCA → K-means (10 clusters) → virtual-compound filtering at max distance 1.37 → stratified sampling over 10 strata (pp. 11811–11812).
- Final library: **632 products = 632 unique substrate pairs, from 70 amines and 66 acids** (447 machine-selected + 186 expert-selected) (p. 11812). A 71,000-product virtual space was used only to size the selection (p. 11812).
- **95 conditions** (Table S1); all commercial coupling reagents except acyl chloride; **DMF only**, so substrates with poor DMF solubility were excluded (pp. 11812–11813).
- **> 47,000 yield data points** collected, after discarding runs where chromatogram overlap or NMR difficulty blocked quantification (p. 11813). Analysis by UPLC, MS and NMR (p. 11813).
- Six conditions for the intermediate study (Table 2, p. 11815): HATU/DIPEA/DMF (#1), TBTU/DIPEA/DMF (#6), EDC·HCl/HOBT/DIPEA/DMF (#13), HBTU/DIPEA/DMF (#21), PyBOP/DIPEA/DMF (#34), DCC/HOBT/DMF (#79).
- Target is **continuous yield**, regression (R², MAE, RMSE). No binary label, no positive/negative class, so **no class imbalance is defined or reported** — the one exception is a 0.73 binary accuracy on reactivity-cliff pairs (p. 11820).

## 3. Splits

Three levels (Fig. 4, p. 11814): **random**, **partial substrate novelty** (at least one substrate unseen; cases with both substrates seen are excluded), **full substrate novelty** (entirely new substrate *combinations* the model never met). Average pairwise Tanimoto between training and random-split test products: **0.20** (p. 11815).

Note: "full substrate novelty" here means unseen *pairs* built from a 70 × 66 substrate set, not unseen substrates in our sense — see §8.

**Table 1 (p. 11815), multi-condition models, R² (MAE / RMSE):**

| split | XGBoost | SVM | RF | AutoGluon | Yield-BERT | T5-Chem |
|---|---|---|---|---|---|---|
| random | 0.32 (18/22%) | 0.25 (19/23%) | 0.35 (17/21%) | 0.55 (15/20%) | **0.66** (10/15%) | 0.53 (16/22%) |
| partial novelty | 0.26 (4/20%) | 0.23 (16/21%) | 0.26 (14/19%) | 0.66 (13/18%) | **0.68** (14/10%) | 0.58 (20/15%) |
| full novelty | 0.25 (20/24%) | 0.22 (22/27%) | 0.26 (19/23%) | 0.42 (17/22%) | **0.63** (15/11%) | 0.58 (22/17%) |

(The XGBoost partial-novelty MAE of 4% sits beside an RMSE of 20% and is almost certainly a typo in the paper.)

**Table 4 (p. 11817), BERT on the pooled six-condition set, R² (RMSE / MAE):**

| intermediate | random | partial novelty | full novelty |
|---|---|---|---|
| without | 0.77 (12/9%) | 0.71 (14/10%) | 0.62 (10/8%) |
| with | 0.85 (10/7%) | 0.80 (11/8%) | 0.65 (9/8%) |

## 4. Model comparison

- Transformer (Yield-BERT) beats XGBoost-on-ECFP everywhere, and the gap widens with novelty: 0.66 vs 0.32 random, 0.63 vs 0.25 full novelty (Table 1, p. 11815).
- **Intermediate knowledge helps BERT, not XGBoost.** "The inclusion of intermediate descriptors did not enhance results, resulting in marginal declines" for XGBoost (p. 11817, Table S13); same for the other classical algorithms. Explanation offered: XGBoost "struggles with the complexity added by intermediate descriptors" (p. 11817).
- External literature test, 257 reactions from med-chem journals, substrates entirely distinct from the HTE set (p. 11818): **R² 0.71, MAE 7%, RMSE 10%, MSE 1%** (Fig. 6, p. 11817).
- Isayev benchmark (ref. 16) under a random split: plain BERT R² 0.37 (MAE 13%, RMSE 18%); with intermediate knowledge R² 0.42 (MAE 12%, RMSE 16%) (p. 11820).
- Reactivity cliffs (similarity > 0.9, yield gap > 30): mean prediction error 0.34, but binary classification accuracy 0.73 (Fig. 9 and text, p. 11820). The worked cliff example mispredicts badly: experiment 97 %, prediction 58 % (Fig. 9b, p. 11820).

## 5. Substrate categories and per-class results

- **No substrate classification scheme is used anywhere.** No amine-type or acid-type labels, no per-class metrics, no per-class success rates.
- Chemistry is referenced only narratively: aromatic amines with weak nucleophilicity are called the hard case (p. 11809), and a functional-group tolerance list is given from the external set — alkyne, azide, hydroxyl, halide, carboxyl, phosphonate, aldehyde, heterocycles, sulfonamide (p. 11819), with compound numbers but no numbers per group.
- The only class-shaped remark is a limitation: DMF-solubility exclusion "may limit the chemical diversity of certain subclasses of acids or amines" (p. 11812).

## 6. Ablations and generalisation evidence

- **Main ablation = with vs without intermediate**, per condition (Table 3, p. 11816), BERT R², value with intermediate (value without):

| split | TBTU | HATU | PyBOP | DCC | HBTU | EDC |
|---|---|---|---|---|---|---|
| random | 0.84 (0.71) | 0.86 (0.69) | 0.90 (0.80) | 0.86 (0.80) | 0.89 (0.83) | 0.89 (0.82) |
| partial novelty | 0.77 (0.57) | 0.78 (0.53) | 0.82 (0.63) | 0.81 (0.74) | 0.86 (0.72) | 0.88 (0.79) |
| full novelty | 0.85 (0.66) | 0.84 (0.39) | 0.89 (0.40) | 0.67 (0.10) | 0.83 (0.68) | 0.75 (0.46) |

- The gain is **largest exactly where novelty is highest**: HATU 0.39 → 0.84, PyBOP 0.40 → 0.89, DCC 0.10 → 0.67 on full novelty (Table 3, p. 11816). This is their central generalisation claim.
- Second ablation: descriptor pattern (no intermediate / amine+acid+intermediate / amine+intermediate), amine+intermediate best (p. 11817, Table S13).
- Third ablation: one-hot encoding of the conditions "significantly decreased" performance (p. 11817, Section S3.10). Fingerprints or reaction SMILES for the conditions "did not represent the conditions well", so conditions were encoded as a single integer 1–95 (p. 11813, p. 11814).
- Pooling the six conditions into one model gains less than the per-condition models, "especially in cases of complete novelty splitting" (p. 11817, Table 4).

## 7. Code and data

- `https://www.github.com/aichemeco/amide_coupling/tree/main` (Data availability, p. 11820). Both data and code are said to be there, including the six condition SMARTS (p. 11815).
- Article licence CC-BY-NC 4.0 (p. 11809). The repository licence is **not stated** in the paper; check it before reusing anything.

## 8. What this means for our features — and what argues against us

**(a) Encode the activated acid, not the acid.** Their one representation change (acid → HOAt ester for HATU, Fig. 5a p. 11816) moved full-novelty R² from 0.39 to 0.84 for HATU (Table 3, p. 11816). Our conditions are fixed at HATU/DIPEA/DMF, so we can apply exactly their HATU SMARTS and fingerprint the activated ester instead of the acid. Cheap to test.

> **Added by the main session, 2026-10-21 (not from the paper).** I ran their HATU template over the 69
> Chemlex validation acids. It failed to apply to 28 of them; on the other 41 it added 27 bits that are set
> in *every* product (zero variance, the constant HOAt fragment) and raised the mean pairwise Tanimoto
> between acids from 0.145 to 0.384. Under a single fixed condition the template is the same transformation
> for every acid, so it cannot add information and it measurably compresses the differences between acids.
> That is consistent with (c) below. We are not adopting it; see
> `2026-10-21-subclasses-and-features-research.md`.

**(b) Their result argues the amine carries the acid-side signal once the intermediate is in.** "amine + intermediate" beat "amine + acid + intermediate" (p. 11817) — the activated acid is a sufficient statistic for the acid. That is an argument for not adding more acid-side descriptors before trying the intermediate.

**(c) Their result argues *against* the model family we would pair it with.** The intermediate trick gave XGBoost "marginal declines" (p. 11817). Benefit (a) is documented only for a transformer on tokenised reaction SMILES. On ECFP + a tree/linear model — closer to our Morgan-count setup — there is no evidence in this paper that it helps at all.

**Scepticism / where their setting differs from ours:**
- **Regression on continuous yield vs our ranking of a binary ≥ 20 % conversion label.** R² on a well-populated yield distribution says nothing directly about top-of-list precision.
- **"Full substrate novelty" is not our "both substrates unseen".** Their test pairs are new *combinations* drawn from the same 70 amines and 66 acids (p. 11812, Fig. 4a p. 11814); every individual substrate was seen in training. Our Chemlex validation holds out both substrates entirely. Their 0.84 is therefore a much weaker novelty test than ours — the closest comparison to our setting is the 257-reaction literature set (R² 0.71, p. 11817), where substrates were "entirely distinct" (p. 11818).
- **No positive-unlabeled anything.** Their negatives are real measured low yields from HTE; ours are inferred. Nothing here transfers about PU.
- **Tiny substrate set.** 70 amines × 66 acids, against our ~285k inventory pairs. Their intermediate feature may simply be helping a model memorise a 136-substrate vocabulary.
- **Nothing supports or attacks our 5-level amine subclass directly** — they use no classes at all (§5). So this paper is no evidence that subclasses help, and no evidence they hurt; it is a reason to also test the orthogonal idea (intermediate SMARTS) rather than to add more subclass levels.
