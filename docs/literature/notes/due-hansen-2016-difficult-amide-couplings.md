> **Provenance.** Summary written by a Claude agent on 2026-10-21 from the full PDF (4 pages, pp. 430–433). **Not yet checked by Luka.** The ESI (synthetic procedures, characterisation) was not available to me; everything below comes from the main text. Read the paper before citing a detail.
>
> **Reference.** M. E. Due-Hansen, S. K. Pandey, E. Christiansen, R. Andersen, S. V. F. Hansen, T. Ulven, "A protocol for amide bond formation with electron deficient amines and sterically hindered substrates", *Org. Biomol. Chem.* 2016, **14**, 430–433, doi:[10.1039/c5ob02129d](https://doi.org/10.1039/c5ob02129d). Licensed CC-BY 3.0.

# Due-Hansen 2016: which amide couplings fail, and why

A 4-page communication. One hard model reaction, a reagent screen, and a 21-compound scope. Relevant to us because it is a clean, explicit list of *failure* classes for acid × amine coupling, including HATU.

## 1. The difficult classes they name

| class | example substrate | stated reason |
|---|---|---|
| electron-poor heteroaryl amine | **1b**, *N*-cyclopropyl-2-aminothiazole ("electron poor and sterically hindered amine", p. 430 col. 2); 2-amino-5-nitropyridine, 2-amino-4,6-dimethylpyrimidine ("poor nucleophiles", p. 431 col. 2) | low N nucleophilicity |
| hindered acid | **1a** (α-benzyl acid bearing a CO₂tBu); 2-(4-chlorophenyl)-3-methylbutanoic acid; diphenylacetic acid; triphenylacetic acid; adamantane-1-carboxylic acid; mesitylcarboxylic acid | steric hindrance at the acid carbon (pp. 430–432) |
| hindered secondary aliphatic amine | diisopropylamine; 2,2,6,6-tetramethylpiperidine (TMP) | steric; TMP **failed even with their method** (p. 432, col. 2) |
| hindered + electron-poor aniline | 2,5-dichloroaniline with mesitylcarboxylic acid | combination of the two; **failed**, acyl fluoride recovered (p. 432, col. 2) |
| acid-labile product/substrate | the tBu ester in **1a**; amide **1** itself | tBu ester "sensitive to forcing conditions" (p. 430); amide **1** hydrolyses back to **1a** + **1b** on exposure to even acetic acid (p. 431, col. 1) |
| arylacetic acid with no second α-substituent | phenylacetic acid, indole-3-acetic acid, benzothiophene-3-acetic acid | **failed** with their method; cause not investigated, "might be related to decomposition via ketene formation" (p. 432, col. 2) — a *side reaction*, not low nucleophilicity |

No tautomer argument is made anywhere in the paper.

## 2. What fails with HATU and other standard reagents

Table 1 (p. 430), all on **1a + 1b**, conversion by HPLC. Solvent for entry 1 is anhydrous DMF (stated in the text, p. 430 col. 2); solvents for entries 2–5 are not given in the table.

| entry | reagent(s) | temp | time | conv. |
|---|---|---|---|---|
| 1 | HATU / DIPEA (DMF) | rt | 4 d | **17 %** |
| 2 | EDC / HOBt | rt | 24 h | **0 %** |
| 3 | DCC / DMAP | 40 °C (reflux) | 5 d | **0 %** |
| 4 | Ghosez's reagent (→ acid chloride) | rt | 4 d | **9 %** |
| 5 | acid chloride / AgCN | rt | 24 h | **decomposition** |

The HATU/DIPEA/DMF/4 d condition is the one taken from the patent literature (ref. 4b); the authors could not reproduce it (p. 430).

Second HATU-free comparison (p. 432, col. 1): standard **EDC/HOBt** on the scope compounds gave **38 % conversion for 7** and only **traces of 8–10** — i.e. the hindered-acid × electron-poor-heteroarylamine pairs.

## 3. What they recommend instead

In-situ acyl fluoride with **BTFFH** (bis(tetramethylene)fluoroformamidinium hexafluorophosphate), two steps in one pot (Table 3 footnote a, p. 431):

- (i) acid 1.3 eq, **BTFFH 1.5 eq**, **DIPEA 4.5 eq**, **CH₂Cl₂ 2 mL mmol⁻¹**, **rt, 30 min** (pre-activation to the acyl fluoride);
- (ii) add amine (1 eq, 0.5 mmol), **80 °C, 12–24 h**.

Mechanistic reason (p. 430, col. 2): the acyl fluoride is small, so it "minimised steric hindrance between the coupling partners", while still being reactive towards amines and more stable/less hydrolysis-prone than an acid chloride. BTFFH over TFFH because TFFH forms toxic by-products (p. 431, col. 1). CH₂Cl₂ over DMF because acid-fluoride formation is faster and less prone to hydrolysis there (p. 431, col. 1).

Optimisation, Table 2 (p. 431), all on **1a + 1b**, conv. by HPLC / isolated yield:

| entry | solvent | T (°C) | t (h) | conv. | yield |
|---|---|---|---|---|---|
| 1 | DMF | rt | 24 | 8 % | n.d. |
| 2 | DMF | 80 (MW) | 2.5 | 43 % | 20 % |
| 3 | CH₂Cl₂ | rt | 24 | 23 % | n.d. |
| 5 | CH₂Cl₂ | 80 (MW) | 4 | 85 % | 61 % |
| 7 | CH₂Cl₂ + 0.2 eq DMAP | 80 (MW) | 4 | 63 % | 35 % |
| 9 | CH₂Cl₂ | 100 (MW) | 7 | 96 % | 78 % |
| 11 | CH₂Cl₂ | 80 (sealed vial) | 24 | **100 %** | **85 %** |

Omitted rows: entry 4 (TFFH, rt, 24 h, 21 %), entry 6 (DCE reflux 84 °C, 24 h, 50 %), entry 8 (CH₂Cl₂, 80 °C MW, 7 h, 94 %), entry 10 (CH₂Cl₂, 140 °C MW, 3 h, 87 %, 75 % isolated). DMAP *lowers* the yield (entry 7), ascribed to the bulkier N-acyl-DMAP intermediate (p. 431, col. 1).

Scope, Table 3 (p. 431), isolated yields: **1** 83 %, **2** 80 %, **3** 86 %, **4** 88 %, **5** 81 %, **6** 79 %, **7** 88 %, **8** 61 %, **9** 93 %, **10** 88 %, **11** 71 %, **12** 77 %, **13** 55 %, **14** 73 %, **15** 91 %, **16** 96 % (>99 % de), **17** 97 % (>99 % de), **18** 78 %, **19** 50 %, **20** 90 %, **21** 86 %. No α-racemisation with N-Boc-L-proline (**16/17**, p. 432).

## 4. Side reactions reported

- **No guanidinylation / N-capping by a uronium reagent is reported anywhere in this paper.** HATU appears only once, as Table 1 entry 1, with no side-product analysis.
- Fluorination reagents DAST, Deoxo-Fluor and XtalFluor-E release a secondary amine (diethylamine, bis(methoxyethyl)amine) that competes with the substrate amine (p. 430, col. 2; p. 432, col. 1). Concrete case: octanoic acid + *tert*-pentylamine by XtalFluor-E, **8 % yield**, "due to reaction with the released diethylamine" (ref. 12, p. 432) — BTFFH gave **91 %** (**15**).
- Acid chloride + AgCN: **decomposition** (Table 1, entry 5). Suspected **ketene formation** from arylacetyl fluorides lacking a second α-substituent (p. 432, col. 2). Over-hindered pairs: unreacted **acyl fluoride recovered** (p. 432, col. 2).
- Product instability: amide **1** hydrolyses back to **1a** + **1b** with weak acids such as acetic acid (p. 431, col. 1).

## 5. Would these reach ≥ 20 % in 30 min at 30 °C (our Chemlex benchmark condition)?

What the paper shows directly:

- HATU/DIPEA/DMF at rt for **4 days** gave **17 %** on **1a + 1b** (Table 1, entry 1) — already below our 20 % cut-off after 96 h.
- BTFFH in DMF at rt for **24 h** gave **8 %** (Table 2, entry 1); in CH₂Cl₂ at rt for 24 h, 23 % (entry 3). Every synthetically useful result needed **80–140 °C for 3–24 h** (Table 2, entries 5–11; Table 3 footnote a).

My inference (not in the paper): with rate falling off that steeply, **30 min at 30 °C in DMF would give essentially 0 %** for every substrate class listed in §1. So substrates of this kind are near-certain Chemlex negatives, and Chemlex "failure" for them means "too slow at 30 °C in 30 min", not "impossible". The paper gives no kinetics, so this is an extrapolation.

## 6. Quantitative structure-reactivity content

Thin. There are **no pKa values, no Hammett constants and no computed steric parameters** in this paper. What is quantitative is the condition dependence (Table 2: rt 23 % → 80 °C/4 h 85 % in CH₂Cl₂) and the reagent comparison (EDC/HOBt 38 % for **7**, trace for **8–10**, p. 432). The usable structural statements are categorical:

amines called "poor nucleophiles" are all **2-amino-azines/azoles** — an sp² ring N α to the amino group (p. 431); the acid failure rule is **arylCH₂COOH with no second α-substituent** fails while arylCR₂COOH works (p. 432); the hardest amine found is **2,2,6,6-tetramethylpiperidine**, two quaternary α-carbons, which no method here couples.

## 7. Three things that bear on our taxonomy

1. **Our `heteroaryl_amine` class is too coarse.** The paper's "poor nucleophiles" are specifically amines on a ring carbon **adjacent to a ring sp² nitrogen** (2-aminothiazole, 2-aminopyrimidine, 2-aminopyridine): §1, p. 431. A 3-aminopyridine or a 4-aminopyridine (compound **13**, 55 %, p. 431/432) is a different beast. Worth splitting `heteroaryl_amine` into *α-to-ring-N* vs *not*, which is a cheap SMARTS at the reacting N.
2. **We have no acid-side class, and the paper's two sharpest failures are acid-side.** (a) steric bulk at the acid α-carbon (di-/triphenylacetic, mesitylcarboxylic, adamantoyl); (b) **arylacetic acid without a second α-substituent** — a side-reaction failure (ketene), orthogonal to sterics, p. 432. Both are SMARTS-expressible at the carboxyl carbon. An acid taxonomy of roughly {aryl-CH₂-COOH, α-branched/α-aryl-disubstituted, α-quaternary/neopentyl-like, aliphatic unhindered, aryl-COOH (incl. ortho-disubstituted)} would be supported by this paper alone.
3. **Interaction, not independent classes.** The only pairs that failed outright are *hindered acid × electron-poor or hindered amine* (2,5-dichloroaniline + mesitylcarboxylic acid; TMP; p. 432), while each partner alone is fine with a better counterpart. Our model needs the acid × amine cross term, not two independent one-hot vectors. Note also that `aniline` as one bucket lumps plain aniline with 2,5-dichloroaniline, which the paper treats as a genuinely hard case.
