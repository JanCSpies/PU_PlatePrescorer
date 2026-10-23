> **Provenance.** Summary written by Claude agents and the main session on 2026-10-21 from the notes in this folder, the Brotzel dissertation, the PDFs named inline and our own counts on the Chemlex HATU validation set. Not yet checked by Luka. Read a paper before citing a detail. Claims tagged **[unverified]** were not confirmed against the paper.

# Amine subclasses and features: why we questioned them, what we found

## Question

Stage F (2026-10-21) added amine subclasses anchored at the reacting N and 9 reacting-centre descriptors. Before building stage S we asked three things. Is the taxonomy chemically right? Are the features the right ones? How should a categorical class enter a PU model and its evaluation?

## What we measured ourselves

Chemlex HATU validation, 1,725 reactions, 32.8 % positive. Test side never opened. Unless a line says otherwise, every number below comes from one command:

    uv run python scratch/count_subclass_evidence.py ~/University/Phd/Dataset/Datasets

The script asserts that the validation file contains no `both_unseen == "test"` rows, so the test side stays closed. The Gasteiger/MMFF correlation and the 3D vdW-occupancy AUC (0.733) come from `uv run python scratch/count_design_rule_proxy.py` (no argument).

| amine subclass | success rate | distinct amines |
|---|---|---|
| aliphatic_primary | 0.543 | 11 |
| aliphatic_secondary | 0.457 | 14 |
| aniline | 0.177 | 16 |
| heteroaryl_amine | 0.075 | 7 |
| other | 0.252 | 5 |

| alpha_max_heavy_degree | 2 | 3 | 4 |
|---|---|---|---|
| success, amine | 0.63 | 0.29 | 0.16 |
| success, acid | 0.37 | 0.34 | 0.14 |

| zero-training score on validation | ROC-AUC |
|---|---|
| popularity | 0.54 |
| amine subclass | 0.713 |
| single descriptor, amine n_heavy_at_2_bonds | 0.711 |
| 3D occupied vdW volume, 5 A sphere at amine N | 0.733 |
| subclass + topological amine sterics (rank sum) | 0.775 |

- Inside a subclass I split at the median of topological amine sterics. Success below vs above the median: aniline 0.065 vs 0.291, heteroaryl 0.010 vs 0.136. aliphatic_primary is reversed, on only 36 reactions. So sterics carries signal inside a class.
- **Caveat.** I chose the 0.775 combination after looking at validation. It is optimistically biased. It counts as a baseline only once it is written down in advance.
- The raw Chemlex file does not flag the rule-designed negatives. It has Acid, Amine, Reagents, Solvent, Products, Conversion and three split columns. A score stratified by "designed vs diversity" rows is therefore not possible from the released file.
- Reagent families, 11,669 rows: HATU + DIPEA 8,454 (both HATU drawings), TCFH 1,339, PyBOP 949, EDCI 915, PyBrOP 7, BOP 5 (`count_subclass_evidence.py`).
- Gasteiger charge does not stand in for the Hirshfeld charge of the Chemlex design rule. Rank correlation with MMFF charges on the amine N is -0.11. Its scale (about -0.3) does not carry their -0.14 threshold (`count_design_rule_proxy.py`).

## Chemistry of the taxonomy

- **Mayr N in MeCN.** Brotzel, Chu, Mayr, J. Org. Chem. 2007, 72, 3679, doi:10.1021/jo062586z. Read via the Brotzel dissertation; values from the Mayr database.

| amine | N |
|---|---|
| pyrrolidine | 18.58 |
| piperidine | 17.35 |
| morpholine | 15.65 |
| n-PrNH2 | 15.11 |
| Et2NH | 15.10 |
| BnNH2 | 14.29 |
| iPrNH2 | 13.77 |
| aniline | 12.64 |
| tBuNH2 | 12.35 |
| CF3CH2NH2 | 10.13 |

  Cyclic secondary amines are the strongest nucleophiles. "Primary beats secondary" is not supported. An alpha-tertiary primary amine sits at aniline level; an alpha-EWG primary amine sits below it.
- **Hammett.** rho for anilines against benzhydrylium ions is -2.1 in water and -2.5 in MeCN (same source). Within one amine family basicity tracks rate. Across families pKaH is a poor predictor.
- **Due-Hansen et al.**, Org. Biomol. Chem. 2016, 14, 430, doi:10.1039/c5ob02129d (note `due-hansen-2016-difficult-amide-couplings.md`).
  - HATU/DIPEA/DMF, room temperature, 4 days: 17 % for a hindered acid with N-cyclopropyl-2-aminothiazole.
  - 4-Aminopyridine couples (55 %). 2-Amino-azines fail. This supports our rule that heteroaryl means "amino on a ring carbon adjacent to a ring heteroatom". 4-Aminopyridine correctly falls to aniline.
  - The sharpest failures are on the acid side: hindered acids, and arylacetic acids with no second alpha substituent (a suspected ketene route that no steric descriptor captures). Failures are interactions of a hindered acid with a weak amine.
- **Haas et al.**, PNAS 2022, 119, e2118451119, doi:10.1073/pnas.2118451119. For CDI couplings with primary alkyl amines the acid largely sets the rate. The leading descriptors are the NBO charge and %Vbur(2.0 A) at the carbonyl carbon, over five orders of magnitude in rate. We give acids no class at all.
- **Consequence for reading Chemlex.** Its label is "fast enough in 30 min at 30 C", not "feasible". A substrate class that is merely slow is a Chemlex negative.

## Features

- **Liu, Moroz, Isayev**, Chem. Sci. 2023, 14, 10835, doi:10.1039/d3sc03902a. On 41k literature amide couplings, QM descriptors, buried volume and AEVs added nothing over a whole-molecule Morgan fingerprint with a random forest. R2 differs by amine type (0.36-0.43).
- **Zhong et al.** (Chemlex), on their own both-unseen split: DRFP 0.79, GNN 0.79, rxnfp 0.81. The archived prototype's unverified 0.768 sits at this in-domain supervised ceiling. That is a reason to look for leakage when we reproduce it.
- **Zhang et al.**, Chem. Sci. 2025, 16, 11809, doi:10.1039/d5sc03364k (note `zhang-2025-amide-intermediate-knowledge.md`).
  - Replacing the acid by its activated ester before featurisation lifts a transformer from R2 0.39 to 0.84 at full substrate novelty.
  - It made XGBoost on ECFP worse.
  - Their "full novelty" means unseen combinations of seen substrates.
  - Our check on the 69 validation acids: the HATU template adds 27 zero-variance bits and raises the mean pairwise Tanimoto between acids from 0.145 to 0.384. It makes acids look more alike. The template failed to apply to 28 of 69 acids (`count_subclass_evidence.py`).
  - We are not adopting it.
- A whole-molecule Morgan count sum is about (radius+1) x heavy atoms, so it encodes size. Size is our largest domain shift (`count_subclass_evidence.py`):

| mean heavy atoms | P | library | validation |
|---|---|---|---|
| acids | 20.6 | 13.2 | 21.7 |
| amines | 17.2 | 10.6 | 19.8 |

 I keep n_heavy_atoms for stratification and a size-only baseline, not as a model feature.
- Our cheap topological descriptors already match a 3D buried-volume descriptor on validation (0.711 vs 0.733). No conformer generation and no new dependency is needed for sterics for now.

## PU and evaluation

- **Within-class AUC and reporting bias.** If the probability that a success is reported depends only on the amine class, each class is multiplied by its own constant and the ranking inside a class is unchanged. Within-class AUC is then robust to class-level reporting bias; pooled AUC is not. This is derivation by the research agent from the SAR definition. Under SCAR it is the Elkan and Noto rank equivalence (KDD 2008, doi:10.1145/1401890.1401920). **Unverified reasoning, not a cited result.**
- **The baseline has a hole.** The amine-class baseline gives every amine in a class the same score, so its within-class AUC is exactly 0.5. ADR 006 item 3 as written therefore asks the model only to beat chance inside a class. This is the main reason for the coming ADR 007.
- **Bekker, Robberechts, Davis**, ECML-PKDD 2019, arXiv:1809.03207. The propensity may depend on fewer attributes than the classifier. With a discrete propensity attribute one can stratify, and inside each stratum SCAR methods apply. A class-prior assumption is still needed per stratum.
- **Encoding.**
  - One-hot for scikit-learn models.
  - CatBoost: declare the class in `cat_features` with `one_hot_max_size` at least 5. The CPU default of 2 would otherwise build target statistics, and the statistic for a class is that class's P-vs-U rate, i.e. reporting bias, not reactivity.
  - Never pass substrate identity as a categorical feature; its target statistic is the substrate's popularity.
  - Monotonic constraints do not work on a categorical feature, and would constrain the P-vs-U score, not reactivity.
  - CatBoost settings: **[unverified, from the research agent; check the CatBoost docs]**.
- **Evaluation recipe for ADR 007.**
  - Primary estimand: the pair-weighted within-class AUC.
  - Uncertainty: Owen's pigeonhole bootstrap over the crossed acid and amine factors (Owen, Ann. Appl. Stat. 2007, 1, 386, doi:10.1214/07-AOAS122). Resample acids and amines separately, not reactions.
  - Paired differences on the same resample.
  - "Beats both baselines and passes in every class" is an intersection-union test. Each part may be tested at the same alpha, with no Bonferroni correction.
- **Power warning.** Validation has 5 to 16 distinct amines per class (table above), so per-class intervals will be wide.

## Decisions taken (Luka, 2026-10-21)

1. Add acid subclasses, after counting the candidates.
2. Count the proposed amine splits before deciding on them.
3. Supersede the evaluation part of ADR 006 with a new ADR 007. Accepted ADRs are never rewritten (CLAUDE.md section 8).
4. Stay with Chemlex as the validation benchmark for now, since it is our validation set. Record what its label actually measures (see "Consequence for reading Chemlex").

## Papers still missing

- El-Faham and Albericio, Chem. Rev. 2011, 111, 6557, doi:10.1021/cr100048w. HATU scope, guanidinylation, hindered and N-methyl couplings.
- Beutner et al., Org. Lett. 2018, 20, 4218, doi:10.1021/acs.orglett.8b01591. HATU versus TCFH/NMI on hindered acids with weak amines.
- The Org. Lett. PDF supplied on 2026-10-21 (doi:10.1021/acs.orglett.6c04074, Sharma et al., oxidation of H-phosphonate diesters by Oxyma-based reagents) is a different paper. It is not relevant to the taxonomy. Benzotriazole reagents such as HATU showed no oxidation there.
