# Cernak group — 50,688-Reaction C–N Coupling Data Set (key literature/HTE data source)
- **Citation:** J. Das, X. Zhang, Y. Tan, M. Abdelalim, T. Lu, C. Mauro, C. J. Regan, T. Cernak. *A 50,688-Reaction
  Data Set Reveals General Ligands and Mechanistic Diversity in C–N Couplings.* *J. Am. Chem. Soc.* 2026, 148 (25),
  DOI 10.1021/jacs.6c05959. (Cernak group, Univ. Michigan.)
- **Type:** data (HTE) — the "Cernak ~50k C–N" set the task asked us to find; **already downloaded locally.**
- **File:** literature note only; data at `datasets/raw/Cernak/50K_dataset_v2.csv` (+ `morpholine_yield_v1.csv`,
  `piperidine_yield_v1.csv`). Verified counts below are from the local CSVs.
- **TL;DR:** A very large, condition-dense **informer** C–N screen — only ~4 electrophiles × 2 nucleophiles but ~163 ligands
  across Pd/Ni/Cu, many bases/solvents — built to find **general ligands** and reveal mechanistic diversity; a rich source
  of literature **positives** (and true negatives) for BH C–N, but with almost no substrate diversity.

## Dataset(s) — scope, size, conditions, feasibility definition (VERIFIED from local file)
- **50,688 reactions** in `50K_dataset_v2.csv`.
- **Substrates: only 4 electrophiles × 2 nucleophiles** (an informer library, NOT a substrate-scope screen):
  - Electrophiles: 4-bromobenzotrifluoride, 3-bromopyridine, 4-iodobenzotrifluoride, 3-iodopyridine.
  - Nucleophiles: morpholine (33,792 rxns), 4-phenylpiperidine (16,895 rxns).
- **Conditions (the real diversity):** ~163 ligands / 166 ligand names, **34 metal precatalysts** (`metal_name` 35 unique →
  Pd/Ni/Cu), **17–18 bases** (NaOtBu, NaOTMS, KOtBu, K3PO4, Cs2CO3, P2Et/BTMP phosphazenes, CsOAc, LiHMDS, K2CO3, DBU, …),
  **4–5 solvents** (DMSO 29,183; diglyme 15,360; xylene/DMSO and diglyme/DMSO mixes), plus temperature.
- Yield columns are **fractional 0–1** assay yields (`Assay Yield of Product1`); ~70.8% are 0 (no product) → **negative-rich**.
  For our binary feasibility, apply the same **10% (=0.10) threshold** used by JnJ. (`morpholine_yield_v1.csv` /
  `piperidine_yield_v1.csv` are per-nucleophile yield tables for the two informer amines.)

## Method & features (of the paper)
- Systematically varied C–N couplings designed to **maximize overlap across Pd-, Ni-, and Cu-catalyzed** conditions so that
  "counterfactual" single-variable comparisons are possible; analysis surfaces **general ligands** and distinct mechanistic
  regimes. (Modeling emphasis is dataset-design + ligand generality rather than a new architecture.)

## Relevance to our pipeline (esp. leakage risk vs JnJ test set)
- **Value = a huge, condition-diverse pool of BH/C–N POSITIVES and real NEGATIVES** to mine for our PU literature-positive set
  and to benchmark condition (ligand/base/solvent) effects — especially generality of ligands, complementary to JnJ's 4
  general conditions.
- **Leakage vs JnJ (VERIFIED):**
  - **morpholine and 4-phenylpiperidine are NOT in JnJ's 68 amines** → the two Cernak nucleophiles don't appear in the test.
  - **3-bromopyridine (`Brc1cccnc1`) IS one of JnJ's 110 aryl halides** → aryl-identity overlaps on that one substrate.
  - Cernak has only **8 substrate pairs**, and **none coincide with JnJ's 888 measured pairs** → **pair-level leakage = 0**.
  - So Cernak is safe to use as positives at the pair level; if we do **aryl-OOD**, exclude/track 3-bromopyridine (and the
    two iodo/bromo benzotrifluorides if any JnJ aryl matches). Re-verify with **RDKit-canonical** SMILES before final sign-off
    (the checks above used raw-string SMILES + a couple of hand spellings).
- **Caveat:** because Cernak varies conditions on a tiny substrate set, its positives teach **condition feasibility**, not
  **substrate-pair feasibility** — good for learning "which ligand/base works" but weak for substrate generalization. Weight
  accordingly, or it will bias the PU model toward its 4 aryls / 2 amines.

## Open questions for us
- Use Cernak positives to learn a **condition-feasibility prior** (which of the 4 JnJ general conditions is most general) while
  relying on broader-substrate sources (Neves' 7 sources, ORD, USPTO) for substrate generalization?
- Do the Pd-only rows suffice, or do the Ni/Cu rows help/hurt a Pd-focused BH feasibility model? (Filter by metal.)
- Confirm the exact yield scale + how "no product / decomposition" is coded before thresholding at 0.10.

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Ahneman/Doyle. Predicting C–N performance with ML. *Science* 2018, 360, 186. DOI 10.1126/science.aar5169 🔒
- Rinehart/Doyle. Substrate-adaptive Pd C–N conditions. *Science* 2023, 381, 965. DOI 10.1126/science.adg2114 🔒
- Dataset Design for Building Models of Chemical Reactivity. *ACS Cent. Sci.* 2024, DOI 10.1021/acscentsci.3c01163 🔓
