# Angello, Burke et al. — Closed-Loop Optimization of GENERAL Reaction Conditions (heteroaryl Suzuki)
- **Citation:** N. H. Angello*, V. Rathore*, W. Beker, A. Wołos, E. R. Jira, R. Roszak, T. C. Wu, C. M. Schroeder,
  A. Aspuru-Guzik, B. A. Grzybowski, M. D. Burke. *Closed-loop optimization of general reaction conditions for heteroaryl
  Suzuki–Miyaura coupling.* **Science 2022, 378, 399–405** (note: 2022, not 2024). DOI 10.1126/science.adc8743.
- **Type:** method + application (defines the "general conditions" concept we build on)
- **File:** literature/science.adc8743 (1).pdf
- **TL;DR:** A closed loop of matrix down-selection + uncertainty-minimizing ML + robotic experimentation finds
  **general conditions** (one condition set maximizing average yield across a diverse substrate matrix) for heteroaryl
  Suzuki, **doubling** average yield vs the classic benchmark — by generating balanced positive+negative data on demand.

## Problem & contribution
"General conditions" (high average yield across many substrate pairs) are rare and valuable but live in an enormous
substrate×condition space. Prior literature-mining ML failed (Beker 2022) because reported conditions lack causal links to
structure and negatives are unpublished. Contribution: a de-novo closed loop that (i) **down-selects** both matrices to
tractable, representative subsets, (ii) uses **uncertainty-minimizing** ML to pick informative next experiments, (iii) uses
**robots** for reproducible data — and defines/optimizes an explicit **generality objective**.

## Dataset(s) — scope, size, conditions, feasibility definition
- ~5,400 purchasable (hetero)aryl halides → stratified clustering → **54 centroid halides × 54 MIDA boronates = 2,688**
  representative products; greedy-Tanimoto → **11 maximally-dissimilar substrate pairs** for the optimization loop.
- Conditions after down-selection: **3 solvents** (dioxane/toluene/DMF, 5:1 w/ water) × **2 bases** (Na2CO3, K3PO4) ×
  **2 temperatures** (60/100 °C) × **4 catalysts** (Pd SPhos-G4, XPhos-G4, P(tBu)3-G4, Pd(PPh3)4) = **528-reaction space**.
- Yields by automated LCMS-UV/Vis with authentic standards; **practical feasibility ≈ ≥10% isolable yield** (their success bar).

## Method & features
- **Generality objective:** f(c) = mean over substrates s of yield y(s,c); c_general = argmax_c f(c). Because evaluating f(c)
  needs ALL substrates, they do **BO over conditions C** (acquisition = probability of improvement) + **active learning over
  substrates S** (pick max-uncertainty substrate for a condition). Surrogate = **GP ensemble with an NN kernel, GPE(NN)**.
- Robotic Schlenk (10 vacuum/argon cycles) → ±2% yield reproducibility (vs ~10–15% human).

## OOD / splitting strategy & metrics
- Convergence when model uncertainty hits a calibration threshold (round 5); ~half the reactions of random sampling.
- **Generality validated OOD**: 20 diverse substrate pairs (Butina, dissimilar to training). Top ML condition doubled avg
  yield 21%→46% vs the JACS-2009 benchmark; **products >10% yield rose from 11/20 (benchmark) to 19/20 (top-1)**; product:by-product
  ratio 0.30→0.58 (p=0.0005); less protodeboronation, more halide conversion.

## Relevance to our pipeline (esp. leakage risk vs JnJ test set)
- **Foundational premise supplier:** justifies our "fix a small set of GENERAL conditions, then vary substrates" strategy.
  The 4 dominant JnJ P2Et sets (RuPhos/BrettPhos/XPhos/P(tBu)3-Pd-G4) are exactly the C–N analogue of these general conditions.
  → When we score the BH substrate grid, doing it **per general-condition set** (not per arbitrary condition) is the right,
  literature-grounded granularity.
- Reinforces the whole project's motivation (shared with TACIT/PAYN/Zhong/Neves): **published data is positive-biased and
  negative-poor**; here they fix it with robots-on-demand, we fix it with **PU/implicit negatives from literature** — a cheaper
  route to the same balanced signal.
- **Different reaction (Suzuki C–C), no BH/JnJ substrate overlap → no leakage.** Use conceptually, not as data.
- Generality metric f(c) = mean yield across substrates is a clean **plate-design objective**: rank our general-condition sets by
  predicted mean feasibility across the BH substrate library.

## Open questions for us
- Should our deliverable rank the 4 JnJ general-condition sets by predicted **generality** (mean feasibility across pairs),
  echoing Angello's f(c), to produce an actionable "best general condition" output?
- Their loop needs a robot; we're literature+prediction only — but their **acquisition logic** (BO over conditions + AL over
  substrates) is a template if we ever propose the next HTE plate.

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Beker, …, Burke, Grzybowski. Literature-mining for general Suzuki conditions (the failure they fixed). *JACS* 2022, 144, 4819. DOI 10.1021/jacs.1c12005 🔒
- Knapp, Gillis, Burke. JACS-2009 benchmark general condition. *JACS* 2009, 131, 6961. 🔒
- Shields et al. Bayesian reaction optimization (EDBO). *Nature* 2021, 590, 89. DOI 10.1038/s41586-021-03213-y 🔒
- Strieth-Kalthoff, …, Glorius. positive-bias in reaction data. *Angew.* 2022, 61, e202204647. 🔒
