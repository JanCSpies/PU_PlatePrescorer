# Planning chemical syntheses with deep neural networks (in-scope filter)
- **Citation:** Segler, M. H. S.; Preuss, M.; Waller, M. P. "Planning chemical syntheses with deep neural networks and symbolic AI." *Nature* **555**, 604–610 (2018). DOI: 10.1038/nature25978
- **Type:** method + literature-scale data mining (retrosynthesis; but the "in-scope filter" is a feasibility classifier)
- **File:** literature/nature25978.pdf
- **TL;DR:** Mines ~12.4 M literature reactions and trains a binary **feasibility ("in-scope") classifier** — but because failed reactions are absent from databases, it *manufactures* negatives from positives-only data; this is the exact PU/PN dilemma our project addresses, and its time-split protocol is a reusable leakage guard.

## Problem & contribution
3N-MCTS retrosynthesis: an expansion policy, a rollout policy, and an **in-scope filter** that predicts whether a proposed single step will actually work. Networks trained on **12.4 million single-step reactions from Reaxys**; rule extraction kept rules occurring ≥3 (expansion, 301,671 rules) or ≥50 times (rollout, 17,134 rules). The whole system is "purely data-driven … without tedious and biased expert encoding."

## Key findings (bias mechanisms / data availability)
- **The negative-data problem, stated plainly:** *"The classifier has to be trained on successful and failed reactions. Unfortunately, failed reactions are rarely reported and not contained in reaction databases."* Same premise as Jia and Toniato → the reason PU/negative-generation is necessary.
- **Two negative-generation strategies (positives-only → pseudo-negatives):**
  1. Apply reaction rules *forward* to reactants of real reactions to produce plausible-but-wrong products (wrong regio-/chemoselectivity).
  2. Shuffle product↔reaction pairs. Generated **100 M negatives** (pre-2015) vs **10 M** test (2015+).
- **Feasibility classifier performance:** AUC 0.99, AUPRC 0.94, FPR 1.5 %, **FNR 14 %** — even with fabricated negatives, a strong "does this reaction go?" model is learnable; the residual 14 % false-negative rate mirrors Jia's over-conservatism.
- The in-scope score *correlates with electronic descriptors (Hammett σ, LUMO)* though never trained on them — feasibility is physically grounded, supporting descriptor/QM features (our curated file already carries a `QM` column).
- **Time-split validation (Sheridan):** train pre-2015, test 2015+ molecules only — prevents temporal leakage at discipline scale.
- Reaction storage as **canonical (optionally isomeric) SMILES**, RDKit/CDK — the canonicalization backbone for any dedup/hash.

## Relevance to our pipeline
- **Framing precedent:** the in-scope filter *is* a reaction-feasibility classifier trained on literature-scale positives — precisely our target, minus PU. Two paradigms to contrast in our benchmark:
  - *Segler-style synthetic negatives* (fabricate implausible products) vs
  - *our PU* (real combinatorial substrate space as unlabeled) vs
  - *PN with measured HTE negatives* (Toniato/Cernak). Reporting all three is a strong methods contribution.
- **Rule-mined negatives are a cheap augmentation** we could add to the unlabeled/negative pool for BH specifically (e.g. enumerate wrong-site aminations), but note they are *chemo/regio* negatives, not *substrate-infeasibility* negatives — different failure mode than JnJ.
- **Time-split is a concrete leakage control:** prefer literature reactions dated *before* the JnJ 2024 HTE campaign so JnJ chemistry cannot leak backward into training sources.
- Confirms **canonical SMILES + rule/fingerprint hashing** as the standard dedup primitive for literature-scale mining.

## Leakage / licensing caveats
- Trained on **Reaxys (Elsevier)** — a *licensed* source (Münster has access) whose ToS forbid bulk ML redistribution; usable for internal counts/validation, **not** for a publishable training corpus. This is the central licensing caveat for any Reaxys-derived positives (see catalog).
- Their 100 M "negatives" are synthetic and would be *wrong* to treat as ground-truth infeasibility for BH substrate feasibility.
- No public dataset released (Reaxys-bound); method reproducible, data not.

## Open questions for us
- Is rule-based forward negative generation worth adding for BH, or does the combinatorial-unlabeled + measured-HTE-negatives combination already dominate? (Ablation.)
- Do QM/electronic descriptors (already in curated file) recover the Hammett/LUMO correlation Segler observed for BH feasibility? Useful sanity check on features.

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Segler & Waller. "Neural-symbolic ML for retrosynthesis and reaction prediction." *Chem. Eur. J.* **23**, 5966 (2017). DOI: 10.1002/chem.201605499 🔒 — the rule-extraction + negative-generation method in detail.
- Coley, C. W.; Barzilay, R.; Jaakkola, T.; Green, W. H.; Jensen, K. F. "Prediction of organic reaction outcomes using ML." *ACS Cent. Sci.* **3**, 434 (2017). DOI: 10.1021/acscentsci.6b00064 — candidate-ranking with generated negatives.
- Reaxys (Elsevier) — licensed reaction DB, ~tens of millions of reactions; https://www.reaxys.com 🔒 (Münster).
- Sheridan. "Time-split cross-validation." *J. Chem. Inf. Model.* **53**, 783 (2013). DOI: 10.1021/ci400084k — leakage-safe splitting.
- Landrum, G. RDKit — canonicalization/fingerprints (open).
