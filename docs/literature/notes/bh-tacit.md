# TACIT — Reaction Outcome Prediction from Positive-Only Data (our group)
- **Citation:** N. Wierich*, N. Dobberstein*, F. Glorius. *Implicit Learning: Reaction Outcome Prediction from
  Positive-Only Data* (TACIT = TAckling Chemistry through ImpliciT negative learning). Univ. Münster, OCI.
  (Under review; *Nat. Mach. Intell.* per group notes.) Preprint PDF in library.
- **Type:** method
- **File:** literature/TACIT-1.pdf
- **TL;DR:** A graph-edit reaction-outcome model that predicts **whether a reaction happens at all** — including
  "no reaction" — while trained on **positive-only** data, by framing each step as a ranking between bond pairs vs a
  dedicated **stop action**, so it implicitly learns unreactive substrate combinations.

## Problem & contribution
Standard forward-prediction models trained on positive-only literature data systematically over-predict reactivity
(they can't say "no reaction"). TACIT extracts implicit negative information from purely positive data. Three pieces:
1. **Blockspace algorithm** — decomposes any reaction into a minimal sequence of elementary graph-edit steps using
   Bond-Electron (BE) matrix algebra + BFS; doubles as a data cleaner (chemically impossible / mis-atom-mapped
   reactions have no BE solution and are filtered out).
2. **Implicit negative learning** — each stage is a ranking problem over all bond pairs plus a **stop action**; learning
   which bonds are reactive vs unreactive implicitly teaches whether a reaction occurs at all.
3. **Single elementary step** = break 2 bonds, form 2 bonds (four loose ends recombined). Iterated with step history;
   full atom/mass conservation, hallucination-free by design.

## Dataset(s) — scope, size, how feasibility is defined
- Trained on **USPTO-MIT** and a larger compiled **PUCD** dataset (Lowe/Pistachio/etc.).
- "Feasibility" is emergent: the model outputs "no reaction" when the stop action outranks all bond pairs. Two boosters:
  an **auxiliary stop loss** (rank stop below the true reactive pair but above all unreactive bonds) and **synthetic
  negatives** auto-derived from positives by removing reactants (SN25/50/100 = % of train augmented).
- Negative-eval benchmarks (its real contribution): **LITNEGATIVE / LITPOSITIVE** (failed vs successful rxns from the
  same CAS papers), **SOLVENT** (reactant + a solvent it's used with → should be no-reaction), **REGIO** (RegioSQM20).

## Method & features
- Architecture: **GATv2** message passing → per-atom → atom-pair MLP → per-bond reps → **bond transformer**
  (bond self-attention) → bond-pair attention → scores; **+ separate stop action S**; argmax selects next edit or stop.

## OOD / splitting strategy & metrics
- Positive metric: top-1 accuracy on USPTO-MIT. Base model **89.5%** (competitive with Molecular Transformer 88.8,
  Graph2SMILES 90.3, Chemformer 90.9, ReactionT5 92.8/97.5). Adding stop loss → 87.5; +100% synthetic negatives → 84.4
  (only ~5 pts lost for gaining negative prediction).
- Negative metric: fraction of **failed** reactions correctly called "no reaction." Best model (PUCD+SN100) identifies
  **51.2% of LITNEGATIVE failures** while still predicting the correct product for 63.5% of LITPOSITIVE successes.
  Literature models (Molecular Transformer, LocalTransform, ReactionT5) score ~0% on no-reaction. SOLVENT: up to 87.2%
  correctly "no reaction". → In an experimental campaign, TACIT would roughly **halve the number of experiments**.

## Relevance to our pipeline (esp. how it complements PU feasibility)
- **TACIT is the "implicit-negative" half** of our TACIT+PAYN thesis. It shows positive-only data already encodes what
  does NOT react — exactly the signal a PU feasibility model needs. For BH C–N:
  - Use TACIT-style **stop-action / bond-ranking** to generate **reliable implicit negatives** for our BH substrate grid
    (unreactive aryl/amine sites → likely no C–N bond formed), complementing PAYN's **spy-based** reliable negatives.
  - Combine: TACIT gates "can this pair react at the C–N site at all?" (mechanistic feasibility) → PAYN/PU ranks
    "at ≥10% yield under these general conditions?" (quantitative feasibility).
- Not BH-specific and not condition-aware (no catalyst/base/solvent) — so on its own it can't hit the JnJ condition-resolved
  target; its value is as a **negative-data generator / feasibility prior**, or an architecture template (bond-ranking + stop)
  we could adapt to a condition-conditioned BH head.
- **No JnJ leakage risk** (trained on USPTO/PUCD literature, not J&J HTE), but check USPTO/PUCD for any JnJ substrate pairs
  if we fold TACIT outputs into training features.

## Open questions for us
- Can we condition TACIT's graph-edit step on reagents (catalyst/base/solvent) to make it condition-aware for BH?
- Is TACIT's implicit-negative set correlated with, or complementary to, PAYN's spy negatives on the SAME BH pairs?
  (If complementary, ensemble both as the "reliable negative" set for PU.)
- Use TACIT's blockspace cleaner to filter mis-mapped BH literature reactions before mining positives?

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Boser, Spies, Glorius (PAYN). *JACS* 2026, 148, 15066. DOI 10.1021/jacs.6c00127 🔓 (the PU/yield half — ref [27])
- Schrader, Schäfer, Schäfers, Glorius. Reactivity data bias. *Nat. Chem.* 2024, 16, 491. DOI 10.1038/s41557-024-... 🔒
- Joung, Fong, …, Coley. FlowER (electron-flow matching). *Nature* 2025, 645, 115. DOI 10.1038/s41586-025-09426-9 🔒
- Sacha et al. MEGAN (graph-edit). *JCIM* 2021, 61, 3273. DOI 10.1021/acs.jcim.1c00537 🔒
- Chen, Jung. LocalTransform. *Nat. Mach. Intell.* 2022, 4, 772. DOI 10.1038/s42256-022-00526-z 🔒
- Ugi/Dugundji BE-matrix formalism (blockspace basis). *Angew.* 1979, 18, 111.
