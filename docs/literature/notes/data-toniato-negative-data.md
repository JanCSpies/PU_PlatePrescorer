# Negative chemical data boosts reaction-outcome prediction
- **Citation:** Toniato, A.; Vaucher, A. C.; Laino, T.; Graziani, M. "Negative chemical data boosts language models in reaction outcome prediction." *Sci. Adv.* **11**, eadt5578 (2025). DOI: 10.1126/sciadv.adt5578
- **Type:** method + data-framing (RL on transformer forward-prediction with explicit negatives)
- **File:** literature/sciadv.adt5578 (1).pdf
- **TL;DR:** Explicit failed/low-yield reactions materially improve reactivity models especially when positives are scarce — and the paper catalogs *where* machine-readable negative data actually exist, which is exactly what we need to validate PU-recovered "reliable negatives."

## Problem & contribution
Most reaction models train on **successful patented reactions, "biased toward the positive samples"** (their words, citing Lowe/USPTO and Thakkar). Negative data are scarce, often "inaccessible or not in machine-readable format." They extend RLHF to chemistry: a transformer forward-predictor is tuned with a reward model (ALBERT + SVM on reaction fingerprints) that rewards observed-positive and penalizes observed-negative reactions. With **as few as 20–22 positives** supported by a ≥40× larger negative set, RL matches/beats fine-tuning that fails entirely in the low-positive regime.

## Key findings (bias mechanisms / data availability)
- **Two kinds of negatives:** *type 1* = unexpected-but-real product (informative, e.g. wrong regiochemistry); *type 2* = intended product not formed / starting materials unreacted. **HTE screens (incl. BH) are type-2 negatives** — "an array of catalysts and solvents … screened with the same reactants," low-yield wells = negatives. This is precisely our JnJ/Cernak/Merck data.
- **Negatives help most where positives are sparse** — the low-data regime is exactly our situation for OOD/unpopular JnJ substrates.
- **Concrete negative-data sources they use:**
  - **USPTO/Lowe** patents = positive-biased baseline (ref 6, figshare 10.6084/m9.figshare.5104873).
  - **RegioSQM20** (552 EAS reactions) — negatives *synthetically generated* by repositioning the halogen (analogous to Segler's rule-based negatives).
  - **HiTEA** = King-Smith et al. Nat. Chem. 2024 "chemical reactome" HTE — real experimental negatives; **this is our bundled "HiTEA 2024" source (2,632 rows)**. Note their caveat: coverage is non-uniform, some reactants over/under-represented, and only 9/130 high-yield products had a matched negative from identical reactants → HTE negatives are unbalanced and condition-specific.
  - **Nanomole HTE** (Buitrago Santanilla/Dreher/Cernak, Science 2015) — cited as HTE negatives.
- **Threshold choice:** they binarize HTE at >1 % yield = positive. A **direct precedent for our feasibility label threshold** (our JnJ labels already split ~39 % feasible).
- Reward-model representation: reaction fingerprints (rxnfp, Schwaller 2021); classification-tuning separates positives/negatives better than raw embeddings.

## Relevance to our pipeline
- **Validation of PU "reliable negatives":** PU learning will *infer* likely-negatives from the unlabeled pool. Toniato's message is that *explicit, measured* negatives (Cernak's 71 % zero-yield wells; the label=0 rows of Merck/Roche/HiTEA/Dreher/AZ in our curated file) are the gold standard to **benchmark whether our inferred negatives are real** — a leakage-free external check on PU quality.
- Confirms our **type-2 framing**: an HTE well at ~0 % yield under standard BH conditions is genuine evidence of infeasibility, unlike literature silence.
- Suggests an alternative/ablation to pure PU: **treat HTE negatives as true negatives** and compare a PU model (positives + unlabeled) vs a PN model (positives + HTE negatives) vs a hybrid — a clean addition to our source/strategy benchmark.
- The >1 % yield threshold gives us a defensible, literature-backed definition of "feasible."

## Leakage / licensing caveats
- **HiTEA (King-Smith 2024) is already inside our curated BH file** — if we also mine it from its original release we would double-count and risk train/test contamination. Deduplicate by reaction hash across sources.
- RegioSQM negatives are *synthetic* (repositioned halogen) — fine for their EAS task, but **not** a source of real BH negatives; do not import as ground truth.
- Software on Zenodo (10.5281/zenodo.15132404, CC-BY); datasets are the public USPTO + RegioSQM. No barrier for us.

## Open questions for us
- Do our PU-recovered reliable negatives overlap the *measured* zero-yield wells in Cernak/Merck/Roche/HiTEA? Precision/recall of PU-negatives vs measured negatives = a headline validation metric.
- Should the JnJ label=0 rows (6,896 of them) ever enter training, or stay strictly test-only? (Strictly test-only for no leakage — but they are ideal for reporting PU-negative recall.)

## Cited works worth chasing (DOIs; 🔒 if paywalled)
- Maloney, Coley, Genheden, Carson, Helquist, Norrby, Wiest. "Negative data in data sets for machine learning training." *Org. Lett.* **25**, 2945–2947 (2023). DOI: 10.1021/acs.orglett.3c00792 🔒 — the reference on negative-data definitions.
- King-Smith, E. et al. "Probing the chemical 'reactome' with high-throughput experimentation data." *Nat. Chem.* **16**, 633–643 (2024). DOI: 10.1038/s41557-024-01470-8 🔒 — **= our HiTEA source**.
- Buitrago Santanilla, A. … Cernak, T.; Dreher, S. D. "Nanomole-scale high-throughput chemistry." *Science* **347**, 49–53 (2015). DOI: 10.1126/science.1259203 🔒
- Strieth-Kalthoff, F.; Sandfort, F.; … Glorius, F. "Machine learning for chemical reactivity: the importance of failed experiments." *Angew. Chem. Int. Ed.* **61**, e202204647 (2022). DOI: 10.1002/anie.202204647 🔒 — **Glorius-group; directly on failed-reaction value**.
- Thakkar, A.; Vaucher, A. C.; … Laino, T. "Unbiasing retrosynthesis language models with disconnection prompts." *ACS Cent. Sci.* **9**, 1488–1498 (2023). DOI: 10.1021/acscentsci.3c00372 — quantifies USPTO positive bias.
- Raghavan, P. et al. "Incorporating synthetic accessibility … AbbVie's 15-year Suzuki dataset." *J. Am. Chem. Soc.* **146**, 15070 (2024). DOI: 10.1021/jacs.4c03352 🔒 — industrial ELN yield data precedent.
- Schwaller, P. et al. "Mapping the space of chemical reactions (rxnfp)." *Nat. Mach. Intell.* **3**, 144 (2021). DOI: 10.1038/s42256-020-00284-w — reaction fingerprints for the reward model / our hashing.
