# Literature Review — Combining TACIT + PAYN for Literature-Trained Reaction-Outcome Prediction → Higher-Hit-Rate HTE Plates → Direct-to-Biology Drug Discovery

*Compiled for Florian Boser & Prof. Frank Glorius (Glorius group, Univ. Münster). Assembled from five parallel literature sweeps (outcome prediction · yield/PU-learning · HTE optimization & plate design · chemistry→biology · self-driving-labs/agents/featurization). All DOIs were cross-checked by the searching agents against Crossref/publisher records; items the agents could not verify are marked **⚠︎flag**. Preprints are marked **[preprint]**.*

**The pipeline this review supports:**
> Fix a set of **general reaction conditions** for a common, pharma-relevant class (C–N Buchwald–Hartwig, heteroaryl Suzuki) → score a large **substrate × substrate** matrix with **TACIT** (feasibility / "does it react", implicit negatives) as a gate and **PAYN** (yield ranking, spy-based reliable negatives) as the ranker → **design a high-hit-rate, diverse HTE plate** → send crude products **direct-to-biology** for small-molecule hit discovery.

Legend: 📁 = already a PDF in `Literature/` · ⭐ = priority read · 🔓 = open access · 🔒 = likely paywalled (see pull-list at end).

---

## 0. Core papers already in your `Literature/` folder

| Paper | What it is |
|---|---|
| 📁 **TACIT** — Wierich, Dobberstein, Glorius, *Implicit Learning: Reaction Outcome Prediction from Positive-Only Data* (under revision, *Nat. Mach. Intell.*) | Outcome/product prediction learning implicit negatives via bond-ranking + stop action; the **feasibility gate**. |
| 📁 **PAYN** — Boser, Spies, Glorius, *JACS* 2026, 148, 15066. DOI 10.1021/jacs.6c00127 🔓(CC-BY) | Spy-based positive-unlabeled learning → reliable negatives → yield; the **yield ranker**. |
| 📁 **Angello, …, Aspuru-Guzik, Grzybowski, Burke** — *Closed-loop optimization of **general reaction conditions** for heteroaryl Suzuki–Miyaura*, *Science* 2024. DOI 10.1126/science.adc8743 🔒 | The "general conditions across many substrates" strategy — your condition-fixing premise. |
| 📁 **Neves, …, Schwaller, Strambeanu** — *Robust **Out-of-Distribution** Prediction of Buchwald–Hartwig*, ChemRxiv 2025, DOI 10.26434/chemrxiv-2025-xcr46 **[preprint]** 🔓 | Your target class (C–N) + the OOD-generalization bar; data-standardization + active learning. |
| 📁 **Toniato, Vaucher, Laino, Graziani** — *Negative chemical data boosts language models in reaction outcome prediction*, *Sci. Adv.* 2025, eadt5578. DOI 10.1126/sciadv.adt5578 🔓 | Independent proof negatives improve outcome prediction — validates TACIT's premise; RL recipe to inject negatives. |
| 📁 **Jia, …, Norquist, Schrier** — *Anthropogenic biases in chemical reaction data*, *Nature* 2019, 573, 251. DOI 10.1038/s41586-019-1540-5 🔒 | Foundational evidence positive-only/human-selected data is biased — the problem you solve. |
| 📁 **Segler, Preuss, Waller** — *Planning chemical syntheses with DNNs and symbolic AI*, *Nature* 2018, 555, 604. DOI 10.1038/nature25978 🔒 | Retrosynthesis-planning lineage (context). |
| 📁 **Douthwaite, …, Cernak** — *Nanoscale **Direct-to-Biology** Optimization of Cdk2 Inhibitors*, *J. Med. Chem.* 2026, 69, 9142. DOI 10.1021/acs.jmedchem.? 🔒 | The biology endpoint — plates → crude → bioassay → optimized inhibitors. |

---

## 1. The problem: biased / positive-only data & the value of negatives
*(Why TACIT + PAYN exist at all.)*

- ⭐ **Maloney, Coley, … Wiest** — *Negative Data in Data Sets for ML Training.* *Org. Lett.* 2023, 25, 2945. DOI 10.1021/acs.orglett.3c01282 🔒 — Defines what "negative" means (no-reaction vs wrong-product vs low-yield); grounds treating TACIT's implicit and PAYN's reliable negatives as one object.
- 📁 Jia/Schrier 2019 (above) — power-law reagent popularity; random-sampled beats human-selected training data.
- **Ma et al.** — *Revisiting Yield Prediction from an **Imbalanced-Regression** Perspective.* WWW '24 Companion, 790. DOI 10.1145/3589335.3651470 🔓 — Reported gains inflated by well-populated low-yield range; cost-sensitive re-weighting recovers the high-yield tail. The evaluation critique motivating negative augmentation.
- **Zacate, …, Doyle, Sigman** — *Considerations in Pursuing Reaction Scope Generality.* *Angew.* 2025, e202511091. DOI 10.1002/anie.202511091 🔒 — Unbiased substrate selection, informer libraries, statistical scope evaluation.
- (In library / context: Schrader *Nat. Chem.* 2024; Strieth-Kalthoff *Angew.* 2022; Beker *JACS* 2022; Gao/Coley contrastive *JACS* 2025.)

---

## 2. Reaction outcome / product prediction (TACIT's world)

- ⭐ 📁 **Toniato et al.** *Sci. Adv.* 2025 (above) — negatives + RL → SOTA outcome prediction from few positives.
- **Zhong et al.** — *Global reaction **feasibility** and robustness via HTE + Bayesian deep learning.* *Nat. Commun.* 2025, 16, 4522. DOI 10.1038/s41467-025-59812-0 🔓 — **Likely your own group (Glorius, Münster) — verify.** 11,669 acid–amine couplings; predicts feasibility (~89%) + robustness with calibrated uncertainty; active learning cuts data ~80%. **The closest existing "whether + how well" joint model** and a negative-rich benchmark. *(Surfaced by 3 of 5 agents — central.)*
- ⭐ **Xu, …, Qi** — *RXNGraphormer: unified pre-trained framework for cross-task reaction-performance + synthesis.* *Nat. Mach. Intell.* 2025, 7, 1561. DOI 10.1038/s42256-025-01098-4 🔒 — GNN+Transformer on ~13M reactions; SOTA across reactivity/selectivity/**yield** + forward/retro from one encoder. **Architectural blueprint for one model doing outcome + yield.**
- **Sacha et al.** — *MEGAN: Molecule Edit Graph Attention Network.* *JCIM* 2021, 61, 3273. DOI 10.1021/acs.jcim.1c00537 🔒 — Products as ordered graph edits (arrow-pushing); closest published analogue to TACIT's break-2/form-2 action space.
- **Miller, …, Van Vranken, Baldi** — *Mechanism-Aware DL for Polar Reaction Prediction.* *JACS* 2025, 147, 41168. DOI 10.1021/jacs.5c16838 🔒 — Electron-pushing / elementary-step decomposition; reinforces TACIT's philosophy.
- **Tu, Coley** — *Graph2SMILES.* *JCIM* 2022, 62, 3503. DOI 10.1021/acs.jcim.2c00321 🔒 — Permutation-invariant graph encoder + Transformer decoder (template-free).
- **Irwin et al.** — *Chemformer.* *Mach. Learn.: Sci. Technol.* 2022, 3, 015022. DOI 10.1088/2632-2153/ac3ffb 🔓 — BART pre-trained on ~100M SMILES; strong warm-start backbone.
- **Lu, Zhang** — *T5Chem.* *JCIM* 2022, 62, 1376. DOI 10.1021/acs.jcim.1c01467 🔒 — One T5 does forward + retro + type + **yield** via task prefixes — existing proof a single transformer joins outcome + yield.
- **Hoque, …, Sunoj** — *ReactAIvate.* ECAI 2024 / arXiv:2407.10090 **[preprint/conf]** 🔓 — Interpretable attention-GNN localizes reactive "hotspots"; parallel to TACIT's bond-ranking.
- (In library / context: FlowER — Joung/Coley *Nature* 2025, 645, 115, DOI 10.1038/s41586-025-09426-9 🔒, electron-flow matching over bond-electron matrices, very close to TACIT's conservation-respecting steps; Molecular Transformer *ACS Cent. Sci.* 2019; LocalTransform *Nat. Mach. Intell.* 2022; ReactionT5 2025.)

---

## 3. Yield prediction & positive-unlabeled learning (PAYN's world)

- ⭐ **Schwaller, Vaucher, Laino, Reymond** — *Yield-BERT.* *Mach. Learn.: Sci. Technol.* 2021, 2, 015016. DOI 10.1088/2632-2153/abc81d 🔓 — Canonical yield-regression baseline; its out-of-sample collapse is exactly what reliable-negative augmentation fixes. Code: `rxn4chemistry/rxn_yields`.
- **Ma et al.** WWW '24 (see §1) — imbalanced-regression baselines PAYN competes with.
- **Liu, Moroz, Isayev** — *Balancing sensitivity vs robustness in yield prediction (amide couplings).* *Chem. Sci.* 2023, 14, 10835. DOI 10.1039/d3sc03902a 🔓 — Clean amide-coupling HTE benchmark + fair-evaluation discipline.
- **Kwon et al.** — *Uncertainty-aware yield prediction with GNNs.* *J. Cheminform.* 2022, 14, 2. DOI 10.1186/s13321-021-00579-z 🔓 — Aleatoric+epistemic UQ → plug-in applicability-domain layer for deciding which predicted hits to plate.
- **Chen et al.** — *MetaRF.* *J. Cheminform.* 2023, 15, 43. DOI 10.1186/s13321-023-00715-x 🔓 — Meta-learned attention-RF adapts to a new reagent set from a handful of measurements ("design the next plate from little data").
- **Singh, Hernández-Lobato** — *Bayesian Meta-Learning for Few-Shot Asymmetric Hydrogenation.* *Angew.* 2025, e202503821. DOI 10.1002/anie.202503821 🔒 — SOTA few-shot literature-yield method + **new >12,000-reaction AHO literature dataset**.
- **Singh, Hernández-Lobato** — *Deep kernel learning for outcome prediction + optimization.* *Commun. Chem.* 2024, 7, 136. DOI 10.1038/s42004-024-01219-x 🔓 — GP-UQ features drive Bayesian optimization; bridges yield → plate design.
- **Shi et al.** — *ReaMVP: multi-view pre-training for yield.* *J. Cheminform.* 2024, 16, 22. DOI 10.1186/s13321-024-00815-2 🔓 — Self-supervised on unlabeled reactions before yield fine-tuning; complements using that space as PAYN's PU pool.

**PU-learning theory & cross-domain (menu of alternatives to PAYN's spy approach):**
- ⭐ **Kiryo, Niu, du Plessis, Sugiyama** — *PU Learning with Non-Negative Risk Estimator (nnPU).* NeurIPS 2017, arXiv:1703.00593 🔓 — The standard deep-PU risk backbone; alternative to benchmark PAYN against.
- **Sun et al.** — *Contrastive PU learning (CPUL) for crystal synthesizability.* *Comput. Phys. Commun.* 2025, 308, 109465. DOI 10.1016/j.cpc.2024.109465 🔒 — Contrastive + PU fusion (mirrors Gao/Coley contrastive × PAYN PU).
- **Chung et al.** — *Solid-state synthesizability via PU from literature.* *Digital Discovery* 2025, 4, 2439. DOI 10.1039/d5dd00065c 🔓 — Near one-to-one PAYN analogue in materials (literature positives + unlabeled + PU).
- **Aydın et al.** — *Tree-based models across PU frameworks.* *Adv. Theory Simul.* 2026, e70395. DOI 10.1002/adts.70395 🔒 — Head-to-head of 9 PU strategies × 4 tree ensembles — a practical selection menu.
- **Stolfi et al.** — *NIAPU: network-informed adaptive PU for disease-gene ID.* *Bioinformatics* 2023, 39, btac848. DOI 10.1093/bioinformatics/btac848 🔓 — Adaptive reliable-negative labeling; a bridge to the chemistry→biology hand-off (positive-only bioactivity).
- (In library / context: Elkan-Noto 2008; Bekker-Davis 2020; Liu spy 2002; Nishii ACS Omega 2025; materials PU set — Gu/Antoniuk/Jang/SynCoTrain.)

---

## 4. Reaction representation & featurization (feeds both heads)

- ⭐ **Probst, Schwaller, Reymond** — *DRFP.* *Digital Discovery* 2022, 1, 91. DOI 10.1039/D1DD00006C 🔓 — Data-independent fingerprint matching learned FPs on yield with a GBM, **no GPU/pretraining** — fastest featurizer to get the combined model running. Code: `reymond-group/drfp`.
- **Schwaller, Probst, … Reymond** — *RXNFP (learned reaction fingerprints).* *Nat. Mach. Intell.* 2021, 3, 144. DOI 10.1038/s42256-020-00284-w 🔒 — Shared representation for outcome + yield heads; supports diversity-aware plate selection. Code: `rxn4chemistry/rxnfp`.
- **Żurański, Wang, Shields, Doyle** — *Auto-QChem.* *React. Chem. Eng.* 2022, 7, 1276. DOI 10.1039/D2RE00030J 🔓 — Automated DFT descriptors (~168) + queryable DB; interpretable complement to learned FPs. Code: `doyle-lab-ucla/auto-qchem`.
- ⭐ **Sagawa, Kojima** — *ReactionT5.* *J. Cheminform.* 2025, 17, 126. DOI 10.1186/s13321-025-01075-4 🔓 — ORD-pretrained T5 does **product (97.5%) + retro + yield (R²≈0.95)** from one backbone with few-shot transfer — **the closest existing analogue to the TACIT+PAYN vision.** (Also in your library set.) Code: `sagawatatsuya/ReactionT5v2`.
- **Zhou, Gao et al.** — *Uni-Mol.* ICLR 2023 (OpenReview 6K2RM6wVqKu) 🔓 — SE(3) transformer on ~209M conformers; 3D backbone for sterics/chirality-dependent reactivity. Code: `deepmodeling/Uni-Mol`.
- **[preprint]** Uni-Mol3 — arXiv:2508.00920 (2025) — next-gen 3D reaction backbone; await peer review.
- (Preprocessing: **RXNMapper** — Schwaller *Sci. Adv.* 2021, eabe4166 🔓; **LocalMapper** — Chen/Jung *Nat. Commun.* 2024, DOI 10.1038/s41467-024-46364-y 🔓 — atom-mapping quality is upstream of TACIT's bond-edit supervision.)

---

## 5. HTE + optimization + plate / experimental design
*(How outcome+yield scores become an actual plate.)*

- ⭐ **Sivilotti, Friday, Jackson** — *Active learning high-coverage sets of **complementary** reaction conditions.* *Digital Discovery* 2025, 4, 846. DOI 10.1039/D4DD00365A 🔓 — Small sets of complementary conditions cover more substrate space than any single general condition → **the clearest formalization of your plate-design objective.**
- ⭐ **Souza, …, Doyle, Tan** — *Active learning toward a generalizable model for Ni-photoredox XEC.* *JACS* 2025, 147, 18747. DOI 10.1021/jacs.5c02218 🔒 — Uncertainty AL builds a yield model over a 33,312-compound space with <100 added reactions — template for the "next-wells" loop.
- **Torres, …, Doyle** — *EDBO+ multi-objective active-learning optimizer.* *JACS* 2022, 144, 19999. DOI 10.1021/jacs.2c08592 🔒 (code 🔓 `doyle-lab-ucla/edboplus`) — Off-the-shelf multi-objective plate selection; consumer of TACIT/PAYN predictions.
- **Desimpel, …, Stevens** — *Bayesian optimization for chemical reactions (review).* *Chem. Soc. Rev.* 2026, 55, 2731. DOI 10.1039/D5CS00962F 🔒 — Definitive current BO tutorial (surrogates, acquisition, categorical, batch, multi-objective, transfer).
- **Felton, Rittig, Lapkin** — *Summit: benchmarking optimization methods.* *Chem. Methods* 2021, 1, 116. DOI 10.1002/cmtd.202000051 🔓 (code 🔓 `sustainable-processes/summit`) — In-silico benchmarks to validate plate-selection before spending HTE.
- **King-Smith, Berritt, … (Pfizer/Cambridge)** — *Probing the chemical 'reactome' with HTE data.* *Nat. Chem.* 2024, 16, 633. DOI 10.1038/s41557-023-01393-w 🔒 — **39,000+ HTE reactions released** + dataset-agnostic analyzer; anchor corpus to pretrain/validate + mine hit-rate-raising reagent choices.
- **Ranković et al.** — *BO for additive screening beyond one-hot.* *Digital Discovery* 2024, 3, 654. DOI 10.1039/D3DD00096F 🔓 — Richer molecular reps in the GP surrogate (built on GAUCHE).
- **Shim, …, Cernak** — *Prospective active transfer learning on amine + acid coupling.* *Digital Discovery* 2025, 4, 3693. DOI 10.1039/D5DD00309A 🔓 — Reuses prior data, adaptively picks reagents, lifts yields in 3 batches.
- **Haas, Kalyani, Sigman** — *Statistical modeling for **sparse** datasets.* *Sci. Adv.* 2025, 11, eadt3013. DOI 10.1126/sciadv.adt3013 🔓 — Discipline for trustworthy models from sparse/biased data (directly relevant to literature training).
- **Li et al.** — *Closed-loop BO for metallophotocatalyst formulation.* *Nat. Chem.* 2024, 16, 1286. DOI 10.1038/s41557-024-01546-5 🔒 — Multi-component formulation combinatorics = multi-component well selection.
- **Bradshaw, …, Coley** — *Challenging Reaction Prediction Models to Generalize.* *ACS Cent. Sci.* 2025. DOI 10.1021/acscentsci.5c00055 🔓 (arXiv:2501.06669) — **Random splits overstate accuracy;** proposes time/author/class OOD splits. **Essential to validate your hit-rate claims honestly.**
- (Reviews/DoE: Ali *Chem. Asian J.* 2025, e202500825; Romano *Chem. Methods* 2025; Wall *ACS Catal.* 2025, 15, 8885. HTE datasets in library: Ahneman, Perera, Stevens.)

---

## 6. Chemistry → biology: direct-to-biology, libraries, DELs
*(The endpoint — more successful wells → more testable molecules → better drugs.)*

- ⭐ **Nippa et al.** — *Expediting hit-to-lead through reaction prediction + multi-dimensional optimization.* *Nat. Commun.* 2025. DOI 10.1038/s41467-025-66324-4 🔓 — **Your closest prior art:** 13,490 Minisci HTE reactions → GNN predicts outcomes → ~26,375-product library → ~212 synthesized MAGL inhibitors. Read first; define your delta (literature-only, PU/implicit negatives, general conditions).
- ⭐ **Nippa, …, Grether (Roche)** — *Late-stage drug diversification by HTE + geometric deep learning.* *Nat. Chem.* 2024, 16, 239. DOI 10.1038/s41557-023-01360-5 🔒 — Predict late-stage reaction outcome to reliably grow SAR libraries (direct TACIT-half precedent).
- 📁 Douthwaite/Cernak Cdk2 D2B (above) — nanoscale multistep synthesis → crude → bioassay.
- **Hübner, Barthels** — *Direct-to-Biology: Streamlining Chemistry→Biology (review).* *ChemMedChem* 2026, e202501080. DOI 10.1002/cmdc.202501080 🔒 — Best current map of D2B (crude-mixture screening, assays, PROTAC dominance, ~25 papers/yr).
- **McCorkindale et al.** — *Deconvoluting low yield from weak potency in D2B with ML.* *RSC Med. Chem.* 2024, 15, 1015. DOI 10.1039/d3md00719g 🔓 — ML crude-yield estimate lets you tell "reaction failed" from "product inactive" — **why PAYN's yield signal is essential for D2B hit-calling.**
- **Stevens et al.** — *Integrated D2B platform for nanoscale PROTAC synthesis + evaluation.* *J. Med. Chem.* 2023, 66, 15437. DOI 10.1021/acs.jmedchem.3c01604 🔒 — Concrete "plate → straight to biology" workflow.
- **Plesniak et al.** — *Rapid PROTAC discovery: nanomole array synthesis + direct screening.* *ACS Med. Chem. Lett.* 2023, 14, 1882. DOI 10.1021/acsmedchemlett.3c00314 🔒 — Robust nanomole conditions for D2B.
- **Sadybekov et al.** — *Synthon-based ligand discovery in >11B-compound virtual libraries (V-SYNTHES).* *Nature* 2022, 601, 452. DOI 10.1038/s41586-021-04220-9 🔒 — Synthesizability-constrained ultra-large screening → high experimental hit rates.
- **Swanson et al.** — *SyntheMol: generative design of synthesizable antibiotics.* *Nat. Mach. Intell.* 2024, 6, 338. DOI 10.1038/s42256-024-00809-7 🔒 — Generation constrained to real building blocks + reactions; 83% made, 6 active.
- **Raghavan et al.** — *Synthetic accessibility via Suzuki yield prediction on AbbVie 15-yr parallel-library data.* *JACS* 2024, 146, 15070. DOI 10.1021/jacs.4c00098 🔒 — **Closest published analogue to fusing PAYN-style yield prediction with library selection.**
- **DELs + ML:** McCloskey et al. *J. Med. Chem.* 2020, 63, 8857, DOI 10.1021/acs.jmedchem.0c00452 🔒; Iqbal et al. *npj Drug Discovery* 2025, DOI 10.1038/s44386-025-00007-4 🔓; Puglioli et al. *Chem. Sci.* 2023, 14, 12026, DOI 10.1039/d3sc03688j 🔓 (library input → hit rate); Fitzgerald et al. *JCIM* 2024, 64, 4661, DOI 10.1021/acs.jcim.4c00232 🔒.

---

## 7. Self-driving labs & agentic chemistry
*(Where a deployed TACIT+PAYN "brain" lives — relevant to Florian's agentic-workflow focus.)*

- ⭐ **Tom, …, Aspuru-Guzik** — *Self-Driving Laboratories for Chemistry and Materials (review).* *Chem. Rev.* 2024, 124, 9633. DOI 10.1021/acs.chemrev.4c00055 🔒 — Reference map for embedding a predictor in a closed loop.
- **Boiko, …, Gomes** — *Coscientist.* *Nature* 2023, 624, 570. DOI 10.1038/s41586-023-06792-0 🔒 — GPT-4 agent (search/code/robot) autonomously optimized Pd couplings; blueprint agent that calls TACIT/PAYN as tools.
- **M. Bran, …, Schwaller** — *ChemCrow.* *Nat. Mach. Intell.* 2024, 6, 525. DOI 10.1038/s42256-024-00832-8 🔒 — LLM + 18 cheminformatics tools; template for exposing TACIT/PAYN as callable tools.
- **Burger, …, Cooper** — *A Mobile Robotic Chemist.* *Nature* 2020, 583, 237. DOI 10.1038/s41586-020-2442-2 🔒 — 688 experiments / 10-variable space via batched BO.
- **Volk, Abolhasani** — *AlphaFlow.* *Nat. Commun.* 2023, 14, 1403. DOI 10.1038/s41467-023-37139-y 🔓 — RL over the *sequence* of operations (discrete+continuous action design).
- **Volk, Abolhasani** — *Performance Metrics to Unleash SDLs.* *Nat. Commun.* 2024, 15, 1378. DOI 10.1038/s41467-024-45569-5 🔓 — Acceleration factor / cost / autonomy — yardsticks to prove hit-rate uplift.
- **Ramos, …, White** — *BO of Catalysis with In-Context Learning.* *ACS Cent. Sci.* 2026, 12, 599. DOI 10.1021/acscentsci.5c02418 🔒 — Frozen-LLM in-context regression → BO with no surrogate training; low-friction cold-start.
- **⚠︎flag ChemAgents** — multi-agent robotic AI chemist, *JACS* 2025, DOI 10.1021/jacs.4c17738 (author list unverified).
- (In library / context: Coley robotic flow *Science* 2019; Granda/Cronin *Nature* 2018; Sandfort/Glorius *Chem* 2020.)

---

## 8. Data infrastructure, benchmarks & evaluation

- **Kearnes, …, Coley, Dreher** — *Open Reaction Database (ORD).* *JACS* 2021, 143, 18820. DOI 10.1021/jacs.1c09820 🔒 — Open schema/repo (2M+ reactions, HTE/flow); target format for your outputs + source of mixed +/− data.
- **Wigh, …, Lapkin** — *ORDerly benchmarks.* *JCIM* 2024, 64, 3790. DOI 10.1021/acs.jcim.4c00292 🔓 — Reproducible cleaning over ORD → forward/retro/condition splits. Code: `sustainable-processes/ORDerly`.
- **Joung, …, Coley** — *Reproducing mechanisms from a large mechanistic dataset.* *Angew.* 2024, e202411296. DOI 10.1002/anie.202411296 🔒 — ~5.18M elementary steps — data-side analogue of TACIT's steps; candidate pre-training corpus.
- **⚠︎flag SynRXN** — five-task CASP benchmark, *Sci. Data* 2026, DOI 10.1038/s41597-026-07260-w (arXiv:2601.01943) — verify author list before citing.
- **Datasets to know:** USPTO-MIT/480k/50k (positive-biased); ORD; **King-Smith/Pfizer reactome 39k**; Ahneman B–H (📁-adjacent), Perera Suzuki, Stevens borylation, Neves B–H (📁); Zhong 11,669 acid–amine; **AHO >12,000** (Singh 2025, new); AbbVie Suzuki (proprietary); **BELKA** DEL set (Leash, NeurIPS 2024 / Kaggle — ⚠︎flag details).

---

## 9. "Read these first" (10-paper path)

1. **Nippa, *Nat. Commun.* 2025** (§6) — the closest full realization of your pipeline; defines your delta.
2. **Zhong, *Nat. Commun.* 2025** (§2) — joint feasibility+robustness (likely your group); benchmark to beat.
3. **ReactionT5** (§4) & **RXNGraphormer** (§2) — two "one backbone → outcome + yield" precedents.
4. **Toniato, *Sci. Adv.* 2025** 📁 (§2) — negatives improve outcome prediction.
5. **Sivilotti/Jackson, *Digital Discovery* 2025** (§5) — plate design = complementary-condition coverage.
6. **Souza/Doyle, *JACS* 2025** (§5) — the active-learning next-wells loop.
7. **Bradshaw/Coley, *ACS Cent. Sci.* 2025** (§5) — OOD evaluation so hit-rate claims survive.
8. **Raghavan, *JACS* 2024** (§6) — yield prediction as a library-selection filter.
9. **McCorkindale, *RSC Med. Chem.* 2024** (§6) — why yield is essential to D2B hit-calling.
10. **Angello/Burke, *Science* 2024** 📁 (§0) — the general-conditions strategy underpinning "many substrates, few conditions."

---

## 10. 🔒 "Needs Münster access" — priority paywall pull-list

Open-access items above (🔓) don't need this. These high-value 🔒 papers are worth pulling the full PDF via university access (methods/datasets/equations matter):

1. **Nippa — late-stage HTE + GDL**, *Nat. Chem.* 2024, DOI 10.1038/s41557-023-01360-5
2. **King-Smith — reactome HTE (39k) + analyzer**, *Nat. Chem.* 2024, DOI 10.1038/s41557-023-01393-w
3. **Xu — RXNGraphormer (outcome+yield backbone)**, *Nat. Mach. Intell.* 2025, DOI 10.1038/s42256-025-01098-4
4. **Raghavan — Suzuki yield for synthetic accessibility (AbbVie)**, *JACS* 2024, DOI 10.1021/jacs.4c00098
5. **Souza/Doyle — active-learning generalizable yield model**, *JACS* 2025, DOI 10.1021/jacs.5c02218
6. **Singh/Hernández-Lobato — few-shot AHO + new dataset**, *Angew.* 2025, DOI 10.1002/anie.202503821
7. **Torres/Doyle — EDBO+**, *JACS* 2022, DOI 10.1021/jacs.2c08592 (code is open; PDF for method)
8. **Desimpel — BO review**, *Chem. Soc. Rev.* 2026, DOI 10.1039/D5CS00962F
9. **Sadybekov — V-SYNTHES**, *Nature* 2022, DOI 10.1038/s41586-021-04220-9
10. **Sacha — MEGAN (graph-edit analogue to TACIT)**, *JCIM* 2021, DOI 10.1021/acs.jcim.1c00537
11. **Joung/Coley — mechanistic dataset (5.18M steps)**, *Angew.* 2024, DOI 10.1002/anie.202411296
12. **FlowER**, *Nature* 2025, DOI 10.1038/s41586-025-09426-9 (if not already accessible)

*Drop any of these into `Literature/` and I'll analyze them the same way I did TACIT and PAYN.*

---

## 11. Key groups / people to track
Coley (MIT) · **Glorius (Münster — home)** · Schwaller (EPFL) + IBM RXN (Laino/Toniato/Vaucher/Graziani) · Reymond/Probst (Bern) · Doyle (UCLA) + Dreher (Merck) · Sigman (Utah) · Cernak (Michigan) · Nippa/Grether (Roche) · Aspuru-Guzik (Toronto) · Abolhasani (NC State) · White (Rochester) · Lapkin/Felton (Cambridge) · King-Smith / A. Lee (Cambridge) · Chen & Jung (SNU/KAIST) · Baldi & Van Vranken (UC Irvine) · Katritch/Roth (V-SYNTHES) · Shoichet (UCSF) · Hernández-Lobato & S. Singh (Cambridge) · Schrier/Norquist (bias/dataset design) · Sugiyama/Niu (PU theory) · Neri/Scheuermann (ETH, DELs).

*Compiled 2026-07-05. Sources: 5 territory sweeps (74 verified papers, de-duplicated to ~60 here). Preprints and unverified-author items flagged; verify ⚠︎-marked entries before citing in a manuscript.*
