# Prospective Plate Plan — scan ORD/Pistachio for new reaction classes, then run a best-vs-random 96-well plate

> **Status:** plan / handoff doc, 2026-07-11 (Jan's session, claude). Decision NOT yet made — this
> is the runway for post-hackathon work.
> **Purpose:** convert our *offline* result ("literature-only PU pre-scorer beats random/positives-
> only on held-out HTE sets") into a *prospective* wet-lab result ("we ranked a real substrate matrix,
> ran the top-96 and a random-96, and the model's plate had a materially higher hit rate").
> **Companions:** `CLAUDE.md` (charter), `LAB_JOURNAL.md` (append a dated entry when you start any
> phase), `Future_datasets.md` (test-set shortlist), `docs/decisions/007-…md` (why we scoped to
> substrate-level feasibility under general conditions). The leakage-audit reasoning this plan must
> preserve lives in both plate-report PDFs and the 2026-07-11 journal entry.

---

## 0. Why this is the right next move

We already have two clean offline wins with the *same* workflow (literature positives **P** + a
combinatorial candidate grid **U**, substrate-only raw-ECFP, one L2-regularised HistGB):

- **Chemlex amide C–N** — TEST ROC-AUC **0.768** (inductive), beats the in-domain ceiling (0.733).
- **HiTEA Buchwald–Hartwig C–N** — TEST ROC-AUC **0.709** (n=36), beats positives-only and random.

That is the strongest *retrospective* evidence we can get. The remaining tier of the validation
hierarchy is prospective:

```
validation split (optimize freely)  →  held-out HTE test (touched once, offline)  →  WET-LAB PLATE
```

A **best-96 vs random-96 (vs worst-96)** plate is the single experiment that turns "our AUC is 0.77"
into "we ran it and the model's plate hit N× more often." That is the slide that sells plate
pre-scoring as a product and closes the loop to direct-to-biology.

**Two-part idea:**
1. **Generalise** — scan ORD + Pistachio for reaction classes with many examples under *similar*
   conditions, and train a feasibility model per promising class with our existing workflow.
2. **Prospectively validate** — pick one class (**Suzuki–Miyaura** first), take the lab's real
   boronic-acid × aryl-halide stock as the matrix, score it, and run best/random/worst 96-well plates.

---

## 1. Why Suzuki–Miyaura is the first prospective target

- **Positives are abundant and diverse** in ORD/Pistachio/USPTO — Suzuki is (arguably) the most-
  reported reaction in the literature. Signal will not be the bottleneck; this is the regime where our
  method already won (258k amide positives out-signalled the in-domain model).
- **Cheap, stable, abundant building blocks** — boronic acids + aryl halides from lab stock make the
  physical combinatorial matrix trivial to assemble.
- **Real, learnable failure modes** — protodeboronation of 2-heteroaryl / electron-poor boronic acids,
  ortho,ortho-disubstituted steric failure, base-sensitive partners, unstable heteroaryl boronates.
  These are exactly the substrate-intrinsic failures a literature-trained model can plausibly capture
  from under-representation, and they are what make the prediction non-trivial.

Other classes the scan will likely surface (keep as follow-ups): SNAr, reductive amination,
Sonogashira, Chan–Lam, esterification/amidation (amide already done).

---

## 2. The three make-or-break risks (read before designing anything)

**R1 — Condition homogeneity (deepest scientific risk).** Our method is *substrate-only* under
"general conditions." Suzuki feasibility is more condition-sensitive than amide/BH (protodeboronation
is strongly base/temp dependent). Mitigations:
- Filter literature positives **P** to a **matched, robust general-condition regime** (e.g. a
  Pd–XPhos/SPhos or Pd(dppf)Cl₂ precatalyst; K₃PO₄/K₂CO₃; aqueous dioxane or EtOH), not the full
  heterogeneous Suzuki corpus. Otherwise P encodes "feasible under *some* literature condition" while
  the plate runs *one* condition.
- Choose the plate condition to **maximise coverage/robustness**, and state plainly that the model
  predicts *substrate-intrinsic* feasibility, not condition optimisation.

**R2 — Base rate / headroom (design the library, not just the model).** Suzuki under good general
conditions often runs 75–90 % success in HTE. If the matrix is mostly easy pairs, random-96 already
hits ~80 % and best-96 cannot separate from it — the ceiling effect that squeezed HiTEA (69 % base
rate). **Fix = library composition:** deliberately salt in failure-prone members (heteroaryl boronic
acids, ortho-hindered partners, dihalides, base-sensitive groups) so there is genuine failure to
predict. Target an *expected* base rate ~40–60 %, where a 96-well contrast is statistically clean
(e.g. best 70/96 vs random 48/96 is a decisive Fisher/McNemar result; a 5-well gap is not).

**R3 — De-risk offline before spending lab time.** Do **not** run the plate blind. First score a
**public Suzuki HTE set** (candidates to confirm in Phase 2: Perera et al. *Science* 2018 flow-nanomole
Suzuki screen; AstraZeneca / Merck Suzuki HTE sets) with the trained model and confirm AUC > random +
a best-vs-random uplift *retrospectively*, using the exact `run_hitea_eval.py` pattern. Go/no-go gate:
offline Suzuki AUC ≳ 0.65 → commit the plate; ~0.5 → fix the model/conditions first, at zero wet-lab
cost.

---

## 3. Phased plan

### Phase 0 — Reaction-class scan of ORD/Pistachio  *(this week; reusable artifact)*
- Classify reactions (reaction SMILES + templates / NameRxn / RXNMapper, or ORD reaction-type fields).
- Rank classes by **count × condition-homogeneity × substrate-diversity**. Condition-homogeneity =
  how concentrated the (Pd source, ligand, base, solvent, temp) distribution is; we want a class with
  a dominant general-condition regime so R1 is tractable.
- **Deliverable:** `artifacts/…/reaction_class_scan.json` + a short ranked table. Output feeds the
  reaction choice and the condition filter.

### Phase 1 — Mine Suzuki positives under a matched condition filter
- Extract Suzuki positives from ORD/Pistachio restricted to the chosen general-condition regime.
- Reuse the amide/BH mining + merge path (`scripts_JSP/assemble_amide_positives.py`,
  `mine_amide.py` as templates) → `positives_suzuki_merged.parquet`.
- **Leakage discipline (carry over from the audit):** the model is fit on (P, U) only; target labels
  never enter training, no early stopping / HP tuning / feature selection on any test. Tag, at scoring
  time, which high-scored wells are *novel pairs* vs pairs already present in P — the novel-pair hits
  are the convincing ones.

### Phase 2 — Offline de-risk on a public Suzuki HTE set  *(go/no-go gate — see R3)*
- Clone the `run_hitea_eval.py` + `make_report_hitea.py` pattern for Suzuki. Multi-seed PU +
  positives-only + (if viable) in-domain reference, bootstrap CI, honest caveats.
- **Gate:** proceed to wet-lab only if the offline Suzuki AUC clears the R3 threshold.

### Phase 3 — Score the lab substrate matrix  *(the deployment)*
- **U = the real lab matrix**: our boronic acids × our aryl halides (this is transductive by design and
  correct here — we are scoring exactly the wells we can run).
- Reuse `run_dataset_switch.py` (`build_*` → featurize P/U → `sup_histgb_l2.fit(P, U)` →
  `predict_proba(U)`), producing a ranked score per candidate well.

### Phase 4 — Plate design + pre-registration  *(commit BEFORE running)*
- **Three plates from the same matrix + same condition:** **best-96** (top score), **random-96** (true
  random sample = honest base-rate baseline), **worst-96** (bottom score — the money shot: near-zero
  hits shows the model finds *failures*, the "predict no-reaction"/TACIT claim made physical).
- **Pre-register** (freeze in this doc / an ADR before any well is run):
  - readout metric + success threshold, matched to training labels (we used Conversion ≥ 20 % /
    yield ≥ 10 %) — e.g. LC-MS product call or UV area % ≥ X;
  - the fixed reaction condition;
  - randomisation of physical plate/day across the three sets (avoid confounding model quality with
    batch effects);
  - the analysis: hit-rate uplift + a McNemar/Fisher test, plus a **calibration plot** (do 0.9-scored
    wells work more than 0.6-scored? ~192+ wells supports score-binned deciles).

### Phase 5 — Run, read out, analyse, report
- Run plates; read out; compute best/random/worst hit rates + significance + calibration.
- Extend the report generator (a `make_report_suzuki.py` sibling) to a prospective-results page.
- Append a `LAB_JOURNAL.md` entry (hypothesis · setup · numbers · interpretation · next).

---

## 4. What to reuse (low incremental cost)

| Need | Reuse |
|---|---|
| P/U build + fit + score | `scripts_JSP/run_dataset_switch.py` (`build_*`, `MODEL_SPEC`, `FEATURIZERS`) |
| multi-seed eval + baselines | `scripts_JSP/run_hitea_eval.py`, `run_payn_baselines.py` |
| PDF report (summary/confusion/curves/plate/leakage) | `scripts_JSP/make_report_hitea.py` |
| featurizers / model | `src/pubench/features` (raw ECFP), `src/pubench/models` (`sup_histgb_l2`) |
| mining/merge | `scripts_JSP/assemble_amide_positives.py`, `mine_amide.py` |
| compute | SLURM (`docs/slurm.md`), `--nice=1000000` |

---

## 5. Open decisions (resolve as we go)
- **Which general condition** to fix for the plate (drives the P filter and R1) — chemist's call.
- **Library composition** to hit the ~40–60 % expected base rate (R2) — needs a first-pass score of the
  full lab matrix, then curate.
- **Readout + threshold** — must be pre-registered and matched to the training-label definition.
- **Is a public Suzuki HTE set with real negatives available** at the needed scale (Phase 2)? Confirm
  Perera 2018 / AZ / Merck deposits.
- **Model per class** — reuse `sup_histgb_l2`, but re-confirm it transfers for Suzuki offline before
  trusting it prospectively.

## 6. Success criteria
- **Offline (Phase 2):** Suzuki AUC > random and a best-vs-random uplift, mirroring amide/BH.
- **Prospective (Phase 5):** best-96 hit rate significantly > random-96 (Fisher/McNemar), worst-96
  materially below random, and monotone calibration by predicted score.
- **Story:** "One literature-only pre-scorer, three reaction classes (amide C–N, BH C–N, Suzuki C–C);
  on Suzuki we ran the plate and the model's wells hit N× more often than random."
