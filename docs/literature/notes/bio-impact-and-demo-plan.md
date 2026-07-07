# Impact & Demo Plan — Feasibility-Designed HTE Plates for Direct-to-Biology
*Owner: bio/"so-what" agent. Purpose: turn our literature-trained PU feasibility+yield model into a plate-design objective, a demoable zero-experiment impact metric, and a 3-minute demo aligned to the judging rubric (Impact 25% · Claude 25% · Depth 20% · Demo 30%).*

---

## The one-sentence impact narrative
> **Exploratory HTE and direct-to-biology plates waste most of their wells on reactions that fail — and a failed well produces no molecule to assay. We train a model on *biased literature data only* (no HTE needed) to pre-score a substrate matrix, then design a plate that fits ~2× more real, testable, diverse molecules into the same footprint — and we prove the uplift on an unseen industrial J&J HTE dataset with zero new experiments.**

Why this beats the obvious framings:
- **vs Nippa (Nat. Commun. 2025, DOI 10.1038/s41467-025-66324-4):** Nippa trained a GNN on **13,490 in-house Minisci HTE reactions** to design a library. We do the analogous thing but **from literature (positive-only/biased) data via PU learning** — no proprietary HTE core required. That is the democratizing delta: any group with Reaxys/USPTO/ORD access can design a better first plate.
- **vs Cernak D2B (our assigned paper):** they spent a **4× well tax** (3,072 wells for 768 compounds) *because they couldn't predict which condition works*. We remove that tax with prediction.

---

## (a) Plate-design objective — formalization

**Setup.** Fix a small set of *general* Buchwald–Hartwig conditions `C` (1–4 catalyst/base/solvent combos). Design space = the substrate matrix of aryl halides `A` × amines `M`. A **plate** `P` is a budget-`B` subset of candidate wells `(a, m, c)` (or, in the fixed-single-condition case, of pairs `(a, m)`).

**Per-well score from our two models.** For well `w = (a,m,c)`:
- `f(w) ∈ [0,1]` = TACIT-style **feasibility** (does the intended C–N product form) — the PU/implicit-negative gate.
- `ŷ(w)` = PAYN **expected yield** (calibrated), and `σ(w)` = predictive uncertainty.
- Define **expected usable product** `u(w) = f(w) · Pr[ŷ(w) ≥ τ]`, where `τ` is the D2B-assayability threshold. **Two thresholds happily converge:** the JnJ dataset's own feasibility label is `labels = 1 iff Yield ≥ 10%` (verified by the data agent, see `bh-JnJ-testset-spec.md`), and the Cernak paper's empirical D2B floor is **~12% conversion — below which crude products were generally invisible in the pooled ASMS bioassay.** So **predicting the JnJ `labels` column IS predicting D2B-assayability** — a genuinely lucky alignment we should call out. Use `τ = 10%` to match the labels exactly; optionally report a stricter `τ = 20%` panel for "confidently assayable."

**Objective (diversity-aware expected hits).** Choose `P` to
> **maximize** Σ_{w∈P} u(w) − λ · (redundancy of `P`) **subject to** |P| ≤ B, ≤1 well per product, condition-set ⊆ `C`.

Concretely, three tractable variants, in increasing sophistication:
1. **Greedy top-`u` (baseline model plate):** rank wells by `u(w)`, take top `B`. Maximizes raw expected hits.
2. **Diversity-penalized (facility-location / DPP):** greedily add the well maximizing `u(w) − λ·max_{w'∈P} sim(w,w')` on DRFP/rxnfp features → high hit-rate **and** scaffold coverage (the real library goal; Cernak explicitly diversity-selected via Morgan-FP t-SNE).
3. **Uncertainty-aware / active-learning (round 2+):** acquisition `u(w) + β·σ(w)` (EDBO+-style, DOI 10.1021/jacs.2c08592) — for the *next* plate, spend some wells reducing model uncertainty where it matters. Positions us as a **closed-loop plate designer**, not a one-shot predictor.

**Condition-selection corollary (the 4× tax lever).** If `|C|>1`, for each pair pick `argmax_c u(a,m,c)` instead of running all of `C`. This is exactly what would have saved Cernak 3× of their 3,072 wells.

---

## (b) Demoable hit-rate-uplift metric — exact recipe on JnJ data

**Headline claim we will show:** *"A 96-well plate designed by our literature-trained model captures ~N× more successful, diverse C–N products than a randomly-chosen plate — measured against real J&J HTE outcomes, with zero new experiments run."*

**Dataset (already in repo):** `datasets/raw/Neves/BH_HTE_Curated_Dataset_v20260219.csv`, `Source == "JNJ HTE 2024"`. Verified structure:
- **11,328 wells**; **110 aryl × 68 amine** (7,480 possible pairs); only **888 pairs measured** (11.9% matrix density); ~12.8 conditions/pair.
- Ground-truth columns: `Yield` (0–114%), `labels` (binary feasibility; **1 = hit**).
- **Base rates (the numbers to beat):** well-level `labels=1` = **39.1%**; well-level `Yield>20%` = **28.2%**; pair-level "works under ≥1 condition" = **85.2%** (so **14.8% of pairs are truly dead under every tested condition** — pure waste a model should avoid).
- The 4 modern general conditions (RuPhos/BrettPhos/XPhos/P(tBu)3 – Pd-G4 with P2Et base) each cover ~600 pairs at **78–81%** hit rate; older P(o-tol)3-Pd-G2 conditions collapse to **6–26%** on the *same substrates* — visceral proof that condition choice, per substrate, swings hit rate 10×.

**Train/test discipline (must be honest — judges will probe this):**
- **Train the PU feasibility + yield models on LITERATURE C–N data ONLY** (USPTO/ORD/Reaxys-derived), never on JnJ. JnJ is the untouched OOD test set. This is the whole point (cf. Neves OOD split, chemrxiv-2025-xcr46; Bradshaw "random splits overstate accuracy," DOI 10.1021/acscentsci.5c00055).
- If literature training isn't ready by demo time, **fall back to a leakage-free cross-source split**: train on the non-JnJ HTE sources in the same CSV (Dreher, Merck, Roche, AZ = ~16k reactions), test on JnJ. State clearly which was used.

**Computation (per plate budget `B`, e.g. 96 and 384):**
1. Score every JnJ candidate (well-level for the headline; pair-level for the honest robustness view) with `u(w)` from the model.
2. **Model plate** `P_model` = top-`B` by `u` (variant 1) and by diversity-penalized `u` (variant 2).
3. **Random plate** `P_rand` = `B` wells sampled uniformly; repeat 1,000× → mean ± 95% CI (this is the "naive exploratory plate" baseline).
4. **Oracle plate** `P_oracle` = top-`B` by *true* `labels`/`Yield` (upper bound; on JnJ, precision@96 = **100%**).
5. **Metrics:**
   - **Hit rate = fraction of plate wells with `labels==1`** (and a second panel with `Yield>τ`).
   - **Uplift = hitrate(P_model) / hitrate(P_rand)** and **Δ = extra hits per plate = B·(rate_model − rate_rand)**.
   - **Enrichment factor** EF = precision@B / base rate.
   - **Sweep `B`** → an enrichment curve (model vs random±CI vs oracle).
6. **Sanity ceiling already computed:** oracle EF = 2.56× at the 39% base rate; a competent literature-PU model landing at ~65–80% precision → **~1.7–2.0× uplift, ~25–40 extra usable molecules per 96-well plate.** Report the model's *actual* number, whatever it is — even 1.5× on true OOD data is a strong, honest result.

**Two-panel result to report (both matter):**
| View | Design space | Random baseline | What "uplift" proves |
|---|---|---|---|
| **Well-level (headline)** | 11,328 wells | 39.1% hits | model ranks pair×condition wells → biggest, most visual uplift |
| **Pair-level (honest/generality)** | 888 pairs, "ever-works" label | 85.2% ceiling / avoid 14.8% dead | model avoids truly-dead substrate pairs under fixed general conditions |

**The Cernak-grounded impact translation (the killer slide):** *"Cernak et al. spent 4 conditions × 768 compounds = 3,072 wells to make 768 molecules, because they couldn't predict the right condition. Our model picks the condition per pair — recovering that 4× as either 4× more diverse molecules on the same plate, or 4× freed ASMS/assay capacity."* Back it with our condition-selection number: model `argmax_c u` hit rate vs single-fixed-condition hit rate on the 888 pairs.

---

## (c) 3-minute demo storyboard (mapped to rubric)

**Format:** pre-recorded screen capture + voiceover; Claude Code visibly in the loop. Target the 30% Demo weight with a *live, watchable* plate-design run and one unforgettable visual.

| Time | Beat | On screen | Rubric hit |
|---|---|---|---|
| 0:00–0:25 | **The waste problem** | Cernak circos figure / our matrix heatmap of failures; "most exploratory wells fail; a failed well = no molecule to assay; the assay slot, not the material, is the scarce resource." | Impact |
| 0:25–0:50 | **The idea** | One diagram: *biased literature C–N data → PU learning (PAYN reliable negatives) + implicit-negative feasibility (TACIT) → per-well `u(w)` → plate objective*. State the delta vs Nippa (literature-only, no HTE core). | Impact + Depth |
| 0:50–1:35 | **Live plate design** | Run the designer live: 7,480-cell matrix → model heatmap of `u` → **watch a 96-well plate get selected** (dots light up), diversity-penalized so it spreads across scaffolds. Show Claude Code orchestrating the pipeline (data prep, scoring, selection). | Claude Use + Demo |
| 1:35–2:20 | **The proof (money shot)** | Side-by-side: **random plate vs model plate**, colored by *true* JnJ outcome. Random ≈ 39% green; model ≈ 70–80% green. Then the **enrichment curve** (model vs random±CI vs oracle) with the headline number burned in: "**~2× more usable molecules, zero new experiments.**" | Demo + Impact |
| 2:20–2:45 | **So what → drugs** | Translate to D2B: more green wells → more crude mixtures above the ~15% assayability line → more real testable molecules per campaign → faster hit-to-lead (Cernak's 1.2 nM leads in one experiment). Show the "4× well-tax recovered" line. | Impact |
| 2:45–3:00 | **Honesty + reproducibility** | "Trained on literature, tested on unseen J&J HTE — a true OOD test. Open-source extension of PAYN." One line on active-learning next plate. | Depth + Impact |

**Visual guidance (see dataviz skill before building):** the enrichment curve and the random-vs-model green-dot plate are the two load-bearing visuals — invest there. Colorblind-safe, works in light/dark, one headline number per frame. Consider an interactive Artifact of the plate designer (slider = budget `B`, live hit-rate readout) as the linkable demo asset.

**Claude-Use angle to make explicit (25%):** Claude Code as the *research-and-build* agent — it read the literature corpus (this notes folder), wrote the PU/plate-design code, ran the JnJ evaluation, and generated the figures. Show the agentic loop, not just a script.

---

## Risks / honesty checklist (pre-empt the judges)
- **JnJ is an optimization set, not a naive exploratory plate** → its 39% base rate is already enriched vs a true cold-start plate. State this; it makes our uplift a *conservative* lower bound (a real exploratory plate has more room). Optionally report a second baseline over the full 888-pair "ever-works" view.
- **Leakage:** literature-train / JnJ-test only; no JnJ rows in training; report which split.
- **Hit definition:** primary = the dataset's own `labels` (Yield≥10%, which doubles as the D2B-assayability line); also show a stricter `Yield≥20%` panel so we aren't accused of threshold-shopping.
- **Diversity:** report scaffold/DRFP coverage of the model plate, not just hit rate, so "maximize hits" doesn't degenerate to one easy corner of the matrix.

## Concrete "run this week" checklist
1. Build literature C–N training set (or use non-JnJ CSV sources as the leakage-free proxy).
2. Train PU feasibility (+ PAYN yield) → produce `f`, `ŷ` for all JnJ wells.
3. Implement `u(w)`, greedy + diversity-penalized selection, random/oracle baselines.
4. Compute enrichment curve + Δhits + EF for B∈{96,384}; the condition-selection number.
5. Build the two money-shot visuals (+ optional interactive Artifact).
6. Record the 3-min demo to the storyboard above.
