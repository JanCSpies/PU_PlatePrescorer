# CLAUDE.md — Project charter & working agreement

> **Auto-loaded into every Claude Code session in this directory.** Keep it accurate and concise.
> Fast-moving detail lives in `LAB_JOURNAL.md` and `docs/` — link, don't inline.

---

## 1. What this project is

**PU_PlatePrescorer** — a publication-track research project by **Jan Christopher Spies**
(PhD student, Glorius group, Organisch-Chemisches Institut, University of Münster).

**Target:** a full paper in **JACS or higher**.

**Thesis (one sentence):** Positive-Unlabeled learning on **real literature reaction data**,
combined with a **virtual (in-silico enumerated) unlabeled substrate space**, pre-scores an HTE
substrate matrix well enough to **raise the plate hit rate** — including for reaction classes where
**no HTE dataset exists**, which is where plate pre-scoring is actually needed.

**Why it matters.** Exploratory HTE and especially **direct-to-biology (D2B)** campaigns spend plate
capacity on wells that simply fail. A failed well costs the same as a successful one but yields no
testable compound. If feasibility can be predicted *before* the plate is laid out — from literature
alone, with no prior HTE data for that reaction class — then every campaign in that class starts
from a better plate. In D2B, where crude reaction mixtures go straight into a biological assay,
a higher synthetic hit rate converts directly into more real, assayable compounds per plate.

### Relation to prior group work (this is the novelty argument)

- **PAYN** — *"Positivity is All You Need": Yield Prediction of Organic Reactions in Biased Data Sets
  via Positive-Unlabeled Learning.* **J. Am. Chem. Soc. 2026, 148, 15066–15075.**
  DOI [10.1021/jacs.6c00127](https://doi.org/10.1021/jacs.6c00127) ·
  code <https://github.com/GloriusGroup/PAYN>
- **TACIT** — reaction outcome/feasibility from positive-only data via implicit negatives
  (under revision, *Nat. Mach. Intell.*).

PAYN established that PU learning recovers usable negatives from biased positive-heavy data — but
its **P and U were synthetic**, carved out of existing HTE datasets by biased subsampling. That is a
proof of principle under a controlled bias, not a demonstration that the method works on the bias
chemistry actually has.

**This project's contribution is the step PAYN did not take:**

| | PAYN (JACS 2026) | **PU_PlatePrescorer (this work)** |
|---|---|---|
| Positives **P** | synthetic — biased subsample of an HTE set | **real literature reactions** (ORD / Pistachio / USPTO) |
| Unlabeled **U** | the held-out remainder of the same HTE set | **virtual** — enumerated combinatorial substrate grid |
| Bias model | imposed, known | **real reporting bias**, unknown and unmodelled |
| Task | yield regression | **feasibility → plate design** |
| Validation | retrospective | retrospective **+ prospective wet-lab HTE** |
| Applies to | classes with an HTE dataset | **classes with no HTE dataset at all** |

The last row is the point. A method that needs an existing HTE dataset cannot help you on a
reaction where you have none — which is exactly when you would want it.

---

## 2. Preliminary result that motivates the paper

**Chemlex amide coupling.** P = 258,645 literature amide pairs mined from ORD + Pistachio;
U = virtual 272 × 231 substrate grid (62,832 pairs); TEST = 2,234 held-out reactions, 35.5 % feasible.
**Both test substrates unseen** (inductive — test substrates removed from U). Zero target labels ever
used in training.

| ranking model | TEST ROC-AUC | AP |
|---|---|---|
| random | 0.500 | 0.355 |
| positives-only (similarity to 258k known positives) | 0.671 | 0.507 |
| **literature-PU — ours (zero target labels)** | **0.768** | **0.621** |
| in-domain supervised ceiling (trained on *real* Chemlex labels) | 0.733 | — |

Plate hit rate, top 25 % of the ranked matrix: **61.8 %** vs **35.5 %** random — a **1.74×** uplift.

Two things make this publishable rather than incremental:

1. **The literature-PU model beats the in-domain supervised ceiling** trained on real labels for that
   very dataset. 258k diverse literature positives out-signal 4,960 in-domain rows once the test
   substrates are novel. In PAYN the PU model sits *between* the positives-only and true-label
   references; here it tops both.
2. **Beating positives-only by +0.097 AUC shows the virtual U carries real signal** — the model is
   not merely saying "this looks like a reaction someone already ran".

**Honest limits (these define the remaining work).** This is one reaction class. The same pipeline on
HiTEA Buchwald–Hartwig reached only 0.709 (n = 36, wide CIs), and the beyond-SCAR `prop_ips` variant
that won on amide (+0.024) **lost** on BH (−0.033) — so the method's advantage is not yet shown to be
general, and no claim of generality should be made from Chemlex alone. See `LAB_JOURNAL.md`
(2026-07-10 through 2026-07-13) and `RESULTS.md`.

---

## 3. The plan to a JACS-level paper

**The gap:** retrospective evidence on public datasets, however clean, is a benchmark result. What
elevates it is showing the model works **prospectively, on a reaction class with no HTE data**, in a
real plate.

**Validation hierarchy** — each tier is a stronger claim than the last:

```
validation split (optimise freely)
  → held-out public HTE test, touched once (Chemlex ✅, HiTEA ✅)
    → PROSPECTIVE WET-LAB HTE PLATE          ← the paper's core experiment
```

**Core experiment (wet lab, Jan).** For a reaction class where **no literature HTE dataset exists**:
1. Enumerate the virtual substrate matrix from real in-house building-block stock.
2. Mine literature positives **P** for that class; score the matrix with the literature-PU model.
3. Run **model-picked vs random** substrate selections on plate, **under literature conditions**
   (conditions taken from the mined literature consensus, not re-optimised per substrate — the
   model predicts substrate-intrinsic feasibility, so the conditions must be held fixed and general).
4. Report the hit-rate difference. A worst-picked arm, if capacity allows, brackets the effect and
   guards against "the model just picked easy chemistry".

**Why random is the right control:** it is what a chemist without the model actually does when
laying out an exploratory plate. Model-vs-random *is* the claim.

**Design notes that protect the result:** substrate selections must be structurally disjoint where
possible; the model is frozen and scores committed **before** any well is run (pre-registration in
the lab notebook); plate position/edge effects randomised across arms.

**Open scientific questions to resolve on the way:**
- Which reaction classes have enough literature signal? (`PROSPECTIVE_REACTION_CONDITION_GROUPING.md`)
- Does the amide result generalise, or is it class-specific? (the HiTEA discrepancy says: not yet known)
- Does conditioning on a literature-consensus condition cluster beat a single fixed condition set?
- Is the SCAR assumption the binding constraint, and do beyond-SCAR methods help reliably? (so far: no)

---

## 4. Working agreement

- **`LAB_JOURNAL.md` is append-only.** New dated, author-tagged entry at the *top*. Never rewrite an
  existing entry. Log every experiment: context/hypothesis → what I did → result → interpretation → next.
  This is the record a paper's SI is reconstructed from; treat it as a lab notebook, because it is one.
- **Decisions** get their own file: `docs/decisions/NNN-title.md` (ADR-style).
- **Literature notes:** one file per paper in `docs/literature/notes/<prefix>-<slug>.md`.
- **Negative results are recorded, not discarded.** The prop_ips-fails-on-BH entry is exactly as
  valuable as the win — it is what keeps the paper's claims defensible under review.
- **Never tune on TEST.** Model selection on validation only; test touched once. Any result produced
  by changing evaluation after seeing a score is not a result.
- Personal Claude memory is *not* shared with collaborators — anything a collaborator needs goes in
  this repo.

**Provenance.** This repo was split from a shared Glorius-group working tree; the foundational
commits are by **Florian Boser** and remain attributed to him in `git log`. The earlier work was done
during a hackathon that was **not submitted**; that framing is retired and no longer applies. What
carries forward is the science, not the event.

---

## 5. Coding standards (non-negotiable)

- **Package manager: `uv`.** Never bare `pip`/`python`. Use `uv run python ...`, `uv add`, `uv sync`.
- **Python 3.13** (`pyproject.toml`, `requires-python >=3.13`).
- **Object-oriented, typed, documented.** Full annotations; **Google-style docstrings**; comments for
  non-obvious logic. `scripts/` and `scripts_JSP/` hold thin CLI entrypoints that call into `src/`.
- **Quality gates:** `uv run ruff check . && uv run ruff format . && uv run mypy src && uv run pytest`
- **Reproducibility:** raw data immutable; derived data regenerated by scripts with a provenance
  record; fixed seeds; no secrets. Every number in `LAB_JOURNAL.md` must trace to a file in `artifacts/`.

---

## 6. Data map

| Path | What it is |
|---|---|
| `datasets/raw/HiTEA`, `datasets/raw/Chemlex` | **Immutable** source HTE datasets (36 MB). Not in git — irreplaceable, back up separately. |
| `datasets/processed/amide_corpus_ord.*` | Amide positives mined from ORD + ORDerly — 194,625 detected → 52,968 unique pairs. **Open-licensed**, publishable. |
| `datasets/processed/private/` | **Pistachio / CHORISO / USPTO / ORDerly parsed corpora (597 MB). LICENSED — never commit, never publish.** |
| `datasets/processed/features/` | Feature caches (ECFP/DRFP, ZINC availability). Regenerable. |
| `final_datasets/` | Assembled TEST / P / U. `final_datasets/private/` is licensed — never publish. |
| `artifacts/` | Experiment outputs — **tracked in git** (~2 MB); every journal number traces here. |

> ⚠️ **`DRFP`/`QM` columns in the original curated CSV are truncated NumPy string reprs** — not usable
> as features. Recompute DRFP from SMILES; RDKit for descriptors. See `docs/datasets.md`.

**Licensing discipline matters for publication:** results derived from Pistachio can be *reported*,
but the corpus cannot be redistributed. Anything that must ship with the paper has to be reproducible
from **ORD/ORDerly/USPTO (open)** — which is why `amide_corpus_ord.*` is kept separate from `private/`.

---

## 7. Directory layout

```
CLAUDE.md                 # this charter
LAB_JOURNAL.md            # append-only experiment log (the lab notebook)
RESULTS.md                # consolidated retrospective results
PROSPECTIVE_PLATE_PLAN.md # ← the wet-lab validation experiment (paper's core)
PROSPECTIVE_REACTION_CONDITION_GROUPING.md  # picking the reaction class + condition cluster
Future_datasets.md        # candidate public test sets
src/pubench/              # PU benchmark: models, features, mining, eval  (primary)
src/bhpu/                 # earlier pipeline: chem, features, leakage, space  (foundational)
scripts_JSP/              # CLI entrypoints + SLURM sbatch files
artifacts/                # experiment outputs (tracked)
docs/decisions/           # ADRs
docs/literature/notes/    # one note per paper (25)
```

**SLURM:** heavy compute goes to the cluster at `--nice=1000000`. See `docs/slurm.md`.

---

## 8. Current status

**Phase: retrospective evidence secured; moving to prospective validation.**

Done — literature-PU pipeline (mining → virtual U → features → PU models → frozen eval), 9 PU methods
benchmarked, Chemlex amide win over the in-domain ceiling, HiTEA BH replication, leakage audits,
beyond-SCAR arm (negative result).

Next, in order:
1. **Pick the reaction class** for the wet-lab plate — needs the ORD/Pistachio/USPTO class × condition
   catalogue (`PROSPECTIVE_REACTION_CONDITION_GROUPING.md`). Requirement: strong literature signal,
   **no existing HTE dataset**, cheap stable stock.
2. **Freeze and commit** model + scores before running anything.
3. **Run model-picked vs random plates** under literature conditions.
4. **Draft.** Figure 1 = the Chemlex "beats the in-domain ceiling" panel; Figure 2 = the prospective
   plate result. Target JACS.
