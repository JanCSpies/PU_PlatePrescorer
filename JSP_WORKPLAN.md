# JSP Workplan — condition-aware ORD-BH positives for literature→JnJ transfer

> **Handoff doc for a fresh Claude Code session started inside this worktree
> (`/home/student/j_spie17/claude_hackathon_JSP2`, branch `JSP`, author = Jan Spies).**
> It carries the full context from the design conversation that produced this branch, so you can
> kickstart without re-deriving anything. Read §0 → §4 before writing any code.

---

## ⚡ STATUS UPDATE — 2026-07-08 (post-merge)

> Reconciles this doc with what actually landed on `master`. Read this before §0.

**Worktree is healthy and connected.** Linked worktree of the main repo
(`/home/employee/fboser/fboser_mydata/hackathon_claude`, `[master]`); `.git` → shared object store;
symlinks (`external/`, `gpuma/`, read-only `datasets/processed/{test_jnj_hte_2024,literature_hte}.*`)
all intact. `git worktree list` shows both checkouts.

**Path / worktree history (READ — the path changed twice).** This is a **fresh worktree at
`/home/student/j_spie17/claude_hackathon_JSP2`**, created 2026-07-08 on branch `JSP`. It replaces an
earlier worktree at `/home/student/j_spie17/claude_hackathon_JSP` whose `.venv` had become inconsistent
(a half-deleted state from an NFS `rm` that couldn't remove open-handled files); that old worktree was
renamed aside to `/home/student/j_spie17/.trash_claude_hackathon_JSP_2600802` and `git worktree prune`d
(safe — branch `JSP` and all commits live in the shared object store, unaffected). The doc's *original*
path `/home/student/j_spie17/JSP` was never the real worktree either — it's an **orphan holding only a
stray `.venv`** (ignore/delete both it and the `.trash_…` dir when their file handles free up).
If a shell here shows `VIRTUAL_ENV` pointing at any of those old paths, `unset VIRTUAL_ENV`; `uv` uses
this worktree's own `.venv`. **This worktree's symlinks (`external/`, `gpuma/`, the two
`datasets/processed/*` pairs) and Jan's per-worktree git identity were recreated and verified.**

**Master merged (First Action #1 = DONE).** Fast-forward `7f7a27f → e0d4835` (no conflicts). Two
commits arrived — and **Florian already implemented most of this branch's Phase-1 "net-new" code**:
- `bc92b03` — **condition canonicaliser** `src/bhpu/chem/conditions.py`: `CanonicalCondition`,
  `extract_ligand` (precatalyst RuPhos-Pd-G4 ⇄ free RuPhos → same key), `base_anion`
  (Cs₂CO₃ = K₂CO₃), `condition_from_columns` (HTE) + `condition_from_reagents` (mined bag).
- `e0d4835` — **miner Pd-filter + condition capture** in `src/bhpu/data/mining.py`: `MinedReaction`
  now carries `ligand_smiles/base_smiles/solvent_smiles`; `detect(require_pd=True)` rejects
  Cu-SNAr/Ni; each match gets a unified `condition_from_reagents`. New ADR
  **`docs/decisions/006-pd-bh-conditions.md`** + `AUTONOMOUS_RUN.md`.

**⚠️ ADR-006 reframes this branch's central thesis — read it.** This doc (built on ADR-005) argued
*"FP crashes transfer → descriptors are THE lever."* ADR-006 (Florian + Jan, 2026-07-08) argues the
FP crash was caused by **disjoint condition vocabularies + Cu-SNAr pollution**, and that fixing those
(unify via canonicaliser + `require_pd`) makes the vocabularies **overlap** (ORD covers 47 % of JnJ
ligands; 73 % of ORD ligand-bearing couplings use a JnJ ligand). So ADR-006's *first* move is
**unified-condition ECFP** (ligand/base/solvent ECFP + metal), with substrate/condition **descriptors
demoted to a later re-test**, not the headline. Decide with Jan which framing leads (see §8.1a below).

**What is now DONE vs still net-new (supersedes §6):**
- ✅ Miner extended (`MinedReaction` conditions, `require_pd`, unified condition attach) — *Florian*.
- ✅ Condition canonicaliser (`bhpu.chem.conditions`) — *Florian* (this doc expected to merge it in).
- ✅ **Dataset assembly (TEST / P / U) — Jan, 2026-07-08.** New self-contained
  `scripts_JSP/assemble_datasets.py` (`DatasetAssembler`) + `tests/test_assemble_datasets_jsp.py`.
  Writes `final_datasets/{test,positives_ord_bh,unlabeled_grid}.parquet` + `assembly_report.json`, all
  in **one unified condition schema** (`metal`, `ligand_smiles/key`, `cond_base_smiles/key`,
  `cond_solvent_smiles/key`). No edits to shared `src/`. Gates: ruff+mypy+pytest(5) ✅.
- ✅ **Pd-BH re-mine RAN** → `final_datasets/positives_ord_bh.parquet`: **211 positives** (per
  `(pair,condition)`) from 832,809 ORD-forward rows, leakage-clean (0 exact JnJ pair hits); **57%
  ligand overlap with JnJ**. Dedup is now per-`(pair,condition)`, not pair-level.
- ✅ **`iter_orderly` solvent question RESOLVED:** the shipped reader **DROPS `solvent_*`** (bug); the
  new assembler reader pools reactants+agents+**solvents** into the bag so solvents are captured
  (82% of P rows carry a solvent). ⚠️ *Data-quality note for Florian:* `base_anion` maps ~26 ORD bases
  to `O`/water → real base coverage ≈17%; and P is small (211) — grow it with **Pistachio**.
- ⬜ **Condition-aware transfer harness** (generalise `scripts/17` to condition-spanning U; feed it
  `final_datasets/`, unified-ECFP first per ADR-006) — **next**, pending.
- ⬜ **Substrate/condition descriptor featurizer** — Florian owns (descriptors); ADR-006 *step 4 re-test*.

**Gate status (post-merge, fresh venv):** `uv run ruff check .` ✅. ⚠️ `uv run pytest` (with only
`--extra dev`) **fails at collection** on the merged-in `tests/test_sites_and_descriptors.py`, which
imports `morfeus` — that dep lives in the **`qm` extra**, so run `uv sync --extra dev --extra qm`
before pytest (or the suite can't collect). `mypy` not yet run in this worktree. This is a
merged-code/extras issue, not a regression from the swap.

---

## 0. TL;DR — what this branch is for

Florian's shipped model predicts BH C–N **feasibility** from literature data via PU learning, scored
on the **JnJ HTE 2024** plate. Its headline (0.72 ROC-AUC, 2.3× plate uplift) is **substrate-only**
(aryl⊕amine ECFP) and **transductive**. Two problems motivated this branch:

1. **Conditions are ignored, but they *are* the chemistry.** Florian's own ADR-005 shows conditions
   drive **43 %** of JnJ label variance and add **+0.16 AUC in-domain** — yet encoding them as
   *structural fingerprints* **crashes** literature→JnJ transfer (0.69 → **0.26**), because literature
   and JnJ share almost no catalyst/base/solvent identities (the DRFP domain-shortcut again).
2. **The positive set is incoherent for PU.** The positives are HTE-plate wins with their measured
   negatives thrown away — but an HTE plate *has* negatives, so treating it as "positive-only" is a
   simulation, not the real PU problem. The honest positive-only source is **reported literature
   (ORD)**, where negatives genuinely don't exist.

**This branch = the missing piece of ADR-005.** ADR-005 concluded the fix is **condition
DESCRIPTORS** (a shared *physical* space) but noted *"the positives still don't use JnJ's
conditions."* We give the positives real conditions and put them in that shared space:

- **(a)** Re-mine ORD to recover **per-reaction conditions** (the current ORD corpus threw them away).
- **(b)** Keep only genuine **Pd-catalysed BH** couplings as positives (filter Pd + phosphine).
- **(c)** Encode conditions as **descriptors** (ligand/base/solvent physical params), not fingerprints,
  and score a **condition-aware U**.
- **(d)** **Keep** the existing substrate-disjoint VAL/TEST split unchanged.

Hypothesis: giving positives real Pd-BH conditions in a shared descriptor space lets the +0.16
condition signal **transfer** instead of crashing — beating the substrate-only 0.72.

---

## 1. Environment & working rules (read before running anything)

- **You are in a git worktree**, isolated from Florian's shared checkout:
  - Your folder: `/home/student/j_spie17/claude_hackathon_JSP2` (earlier paths
    `/home/student/j_spie17/JSP` and `…/claude_hackathon_JSP` are dead — see the STATUS UPDATE),
    branch `JSP`, commits authored as
    **`Jan Spies <jan.chris.spies@gmail.com>`** (per-worktree identity via `extensions.worktreeConfig`).
    Verify with `git var GIT_AUTHOR_IDENT`.
  - Florian's shared folder `/home/employee/fboser/fboser_mydata/hackathon_claude` is on `master`
    and commits as Florian — **do not edit files there**. `git worktree list` shows both.
- **Package manager `uv`** is on PATH (`~/.local/bin/uv`, your own cache). The `.venv` here is yours
  (`bhpu` resolves to *this* worktree's `src/`). Run everything via `uv run …`.
  - Quality gates (must pass): `uv run ruff check . && uv run mypy src scripts && uv run pytest`.
- **Symlinks already set up:** `external/`, `gpuma/`, and read-only
  `datasets/processed/{test_jnj_hte_2024,literature_hte}.{parquet,csv}` point at the shared tree.
  **Write your NEW outputs (re-mined corpus, features) into this worktree's own `datasets/processed/`
  — never overwrite the shared files.**
- **Raw ORD is on `/mnt` (absolute paths, read-only):** `/mnt/datasets/ORDerly/orderly_{forward,
  condition}_train.parquet`, `/mnt/datasets/DynaMAIK/ord.csv`. Do **not** modify `/mnt`; copy in if needed.
- **Git rules:** never `git push` (only Florian pushes). Local commits encouraged. When JSP merges
  back, use a **real merge/rebase, not squash** (squash erases Jan's per-commit authorship →
  no GitHub credit). Jan must confirm `jan.chris.spies@gmail.com` is verified on his GitHub account.
- **Coding standards (see `CLAUDE.md` §3):** Python 3.13, OOP, full type annotations, Google-style
  docstrings, ruff+mypy strict, tests in `tests/`. `scripts/` are thin CLIs over `src/bhpu/`.
- **Heavy compute → SLURM** at `--nice=1000000` (see `docs/slurm.md`).
- **First housekeeping step — ✅ DONE 2026-07-08:** master was merged (fast-forward `7f7a27f →
  e0d4835`, no conflicts). Picked up `bc92b03` (condition canonicaliser) **and** `e0d4835` (miner
  Pd-filter + condition capture + ADR-006). See the STATUS UPDATE at the top. Keep coordinating; Florian
  is actively editing the same condition files — re-`git merge master` periodically before big changes.

---

## 2. Scientific context (brief; full detail in repo docs)

- **Goal:** predict feasibility (yield ≥ 10 %, binary) of Buchwald–Hartwig C–N couplings, training on
  literature/PU only (no JnJ labels in training), evaluated on JnJ HTE 2024 (Neves).
- **PU framing:** P = reported/positive reactions; U = unlabeled candidate space; recover a
  feasibility ranking. Anchor methods: **PAYN** (spy-PU under SCAR, our group) and **TACIT**.
- **The core difficulty = domain shift, not class-prior shift.** A naive P-vs-U classifier learns
  *"was this reported / is this a literature reaction?"* (domain-discriminator AUC ≈ 1.0), not
  *"does it work?"*. Fix = covariate-shift **importance weighting** (IW) toward the JnJ marginal +
  transductive U. See `docs/decisions/003-crossdomain-transfer-diagnosis.md`,
  `docs/literature/notes/theory-crossdomain-pu.md`.
- **Read these in the worktree:** `CLAUDE.md`, `RESULTS.md`, top of `LAB_JOURNAL.md` +
  `AUTONOMOUS_RUN.md` (roadmap + status checklist), `docs/decisions/003`, `004` (ORD corpus),
  **`005` (conditions) and `006` (Pd-BH + unified conditions — newest, reframes the descriptor thesis;
  read both together)**, `docs/literature/notes/pu-payn-boser-2026.md`.

---

## 3. Why the redesign (the argument, so you can defend it)

- HTE-positives-only is a **false PU premise** (HTE measures its negatives). PAYN's benchmark did this
  as a *simulation*; building the real pipeline on it is incoherent. → Use **ORD reported data** as P,
  where positive-only is real.
- If a source *has* negatives (HTE plates), the honest method is **supervised domain transfer**, not
  PU. Keep HTE as a possible *baseline/ceiling*, not laundered into P.
- The IW "win" (0.55→0.72) is a **transfer-learning** technique; the PU-specific spy/SCAR machinery
  added ~nothing. Even IW only used HTE negatives' *coordinates* (to shape the domain weighting),
  never their *failure labels* — so the negatives were wasted anyway.
- **Conditions matter for BH** (ADR-005: +0.16 in-domain). We *want* to predict per-well outcomes, not
  just substrate-pair feasibility, so conditions must be in the model — but encoded so they transfer.

---

## 4. ⚠️ What Florian ALREADY did (ADR-005) — READ, REUSE, DON'T DUPLICATE

Present in this branch (commit `7f7a27f`) — read these first:

- **`docs/decisions/005-condition-encoding.md`** and the **top `LAB_JOURNAL.md` entry** — the findings.
- **`src/bhpu/features/condition_fp.py`** — structural ECFP of catalyst/base/solvent SMILES.
  **This is the approach that CRASHES transfer (0.26). Do not re-propose reagent/structural condition
  fingerprints — it's already falsified.**
- **`scripts/17_condition_aware_eval.py`** — condition-aware transfer: rebuilds **U to span the JnJ
  condition space** (unmeasured pairs × conditions sampled from the JnJ condition distribution,
  ~85k rows), IW-PU, per-row condition-level target, transductive+inductive. **Reuse this harness**;
  swap its condition featurizer (FP→descriptors) and its positives (lit→ORD-BH-with-conditions).
- **`scripts/18_condition_diagnostic.py`** — in-domain probe (JnJ VAL→TEST, conditions shared):
  substrate 0.609 vs substrate+condition **0.769**. (Breaks train-purity by design; it's a probe.)
- **`scripts/16_pistachio_expansion.py`** — data expansion: lit / +ORD / +ORD+Pistachio →
  0.708 / 0.721 / **0.736**. (Pistachio is licensed → `datasets/processed/private/`, gitignored,
  never publish.)
- Newer master `bc92b03`: **condition canonicaliser** (unifies full Pd-precatalyst SMILES vs
  metal+ligand across sources) — merge it in; you'll need it to align ORD agents with JnJ catalysts.

**ADR-005's decision (adopt it):** conditions must be encoded via **descriptors** — a shared physical
space (ligand %V_bur / cone angle / electronics, base pKa, solvent polarity / coordinating
parameters) so a bulky-rich literature ligand and a structurally-different-but-similar JnJ ligand map
close together. Fingerprints can't bridge disjoint condition vocabularies; descriptors can. **No
reagent-descriptor code exists yet — that is the net-new lever, and it's this branch's core.**

---

## 5. The four decisions

| # | Decision | Status |
|---|---|---|
| a | Re-mine ORD (raw ORDerly + DynaMAIK) to recover **per-reaction conditions** | **LOCKED** |
| b | Keep only genuine **Pd-BH** couplings as positives | **LOCKED**; strictness = **Pd + phosphine** (recommended) — confirm with Jan whether to also require a BH base |
| c | **Condition-aware** features (**descriptors**, per ADR-005) + condition-spanning U (reuse `scripts/17`) | **LOCKED** (descriptors); U reuse recommended |
| d | **Keep** the existing substrate-disjoint VAL/TEST split (`FrozenEvalProtocol`) unchanged | **LOCKED** (Jan said "keep test set splitting"; confirm he did *not* mean the earlier "pragmatic TEST≈all-JnJ" variant) |

---

## 6. Reusable-code map (file paths) + what's net-new

**Reuse as-is:**
- `src/bhpu/data/mining.py` — `BHCouplingDetector` (product-confirmed aryl C–N), source readers
  `iter_orderly`/`iter_ord_csv`, `CNCouplingMiner`, leakage guard. **Extend, don't rewrite.**
- `src/bhpu/leakage/audit.py` `LeakageAuditor` and `src/bhpu/eval/protocol.py`
  `FrozenEvalProtocol` (JnJ substrate/pair reference; `assert_training_clean`; split unchanged per (d)).
- `src/bhpu/features/composite.py` `CompositeFeaturizer` (concatenate substrate ECFP ⊕ condition
  descriptors); `src/bhpu/features/fingerprints.py` `ECFPFeaturizer` (substrate half).
- `src/bhpu/data/schema.py` — `CONDITION_SMILES_COLUMNS`, `CONDITION_NAME_COLUMNS`,
  `ConditionSet`, `GENERAL_CONDITIONS`; JnJ real vocab = 32 cat × 8 base × 6 solv, 1,137 combos.
- Florian's `condition_fp.py` (as a *baseline to beat*, not the solution), `scripts/16/17/18`,
  condition canonicaliser (`bc92b03`).
- `src/bhpu/eval/transfer.py` `TransferExperiment` / `DomainDiscriminator` / IW-PU / `scripts/17`.

**Net-new (this branch's work):** *(⚠️ partly landed by Florian on `e0d4835` — see STATUS UPDATE. The
`MinedReaction` condition fields, `require_pd` filter, and the whole canonicaliser now EXIST; do not
re-implement. What remains below is the RUN + emission + harness + descriptor re-test.)*
1. **Conditions-preserving Pd-BH re-mine.** ✅ *code half done:* `MinedReaction` already carries
   `ligand/base/solvent` (unified via `condition_from_reagents`) and `detect(require_pd=True)` exists.
   ⬜ *Still to do:* (i) **run** it → write `datasets/processed/cn_corpus_ord_bh.parquet` (new name;
   no `scripts/20` yet — add a thin CLI); (ii) `_PairAggregate` dedup is still `(aryl_key, amine_key)`
   — change the key to include a **condition signature** or **emit one row per `(pair, condition)`**;
   (iii) verify `iter_orderly` actually puts the `solvent_000…`/`agent_*` columns into the reagent bag
   `condition_from_reagents` reads (else the solvent slot is silently empty); (iv) optional temp field.
   Note the current `require_pd` test is **Pd-presence only** (`"Pd"`/`"[Pd"` substring), *not* yet
   Pd+phosphine — see §8.1. Keep the diff additive (`mining_bh.py` / `scripts/20_mine_bh_conditions.py`).
2. **Condition DESCRIPTOR featurizer** (the ADR-005 lever, no code exists — but ADR-006 demotes it to a
   *re-test after* trying unified-condition ECFP first; confirm ordering with Jan, §8.1a). Keyed on
   `CONDITION_SMILES_COLUMNS`: ligand steric/electronic params (e.g. %V_bur, cone angle, donor
   count, could reuse `gpuma`/morfeus machinery — all current descriptor code is substrate-only),
   base pKa, solvent polarity/dielectric/coordinating flags. Must map ORD and JnJ conditions into the
   **same** continuous space (use the `bc92b03` canonicaliser to normalise precatalyst vs metal+ligand
   first). Compose `ECFPFeaturizer(substrates) ⊕ ConditionDescriptorFeaturizer` via `CompositeFeaturizer`.
3. **Wire it into the transfer harness:** ORD-BH-with-conditions as P, descriptor features for P/U/JnJ,
   reuse `scripts/17`'s condition-spanning U and IW-PU; report the domain-discriminator AUC (does the
   shared descriptor space *lower* it vs the FP crash?), VAL-select, TEST once.

---

## 7. Implementation plan (phased)

**Phase 0 — sync & orient.** Merge master (`bc92b03`), read §4 files, run the gates to confirm a green
baseline in the worktree.

**Phase 1 — Pd-BH re-mine with conditions** (decisions a, b). Extend mining; add Pd+phosphine filter;
per-condition dedup; write `datasets/processed/cn_corpus_ord_bh.parquet` (new name) into the worktree;
re-run leakage audit vs JnJ (`assert_training_clean` must pass; expect `exact_jnj_pair_hits = 0`).
Report corpus size, Pd/BH counts, condition coverage.

**Phase 2 — condition descriptors** (decision c). Build the descriptor featurizer; canonicalise
conditions first; verify ORD and JnJ conditions land in overlapping descriptor regions (sanity: the
domain-discriminator AUC on descriptor-space conditions should be *well below* the 1.0 that FPs give).

**Phase 3 — condition-aware transfer** (decision d keeps the split). Reuse `scripts/17`; P =
ORD-BH-with-conditions (descriptors), U = condition-spanning JnJ grid, features = substrate ECFP ⊕
condition descriptors; run naive-PU / spy(PAYN) / IW-PU; select on VAL AUPRC; **touch TEST once**.
Headline question: does it beat substrate-only 0.72 and avoid the FP 0.26 crash?

**Phase 4 — evaluate & document.** Metrics + plate demo; write a new ADR (`006-…`), append to
`LAB_JOURNAL.md` (tag Jan), update `RESULTS.md`. Keep the substrate-only 0.72 and the FP-0.26 crash as
explicit comparison rows.

---

## 8. Open decisions to confirm with Jan (don't assume)

1. **Pd-BH strictness:** Pd + phosphine (recommended) vs also-require-BH-base (stricter, smaller).
   *Update:* ADR-006 decided "require a Pd species, reject Cu/Ni/metal-free", and the shipped
   `require_pd` is **Pd-presence only** (substring test) — it does **not** yet require a phosphine.
   Confirm whether to tighten to Pd + phosphine (± BH base) before the re-mine, or mine Pd-only and
   filter later.
   - **1a. (NEW) FP vs descriptors — which leads?** ADR-005 said descriptors are the lever (FP
     falsified); ADR-006 says the FP crash was disjoint-vocab + Cu-pollution and now recommends
     **unified-condition ECFP first**, descriptors as a re-test. Confirm the order: (A) ECFP-of-unified-
     condition first, descriptors only if it still transfers poorly (ADR-006), or (B) go straight to
     descriptors (original branch thesis). Recommend (A) — it's cheaper and directly tests ADR-006.
2. **Descriptor set / source:** which ligand/base/solvent descriptors, and whether to compute via
   `gpuma`/morfeus (GPU) or a lighter RDKit/curated-table route (timebox — descriptor curation is a
   rabbit hole).
3. **(d) exactly:** keep the current substrate-disjoint split (assumed) vs the earlier "pragmatic
   TEST ≈ all-JnJ, small VAL" idea.
4. **Scope vs Florian:** ⚠️ *already shifted* — Florian landed the miner Pd-filter/condition-capture
   **and** the canonicaliser on master (see STATUS UPDATE). Re-confirm the split so JSP owns the
   **re-mine run + reaction-level `(pair,condition)` corpus + condition-aware transfer harness +
   descriptor re-test**, and doesn't collide with Florian's ongoing condition-spanning-U / Pistachio work.

---

## 9. Verification & quality gates

- Unit tests (`tests/`): re-mine retains conditions & Pd filter behaves; descriptor featurizer output
  dims/alignment; ORD and JnJ share a descriptor space (discriminator AUC ≪ 1.0); `CompositeFeaturizer`
  concatenation; leakage `assert_training_clean` passes on the ORD-BH corpus.
- End-to-end: run the transfer on **VAL only** first, sanity-check, then TEST once.
- Always: `uv run ruff check . && uv run mypy src scripts && uv run pytest` before committing.
- Honesty: report transductive vs inductive; don't oversell; keep comparison rows (substrate 0.72,
  FP-crash 0.26, in-domain +0.16).

---

## 10. First actions for the new session

1. `git merge master` (get `bc92b03` canonicaliser); run the gates.
2. Read: this doc → `docs/decisions/005-condition-encoding.md` → `LAB_JOURNAL.md` (top) →
   `scripts/17_condition_aware_eval.py`, `18_condition_diagnostic.py`, `src/bhpu/features/condition_fp.py`
   → `src/bhpu/data/mining.py`.
3. Confirm the §8 open decisions with Jan.
4. Start **Phase 1** (Pd-BH re-mine with conditions) — the net-new foundation everything else needs.
