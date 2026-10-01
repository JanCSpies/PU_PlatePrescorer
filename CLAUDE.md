# CLAUDE.md — PU_PlatePrescorer

> Loaded into every Claude Code session in this repo. Shared with Jan: project rules only.
> Personal preferences go in `~/.claude/CLAUDE.md`. Keep this file short, true, under 200 lines.

## 1. Project

**PU_PlatePrescorer** — Jan Christopher Spies and Luka Praprotnik Čurič, Glorius group,
Organisch-Chemisches Institut, University of Münster. Target: a full paper in JACS or higher.

**Thesis.** Positive-Unlabeled learning on **real literature reactions** plus a **virtual
(enumerated) unlabeled substrate space** pre-scores an HTE substrate matrix well enough to **raise
the plate hit rate** — including for reaction classes with **no existing HTE dataset**.

**Prior group work.** PAYN — *"Positivity is All You Need"*, J. Am. Chem. Soc. 2026, 148,
15066–15075, doi:[10.1021/jacs.6c00127](https://doi.org/10.1021/jacs.6c00127), code
<https://github.com/GloriusGroup/PAYN> — used **synthetic** P and U carved from HTE sets. TACIT —
feasibility from positive-only data (under revision). This project uses real literature positives,
a virtual U and real reporting bias, and validates **prospectively**.

**Validation:** validation split → held-out public HTE test, touched once (Chemlex, HiTEA) →
**prospective plate, model-picked vs random**, model and scores frozen and committed before any
well is run.

## 2. Status and branches

- The AI-assisted prototype is archived as tag **`v0-prototype`** (commit `4dd8481`). Its results,
  including the Chemlex 0.768 ROC-AUC, are **unverified targets** to reproduce or explain.
- **Never copy code from `v0-prototype`.** Read it for ideas only:
  `git worktree add ../PU_v0 v0-prototype`.
- `main` = archived prototype line. **`luka/rebuild-clean-slate` = rebuild base** (PR #1). Until
  it is merged, work is committed directly on it in small, one-topic commits; never on `main`.
- Plan: `docs/design/rebuild.md`. **A stage is built only after that file describes it** (inputs,
  outputs, tools, verification) and Jan and Luka agree.
- Ownership: Luka — data side (ingest → standardise → annotate → catalogue → positives table).
  Jan — models, evaluation, plate. The positives table is the hand-over.

## 3. Commands

    uv sync                     # install (Python 3.13; uv only, never bare pip)
    uv run pytest               # tests
    uv run ruff check . && uv run ruff format . && uv run mypy src && uv run pytest   # before every commit

Dev data (open, 2,000-row samples): `~/University/Phd/Dataset/Datasets/samples_2k/`. The Pistachio
sample is licensed and lives apart in `samples_2k/private/`. Full corpora: cluster,
`/mnt/datasets/`. Paths come from config or arguments, never hard-coded.

## 4. Keep the repository clean

The prototype became unreadable through leftover files. These rules prevent it.

- **Only these are tracked:** `src/`, `tests/` (with small open fixtures in `tests/data/`),
  `configs/`, `scripts/`, `docs/`, `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`,
  `CLAUDE.md`, `LAB_JOURNAL.md`, `LICENSE`, `.gitignore`. Anything else needs agreement first.
- **Never commit** data (`*.csv`, `*.parquet`, `*.tsv`, `*.smi`), run outputs, images, logs,
  notebooks, model files, caches, `.venv/`, `scratch/` or `private/`. Run outputs go outside the
  repo or into a git-ignored folder.
- **Experiments live in `scratch/`** (git-ignored). Code moves into `src/`
  only when it is reused, tested and understood; then the scratch copy is deleted.
- **No new top-level folder or file without agreement.** When one is added, update §10 in the same
  commit.
- **Stage files by name:** `git add path/to/file`. Never `git add .` or `git add -A`.
- **Before every commit:** run `git status` and `git diff --staged`; every changed file must belong
  to this task. Unrelated changes (formatting sweeps, renames) get their own commit.
- **Delete, don't keep:** no `_old`, `_v2`, `_final`, `backup`, `test2` files, no commented-out
  code. Git keeps the history.
- **Small, focused commits** with prefixes: `feat`, `fix`, `test`, `docs`, `chore`, `refactor`.
  One task per branch; PRs of about 100–300 changed lines.

## 5. Keep the code simple (no spaghetti)

- **Stages talk through tables, not through each other.** Each stage reads one table and writes
  one table. Imports go one way only: ingest → standardise → annotate → catalogue. A module never
  imports from a later stage; no circular imports.
- **Separate I/O from chemistry.** Reading and writing files happens at the edges (adapters,
  scripts). Chemistry functions take values and return values; they never open files or read
  globals. That makes them testable.
- **Plain functions and `@dataclass(frozen=True)` records first.** A class only when it holds
  state used by several methods. No abstract base classes, registries, factories, plugin systems
  or "generic" helpers until a third concrete case needs them.
- **Small units:** functions about 40 lines, at most 5 parameters, nesting at most 3 levels;
  modules under about 300 lines. Ruff enforces complexity (`max-complexity = 10`).
- **No duplicated logic.** Chemistry logic (parsing, standardising, keys) exists in exactly one
  function; everything else calls it.
- **No hidden state:** no mutable module-level variables, no global config. Settings are passed in.
- **Names say the chemistry:** `reactant_smiles`, `agent_smiles`, `product_mol`, `n_rejected` —
  never `r`, `a`, `tmp`, `df2`. SMILES strings end in `_smiles`, RDKit molecules in `_mol`.
- **Explicit over clever:** no metaprogramming, no deep inheritance, no comprehension that needs a
  second read. If it is hard to explain, rewrite it.
- **Errors never pass silently:** no bare `except`; catch the specific exception and record why
  the row was rejected.
- Full type hints; Google-style docstrings that say *why*. `logging` in `src/`, never `print`.
- **Ask before adding a dependency.** Prefer RDKit, pandas, pyarrow and what `pyproject.toml`
  already has; test a new library on our hard cases first.
- Tables are pandas/pyarrow; data files Parquet, config YAML. No custom formats.

## 6. Cheminformatics workflow

- **Never trust a raw SMILES.** Every molecule goes parse → sanitise → standardise → canonicalise
  through one function. `Chem.MolFromSmiles` returning `None` is a rejected row with a reason.
  No `sanitize=False` without a written reason.
- **Keep the raw next to the processed:** every row keeps source, source ID, licence and the
  original string, so any result traces back to its record.
- **Identity by key, not string:** molecules by canonical SMILES + InChIKey, reactions by a
  canonical reaction key. Deduplicate across sources before counting anything.
- **Every stage reports** rows in, rows out, rows rejected by reason, and saves the rejected rows.
- **Look at the chemistry:** when a step changes molecules, save a grid image of about 20 changed
  examples (before → after) for a human to check. Numbers alone do not verify chemistry.
- **Chemistry choices live in config** (`configs/*.yaml`): reaction classes, condition families,
  salt rules, the definition of a positive. Code reads them; it does not hard-code them.
- **One standardisation for P, U and TEST;** every row carries `std_version`.
- **No leakage:** evaluation sets never enter training; check overlap by InChIKey and report it.
  Evaluation splits keep test substrates unseen.
- **Samples first:** develop on `samples_2k/`; run full corpora only after the sample run is checked.

## 7. Tests

- **Tests first, then code.** Every module in `src/` has tests in `tests/`.
- Chemistry tests are named after the case: `test_naoh_covalent_equals_ionic`,
  `test_k2co3_differs_from_cs2co3`, `test_orderly_row_without_temperature`.
- Fixtures are a few hand-checked rows from **open** sources only, small (under about 100 KB).
- **Nothing merges that its author cannot explain line by line.**

## 8. Working agreement

- Branch + pull request for every change; reviewed before merge.
- `LAB_JOURNAL.md` is append-only: new dated, author-tagged entry at the top (context → done →
  result → interpretation → next). Never rewrite an old entry.
- Decisions: `docs/decisions/NNN-title.md`; accepted ones are superseded, never rewritten.
- Literature notes: one file per paper in `docs/literature/notes/`.
- Negative results are recorded. Never tune on TEST.
- Reproducibility: raw data immutable; derived data regenerated by scripts with a provenance
  record; fixed seeds; no secrets.
- Personal Claude memory is not shared; anything a collaborator needs goes in this repo.

## 9. Data and licences

- Raw HTE sets (Chemlex, HiTEA) and literature corpora are **never committed**; read in place.
- Open: ORD, ORDerly, USPTO, CHORISO, CJHIF, NiCOlit (MIT).
- **Licensed: Pistachio and everything derived from it** — only under a `private/` folder
  (git-ignored at any depth), never in tests, never published.
- Everything the paper ships must be reproducible from open sources alone.

## 10. Layout (only what exists — update in the same commit when it changes)

    src/platepu/      package, grows one stage at a time
    tests/            tests for every module
    docs/design/      rebuild plan and stage designs

**Provenance.** Split from a shared Glorius-group working tree; foundational commits by
**Florian Boser** remain attributed in `git log`.
