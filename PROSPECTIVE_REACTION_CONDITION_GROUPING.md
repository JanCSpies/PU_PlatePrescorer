# Prospective plan — group ORD/Pistachio/USPTO into (reaction class) → (condition cluster) for plate selection

> **Status:** plan for review, 2026-07-11 (Jan's session, claude). NOT yet running — awaiting go.
> **Goal:** produce a **browsable, ranked catalogue** of *"same reaction, commonish conditions"*
> groups mined from the three big general reaction databases, so a chemist can eyeball the groups
> and choose which transformation + condition regime to build the first prospective plates for.
> **Companion:** `PROSPECTIVE_PLATE_PLAN.md` (this is its **Phase 0**, taken further into condition
> clustering); `CLAUDE.md` §4 (data map); `docs/literature/notes/data-sources-catalog.md`.

---

## 0. What "group into common reactions, then similar conditions" means here

Two nested groupings + a dedup, producing one row per browsable group:

```
Tier 1  reaction CLASS          e.g. "Suzuki–Miyaura", "Buchwald–Hartwig", "amide coupling"
  Tier 2  CONDITION CLUSTER     same class, coarse-matched (metal, ligand-family, base-family,
                                 solvent-class, temp-bin) — "small deviations OK" = the binning
    → metrics for plate choice   N rxns · N unique substrate pairs · substrate diversity ·
                                 yield distribution (headroom) · source breakdown · novelty · example
```

A **condition cluster** is a candidate plate: one transformation, one near-fixed condition regime,
many substrate pairs → exactly a substrate matrix you can run.

---

## 1. Sources, scale, and format reality (verified on `/mnt/datasets`)

| Source (file) | Rows | Conditions | Yield | Notes |
|---|---:|---|---|---|
| **ORDerly** `orderly_condition_train.parquet` | 0.63 M | ✅ **role-typed** (`agent_/solvent_/temperature`) | ✅ `yield_000` | cleaned ORD; the gold standard for conditions |
| **Pistachio** `pistachio.smi.deduped.csv` | 7.25 M | ⚠️ in `>agents>` block, must parse | ✗ | atom-mapped `rxn_smiles` only, **no class labels** |
| **USPTO** `DynaMAIK/uspto_76_16.csv` + `uspto_01_16.csv` | 1.81 M + 1.94 M | ⚠️ in `>agents>` block | ✅ `TextMined/CalculatedYield` | atom-mapped, patent+year provenance |
| **ORD (raw)** `DynaMAIK/ord.csv` | 2.37 M | ✗ reagents lumped into reactants | ✗ | **skip in favour of ORDerly** (same origin, uncleaned) |

**Overlap warning (why dedup is mandatory if mixed):** all four are patent-derived; USPTO ⊂ ORD ⊂
Pistachio to a large degree. Naive concatenation would triple-count the same reaction and inflate
every group count. **→ we dedup on a canonical reaction key and keep provenance** (see §3, Stage 2).

**Recommended source set:** Pistachio (broadest coverage) + ORDerly (best conditions & the ORD
representative) + USPTO 1976–2016 (classic, atom-mapped, carries yields). ~11 M raw → dedup.

---

## 2. Reusable machinery already in the repo (low incremental cost)

| Need | Reuse |
|---|---|
| stream a source → `RawReaction(reactants, agents)` | `scripts_JSP/assemble_datasets.py` `_iter_ord` / `_iter_pistachio` (+ add `_iter_uspto`, `_iter_orderly`) |
| SMARTS reaction detection + multiprocessing batch | `src/pubench/mining/amide.py` (generalise its 1-class detector to N classes) |
| canonical conditions (metal / ligand / base / solvent → InChIKey) | `bhpu.chem.conditions.CanonicalCondition` + `condition_from_reagents` (expand its BH-tuned solvent/base vocab to general) |
| curated base (13) / solvent (18) tables | `src/pubench/features/condition_descriptors.py` |
| SLURM courtesy pattern | `scripts_JSP/*.sbatch`, `--nice=1000000`, `docs/slurm.md` |

---

## 3. Method (3 stages; heavy stage runs on SLURM, per-source in parallel)

### Stage 1 — parse + classify + conditionise each source  *(heavy; one SLURM job per source)*
For every reaction, chunked + multiprocessed with RDKit, emit:
- **`rxn_key`** — dedup key: atom-map-stripped, salt-stripped, RDKit-canonical
  `frozenset(reactants) >> frozenset(products)`.
- **`class`** — matched against a curated **named-reaction SMARTS library** (§4). Interpretable,
  plate-aligned. No match → `other` (kept, collapsed in the ranked view). Mapped template atoms also
  tag **which molecule is which coupling partner** (e.g. halide vs boron) — this is what later turns a
  group into a substrate matrix.
- **`cond`** — `CanonicalCondition`: metal(s), ligand, base, solvent (canonical InChIKeys) + a
  temperature bin. From ORDerly's role columns directly; from `>agents>` bag for the others.
- **`yield`**, **`source`**, patent/year/id provenance where present.
- **`aryl/partner` substrate SMILES** for the matched class (the plate axes).

Output: `datasets/processed/private/rxn_groups/<source>_parsed.parquet`.

### Stage 2 — merge + dedup across sources  *(1 job)*
- Concatenate the per-source parquets; **group by `rxn_key`**; keep one representative; record
  `sources` (set), `n_reports` (report frequency = a popularity/robustness signal), and aggregate the
  yield stats. A reaction reported by all three sources is a high-confidence group member.
- Output: `rxn_unified.parquet` (deduped) — **plus** per-source counts retained as columns so a
  "mixed" and a "per-source" view are both available.

### Stage 3 — build the browsable group catalogue  *(1 job, fast)*
- **Tier-1** group = `class`. **Tier-2** = coarse `cond` tuple (the "commonish conditions" cluster);
  optionally a finer exact-condition sub-cluster inside each.
- Per group compute the **plate-selection metrics**: N reactions, N unique substrate pairs, unique
  partners per axis (matrix size + diversity), **source breakdown**, **yield distribution** (median,
  fraction ≥ threshold → expected base-rate / headroom), **condition-homogeneity** (share of the class
  that falls in this cluster), **novelty** vs our existing amide/BH corpora, and a representative
  reaction + condition string.
- Outputs:
  - `reaction_condition_groups.parquet` (one row per group) + `groups_summary.csv` (ranked, human-read),
  - a compact **HTML/Artifact** to actually browse: class → condition clusters → the metrics + an
    example, sortable so you can pick.

---

## 4. Named-reaction SMARTS library (the "common reactions" we detect)

Plate-relevant, bimolecular bond-forming families first (extensible; `mine_amide.py`'s acid+amine
template is template #1 already written):

- **C–C couplings:** Suzuki–Miyaura · Sonogashira · Heck · Negishi/Kumada · Stille
- **C–N couplings:** Buchwald–Hartwig · SNAr (N-nuc) · Chan–Lam · reductive amination · N-alkylation ·
  amide coupling · sulfonamide formation · urea/carbamate
- **C–O / C–S:** Williamson/O-alkylation · SNAr (O/S-nuc) · esterification · Mitsunobu
- **Collapsed to an `other`/`FGI` bucket (low plate value, high count):** deprotection (Boc/Cbz),
  ester/amide hydrolysis, reduction/oxidation, halogenation.

The ranked catalogue foregrounds the coupling families and collapses `other`, but every reaction is
still classified and counted so nothing is silently dropped.

---

## 5. Condition-cluster definition (tune this — it is the "how small a deviation is OK" knob)

Coarse tuple, each field family-binned so minor write-order / counter-ion / ratio differences collapse:
- **metal / catalyst family** — Pd / Ni / Cu / Pd+Cu / none …
- **ligand family** — exact ligand InChIKey, or phosphine/NHC/none family (precatalyst ≡ free ligand,
  already handled by `extract_ligand`).
- **base family** — carbonate / phosphate / hydroxide / alkoxide / amine / fluoride / acetate / none.
- **solvent class** — ethereal (THF/dioxane/2-MeTHF) / polar-aprotic (DMF/DMAc/NMP/DMSO) / protic
  (MeOH/EtOH/H₂O) / aromatic (toluene) / chlorinated / …
- **temperature bin** — rt / 40–80 / 80–120 / >120 °C (missing = its own bin).

Exact-identity sub-clusters are also emitted, so you can drill from "commonish" to "identical".

---

## 6. Deduplication (explicit — you flagged this)

- **If mixed:** dedup on `rxn_key` (§3 Stage 2). Provenance (`sources`, `n_reports`) kept, yields
  aggregated. This is the recommended default — one unified, honestly-counted catalogue.
- **Per-source view** stays available (columns), so you can also read a group as "reported by all 3"
  (robust) vs "Pistachio-only".
- Substrate-pair counts within a group are also deduped (unique canonical pairs), so a group's matrix
  size is real, not report-inflated.

---

## 7. Compute plan (SLURM)

| Job | Scope | Resources | Est. |
|---|---|---|---|
| `parse_pistachio` | 7.25 M | CPU, 32 cpu, 64 G, 8 h | heaviest |
| `parse_uspto` | 3.75 M | CPU, 32 cpu, 48 G, 4 h | |
| `parse_orderly` | 0.63 M | CPU, 16 cpu, 32 G, 2 h | |
| `merge_dedup_group` | after the 3 above | CPU, 16 cpu, 96 G, 2 h | Stages 2–3 |

All `--nice=1000000`, streaming/chunked so memory stays bounded, RDKit via multiprocessing. Parse
throughput ≈ hundreds–thousands rxn/s/core, so wall times above are conservative.

---

## 8. Decisions to confirm before I build it

1. **Reaction-class method** — curated **named-reaction SMARTS** (interpretable, plate-aligned)
   **[recommended]** vs. unsupervised atom-mapped reaction-center signature (exhaustive, less readable)
   vs. both (SMARTS primary, center to sub-split `other`).
2. **Sources** — Pistachio + ORDerly + USPTO 1976–2016, **dedup-merged with provenance kept**
   **[recommended]**; drop raw `DynaMAIK/ord.csv` (uncleaned, ORDerly supersedes it). OK?
3. **Breadth** — classify everything but **rank/browse only the coupling families**, `other` collapsed
   **[recommended]** vs. full flat catalogue of all classes.
4. **Condition tolerance** — the §5 family-binning as the default "commonish". Want it looser/tighter?
5. **Deliverable** — parquet + ranked CSV **+ a browsable HTML/Artifact** [recommended], or parquet/CSV
   only.

**Nothing runs until you say go.** Then: write the multi-class detector + source readers + condition
generaliser, a quick local smoke test on 10 k Pistachio rows, then submit the 4 SLURM jobs.
