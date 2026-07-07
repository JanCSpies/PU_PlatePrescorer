# ADR 004 — Mined C–N coupling positive corpus (ORD) + leakage audit

- **Status:** accepted (autonomous run, 2026-07-08)
- **Context:** WS1 (Florian's #1 priority) — expand the literature positive corpus with C–N couplings
  and prove it does not leak the JnJ test set.

## What we built

`src/bhpu/data/mining.py` (+ `scripts/07_mine_cn_couplings.py`, `slurm/mine_cn.sbatch`) mines
Buchwald–Hartwig-type **aryl C–N couplings** from the open **Open Reaction Database** exports:

- `/mnt/datasets/DynaMAIK/ord.csv` (tag `ORD`) and
- `/mnt/datasets/ORDerly/orderly_{forward,condition}_train.parquet` (tags `ORDerly-forward/-condition`).

**Detector (high precision, not recall):** a reaction counts only if some (aryl-halide, N–H amine)
pair among its molecules yields — via the RDKit BH template — one of the reaction's **own recorded
products** (matched by connectivity InChIKey). This rejects SNAr-elsewhere, reductive aminations,
and mis-parses. Atom-map numbers on some ORD reactants are stripped before storage.

Detected couplings are deduplicated to **unique (aryl, amine) substrate pairs** (the feasibility
unit) across all sources, aggregating max-yield / any-Pd / source provenance / record count.

## Result (run 55113, CPU_Express, 60 workers, 132 s)

| quantity | value |
|---|---|
| raw reactions scanned | 3,801,400 |
| product-confirmed C–N couplings | 124,518 |
| **unique substrate pairs (corpus)** | **35,929** |
| with detectable Pd (`has_pd`) | 2,724 |
| with a reported yield | 14,377 |

Per source (detections): ORD 69,645 · ORDerly-forward 30,098 · ORDerly-condition 24,775. Output:
`datasets/processed/cn_corpus_ord.{parquet,csv}` + `..._provenance.json`.

This expands positives ~**9×** (4,143 literature-HTE → ~40k with ORD) with large substrate diversity
— the key lever for the cross-domain problem (ADR 003).

## Leakage audit vs JnJ (the user's explicit worry: "is the test set inside ORD/Pistachio?")

Every mined pair is audited against the JnJ substrates/pairs via `FrozenEvalProtocol`:

- **`n_exact_jnj_pair_hits = 0`** — **ORD does not contain the JnJ (aryl, amine) test pairs.** The
  JnJ HTE reactions are not present as reactions in ORD.
- **2,221 pairs dropped** for reusing a JnJ *substrate* (aryl or amine) — common building blocks
  that recur in the literature. Dropped to preserve the frozen protocol's substrate-disjointness
  (a mined pair sharing even one JnJ substrate could leak signal into a held-out substrate group).

So the corpus that enters training is **provably free of JnJ substrates and pairs**.

## Scope decisions

- **Pistachio: deferred.** License-flagged (`datasets/raw/Pistachio/location.md` →
  `/mnt/datasets/pistachio`); usable for validation/enrichment but not a publishable open training
  corpus without confirming ToS. ORD (CC-BY-SA) is the open primary.
- **Cernak 50k: not mined for substrate diversity** — ~4 substrates (it is a conditions/negatives
  HTE screen), so it adds little to a *substrate-pair* positive set. Its yield files remain a
  candidate source of real negatives for a specific amine if needed later.
- **BH-only vs all-C–N:** only 2,724/35,929 pairs have detectable Pd (most ORD C–N is SNAr/Cu). We
  keep the full C–N corpus (feasibility = "this bond is formable") with a `has_pd` flag for a
  BH-only ablation. The JnJ test is Pd/BH, so this is a modelling choice to revisit if it hurts.

## How it is used

Featurised with substrate **ECFP** (`load_corpus_features`; the mined corpus has no conditions, so
DRFP is not applicable) and folded into the transfer harness as extra PU positives + source-marginal
mass (`TransferExperiment(extra_positive_features=...)`, `scripts/06 --extra-corpus`). The
"does ORD help transfer?" benchmark is the `ecfp lit_plus_ord` config vs `ecfp lit_only`.
