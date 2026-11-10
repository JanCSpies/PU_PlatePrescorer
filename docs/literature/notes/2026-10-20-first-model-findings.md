> **Provenance.** Summary written by a Claude agent on 2026-10-20 from the Drive literature folder, the notes in this folder and `scratch/count_m0_anchors.py`. Agent summary, not yet checked by Luka. Read a paper before citing a detail.

# First model prototype: findings and options

## Baselines on the Chemlex HATU validation set (1,725 reactions, 32.8 % positive)

Test side not opened. Reproduce with `uv run python scratch/count_m0_anchors.py ~/University/Phd/Dataset/Datasets`.

| baseline | ROC-AUC | hits in top 96 |
|---|---|---|
| random | 0.50 | 0.33 |
| popularity in literature | 0.54 | 0.69 |
| amine class rule | 0.71 | 0.59 |
| Chemlex supervised model (their split, all reagents) | 0.79 | n/a |

## P and the plate

- 1,176 of 1,177 HATU pairs are reported once; 38 % have no yield. The whole class has 28,158 pairs.
- The inventory has 43 % anilines, so a random well is expected near 33 % hits if Chemlex class rates carry over.
- 48 versus 48 wells detect an improvement to 60 % with power 0.85, but not one to 50 %.
- No published model-picked versus random HTE well comparison was found.

## Where the literature agrees

1. Popularity is a required baseline (Beker 2022; Wigh 2024; Taylor 2023).
2. Literature yield regression fails (Liu 2023, R² 0.38); feasibility is predictable (Boiko 2026).
3. Our setting is SAR, not SCAR; the PAYN spy threshold was only tested under SCAR. Model propensity from reporting variables (Bekker 2019).
4. Ranking needs no class prior; bagging PU was validated prospectively in chemistry (Nishii 2025).
5. Split by patent as well as by substrate (Bradshaw 2025); compare with paired bootstrap differences (Mera 2025).

## Options

1. Follow ADR 003 as is.
2. New ADR first: P at two widths, reacting-centre descriptors from the start, success rule fixed before any run (beat both baselines, paired interval, rank within amine class). Chosen by Luka.
3. Cheap diagnostic first: adversarial validation of literature, inventory and Chemlex substrates.

## Open

New dependencies (scikit-learn, catboost) need Luka's agreement. TACIT has no public record.
